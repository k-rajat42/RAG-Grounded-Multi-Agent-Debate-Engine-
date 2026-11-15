"""
LangGraph debate workflow.

Wires the six agent functions built in Steps 5-7 into an orchestrated
graph matching the architecture diagram:

    topic -> research -> [pro, con] (parallel) -> [pro_rebuttal, con_rebuttal] (parallel) -> judge

research runs first and its output feeds both pro and con, which have no
dependency on each other and so run concurrently. Once both finish,
pro_rebuttal and con_rebuttal both become eligible -- each needs the
OPPOSING side's argument, so both wait on pro AND con before running,
then run concurrently with each other. judge waits on both rebuttals.

Each function below is a "node": it takes the current DebateState, calls
the corresponding agent, and returns a partial state update (LangGraph
merges this into the running state). Nodes only depend on state fields
that earlier nodes have already populated, matching the diagram's edges.
"""
import logging

from langgraph.graph import StateGraph, START, END

from src.agents.research_agent import run_research_agent
from src.agents.pro_agent import run_pro_agent
from src.agents.con_agent import run_con_agent
from src.agents.rebuttal_agent import run_pro_rebuttal, run_con_rebuttal
from src.agents.judge_agent import run_judge_agent
from src.graph.state import DebateState
from src.models.schemas import ResearchOutput, DebateSideOutput, RebuttalOutput

logger = logging.getLogger(__name__)


def research_node(state: DebateState) -> dict:
    """Run research, degrading gracefully instead of crashing the graph.

    research_node is the single point of failure for the whole parallel
    graph below it (pro, con, both rebuttals, and judge all read from
    its output). If run_research_agent() raises for any reason, catch
    it here, fall back to an empty ResearchOutput, and record a
    user-facing message in state["error"] so the UI can surface that
    the debate is running ungrounded rather than erroring out entirely.
    """
    logger.info("Graph: research node started")
    update: dict = {}
    try:
        research = run_research_agent(state["topic"], top_k=state.get("top_k"))
    except Exception as exc:
        logger.error("Research node failed entirely: %s", exc)
        research = ResearchOutput(evidence=[])
        update["error"] = (
            "Evidence retrieval failed, so this debate is running without "
            "grounding in your knowledge base. Arguments below may be generic "
            "or unsupported."
        )

    sources = sorted({
        f"{item.source}, Page {item.page}" if item.page is not None else item.source
        for item in research.evidence
        if item.source
    })
    update["research"] = research
    update["sources"] = sources
    return update


def pro_node(state: DebateState) -> dict:
    logger.info("Graph: PRO node started")
    research = state.get("research", ResearchOutput(evidence=[]))
    pro_argument = run_pro_agent(state["topic"], research)
    return {"pro_argument": pro_argument}


def con_node(state: DebateState) -> dict:
    logger.info("Graph: CON node started")
    research = state.get("research", ResearchOutput(evidence=[]))
    con_argument = run_con_agent(state["topic"], research)
    return {"con_argument": con_argument}


def pro_rebuttal_node(state: DebateState) -> dict:
    logger.info("Graph: PRO rebuttal node started")
    research = state.get("research", ResearchOutput(evidence=[]))
    con_argument = state.get("con_argument", DebateSideOutput(arguments=[]))
    pro_rebuttal = run_pro_rebuttal(state["topic"], con_argument, research)
    return {"pro_rebuttal": pro_rebuttal}


def con_rebuttal_node(state: DebateState) -> dict:
    logger.info("Graph: CON rebuttal node started")
    research = state.get("research", ResearchOutput(evidence=[]))
    pro_argument = state.get("pro_argument", DebateSideOutput(arguments=[]))
    con_rebuttal = run_con_rebuttal(state["topic"], pro_argument, research)
    return {"con_rebuttal": con_rebuttal}


def judge_node(state: DebateState) -> dict:
    logger.info("Graph: judge node started")
    judge_analysis = run_judge_agent(
        topic=state["topic"],
        research=state.get("research", ResearchOutput(evidence=[])),
        pro_argument=state.get("pro_argument", DebateSideOutput(arguments=[])),
        con_argument=state.get("con_argument", DebateSideOutput(arguments=[])),
        pro_rebuttal=state.get("pro_rebuttal", RebuttalOutput(rebuttals=[])),
        con_rebuttal=state.get("con_rebuttal", RebuttalOutput(rebuttals=[])),
    )
    logger.info("Graph: debate completed")
    return {"judge_analysis": judge_analysis}


def build_debate_graph():
    """Construct and compile the LangGraph debate workflow.

    Graph shape (matches the architecture diagram):
        START -> research
        research -> pro, con                      (fan-out, run in parallel)
        pro, con -> pro_rebuttal, con_rebuttal     (fan-in then fan-out;
                                                     each rebuttal node waits
                                                     for BOTH pro and con,
                                                     since each rebuttal reads
                                                     the opposing side's
                                                     argument, then the two
                                                     rebuttals run in parallel)
        pro_rebuttal, con_rebuttal -> judge        (fan-in)
        judge -> END

    LangGraph runs nodes that share a predecessor concurrently, and
    automatically waits for all incoming edges into a node to complete
    before running it (Pregel/superstep model) -- no manual join node needed.
    """
    graph = StateGraph(DebateState)

    graph.add_node("research", research_node)
    graph.add_node("pro", pro_node)
    graph.add_node("con", con_node)
    graph.add_node("pro_rebuttal", pro_rebuttal_node)
    graph.add_node("con_rebuttal", con_rebuttal_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "research")

    # Fan out: pro and con don't depend on each other, only on research
    graph.add_edge("research", "pro")
    graph.add_edge("research", "con")

    # Fan in: each rebuttal needs the opposing side's argument, so both
    # rebuttal nodes wait until BOTH pro and con have finished
    graph.add_edge("pro", "pro_rebuttal")
    graph.add_edge("con", "pro_rebuttal")
    graph.add_edge("pro", "con_rebuttal")
    graph.add_edge("con", "con_rebuttal")

    # Fan out then in again: rebuttals don't depend on each other,
    # but judge needs both
    graph.add_edge("pro_rebuttal", "judge")
    graph.add_edge("con_rebuttal", "judge")

    graph.add_edge("judge", END)

    return graph.compile()


def run_debate(topic: str) -> DebateState:
    """Run the full debate workflow for a given topic and return the final state."""
    if not topic or not topic.strip():
        raise ValueError("Debate topic must not be empty.")

    app = build_debate_graph()
    initial_state: DebateState = {"topic": topic}
    final_state = app.invoke(initial_state)
    return final_state
