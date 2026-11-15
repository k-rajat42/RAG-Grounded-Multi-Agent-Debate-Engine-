"""
AI Debate Arena — Streamlit UI.

This is the entry point: `streamlit run app.py`.

Design choice: instead of calling run_debate() (single blocking call), we
use the compiled graph's .stream() method directly here so we can update
a progress checklist after each node finishes — LangGraph streaming is
what makes "✓ Research completed / ✓ PRO argument generated / ..."
possible without polling or manual node-by-node calls.

Note: pro/con and pro_rebuttal/con_rebuttal run in parallel supersteps
in the underlying graph. app.stream() yields one update dict per
superstep, which may contain more than one node's output when nodes
finish in the same step (e.g. {"pro": ..., "con": ...} together) — the
inner `for node_name, node_output in update.items()` loop below already
handles this, so both checkmarks for a parallel pair will appear back-to-back
in the same UI update rather than one-by-one in strict pipeline order.
"""
import logging

import streamlit as st

from src import config
from src.graph.debate_graph import build_debate_graph
from src.graph.state import DebateState
from src.models.schemas import ResearchOutput, DebateSideOutput, RebuttalOutput, JudgeOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Debate Arena", page_icon="⚖️", layout="centered")

NODE_LABELS = {
    "research": "Research completed",
    "pro": "PRO argument generated",
    "con": "CON argument generated",
    "pro_rebuttal": "PRO rebuttal generated",
    "con_rebuttal": "CON rebuttal generated",
    "judge": "Debate analyzed",
}


def render_evidence(research: ResearchOutput) -> None:
    st.subheader("📚 Research Evidence")
    if not research.evidence:
        st.info("No evidence was retrieved from the knowledge base for this topic.")
        return
    for item in research.evidence:
        with st.expander(item.claim):
            st.write(item.evidence)
            page_str = f", Page: {item.page}" if item.page is not None else ""
            st.caption(f"Source: {item.source}{page_str}")


def render_side(title: str, side: DebateSideOutput) -> None:
    st.subheader(title)
    if not side.arguments:
        st.info("No arguments were generated.")
        return
    for i, arg in enumerate(side.arguments, start=1):
        st.markdown(f"**{i}. {arg.argument}**")
        if arg.evidence:
            st.write(arg.evidence)
        if arg.source:
            page_str = f", Page: {arg.page}" if arg.page is not None else ""
            st.caption(f"Source: {arg.source}{page_str}")


def render_rebuttals(pro_rebuttal: RebuttalOutput, con_rebuttal: RebuttalOutput) -> None:
    st.subheader("↔ Rebuttals")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 PRO responds to CON**")
        if not pro_rebuttal.rebuttals:
            st.info("No rebuttal was generated.")
        for item in pro_rebuttal.rebuttals:
            st.markdown(f"*Opponent claim:* {item.opponent_claim}")
            st.write(item.response)
            if item.source:
                page_str = f", Page: {item.page}" if item.page is not None else ""
                st.caption(f"Source: {item.source}{page_str}")
            st.markdown("---")
    with col2:
        st.markdown("**🔴 CON responds to PRO**")
        if not con_rebuttal.rebuttals:
            st.info("No rebuttal was generated.")
        for item in con_rebuttal.rebuttals:
            st.markdown(f"*Opponent claim:* {item.opponent_claim}")
            st.write(item.response)
            if item.source:
                page_str = f", Page: {item.page}" if item.page is not None else ""
                st.caption(f"Source: {item.source}{page_str}")
            st.markdown("---")


def render_judge(judge: JudgeOutput) -> None:
    st.subheader("⚖️ Judge Analysis")
    st.markdown("**Evidence usage:**")
    st.write(judge.evidence_analysis)

    if judge.pro_unsupported_claims or judge.con_unsupported_claims:
        st.markdown("**Unsupported claims:**")
        if judge.pro_unsupported_claims:
            st.markdown("*PRO:*")
            for claim in judge.pro_unsupported_claims:
                st.write(f"- {claim}")
        if judge.con_unsupported_claims:
            st.markdown("*CON:*")
            for claim in judge.con_unsupported_claims:
                st.write(f"- {claim}")

    if judge.key_disagreements:
        st.markdown("**Key disagreements:**")
        for point in judge.key_disagreements:
            st.write(f"- {point}")

    if judge.unresolved_questions:
        st.markdown("**Unresolved questions:**")
        for question in judge.unresolved_questions:
            st.write(f"- {question}")

    st.markdown("**Overall analysis:**")
    st.write(judge.overall_analysis)


def render_sources(sources: list[str]) -> None:
    st.subheader("📚 Sources")
    if not sources:
        st.info("No sources were cited.")
        return
    for source in sources:
        st.write(f"- {source}")


def run_debate_with_progress(topic: str, top_k: int) -> DebateState:
    """Stream the graph node-by-node, updating a live progress checklist."""
    app = build_debate_graph()
    initial_state: DebateState = {"topic": topic, "top_k": top_k}

    progress_area = st.empty()
    completed: list[str] = []
    final_state: DebateState = dict(initial_state)

    for update in app.stream(initial_state):
        for node_name, node_output in update.items():
            final_state.update(node_output)
            label = NODE_LABELS.get(node_name, node_name)
            completed.append(f"✓ {label}")
            progress_area.markdown("\n".join(completed))

    return final_state


def main() -> None:
    st.title("⚖️ AI Debate Arena")
    st.caption("RAG-powered multi-agent debate, grounded in your documents.")

    with st.form("debate_form"):
        topic = st.text_input(
            "Enter a proposition:",
            placeholder="Should nuclear energy be expanded?",
        )
        st.text("Knowledge Base: using documents in data/documents/")
        top_k = st.number_input(
            "Number of evidence chunks:",
            min_value=1,
            max_value=20,
            value=config.TOP_K,
        )
        submitted = st.form_submit_button("START DEBATE", use_container_width=True)

    if not submitted:
        return

    if not topic or not topic.strip():
        st.error("Please enter a debate proposition before starting.")
        return

    try:
        config.validate_config()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        return

    try:
        with st.spinner("Running the debate..."):
            final_state = run_debate_with_progress(topic.strip(), int(top_k))
    except ValueError as exc:
        st.error(str(exc))
        return
    except RuntimeError as exc:
        st.error(f"The debate failed to complete: {exc}")
        return
    except Exception as exc:
        logger.error("Unexpected error running debate: %s", exc)
        st.error(
            "An unexpected error occurred while running the debate. "
            "Check that your knowledge base is ingested and your API/Ollama connection is working."
        )
        return

    st.divider()
    st.header("Proposition")
    st.write(final_state.get("topic", topic))

    if final_state.get("error"):
        st.warning(final_state["error"])

    st.divider()
    research = final_state.get("research")
    if research:
        render_evidence(research)

    st.divider()
    pro_argument = final_state.get("pro_argument")
    if pro_argument:
        render_side("🟢 PRO Argument", pro_argument)

    st.divider()
    con_argument = final_state.get("con_argument")
    if con_argument:
        render_side("🔴 CON Argument", con_argument)

    st.divider()
    pro_rebuttal = final_state.get("pro_rebuttal")
    con_rebuttal = final_state.get("con_rebuttal")
    if pro_rebuttal and con_rebuttal:
        render_rebuttals(pro_rebuttal, con_rebuttal)

    st.divider()
    judge_analysis = final_state.get("judge_analysis")
    if judge_analysis:
        render_judge(judge_analysis)

    st.divider()
    render_sources(final_state.get("sources", []))


if __name__ == "__main__":
    main()
