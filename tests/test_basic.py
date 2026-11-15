"""
Basic tests for the ingestion pipeline.

Run with: pytest tests/test_basic.py -v

Note: test_build_vector_store_and_full_ingestion requires a valid
GOOGLE_API_KEY in .env (it makes real embedding API calls). The other
tests (loading, splitting) do not need any API key.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.rag.ingestion import load_pdfs, split_documents, ingest_documents
from src.rag.retriever import retrieve_evidence, format_evidence_for_prompt, get_vector_store
from src import config
from langchain_core.documents import Document


def test_load_pdfs_returns_documents_with_metadata():
    pages = load_pdfs()
    assert len(pages) > 0
    for page in pages:
        assert "source" in page.metadata
        assert "document" in page.metadata
        assert "page" in page.metadata
        assert page.page_content.strip() != ""


def test_load_pdfs_raises_on_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_pdfs(documents_dir="data/does_not_exist")


def test_split_documents_produces_chunks():
    pages = load_pdfs()
    chunks = split_documents(pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= len(pages)
    # Metadata should survive the split.
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata


@pytest.mark.skipif(not config.GOOGLE_API_KEY, reason="Requires GOOGLE_API_KEY for embeddings")
def test_build_vector_store_and_full_ingestion():
    vector_store = ingest_documents()
    results = vector_store.similarity_search("nuclear energy emissions", k=2)
    assert len(results) > 0
    assert "page" in results[0].metadata


def test_format_evidence_for_prompt_includes_source_and_page():
    chunks = [
        Document(page_content="Nuclear plants produce low-carbon electricity.",
                  metadata={"document": "ipcc_ar6.pdf", "page": 12}),
    ]
    formatted = format_evidence_for_prompt(chunks)
    assert "ipcc_ar6.pdf" in formatted
    assert "Page: 12" in formatted
    assert "Nuclear plants produce low-carbon electricity." in formatted


def test_format_evidence_for_prompt_handles_empty_results():
    formatted = format_evidence_for_prompt([])
    assert "No evidence" in formatted


def test_retrieve_evidence_raises_on_empty_query():
    with pytest.raises(ValueError):
        retrieve_evidence("   ")


def test_schemas_validate_correct_data():
    from src.models.schemas import EvidenceItem, ResearchOutput, JudgeOutput

    evidence = EvidenceItem(claim="Nuclear is low-carbon", evidence="...", source="ipcc_ar6.pdf", page=12)
    research = ResearchOutput(evidence=[evidence])
    assert len(research.evidence) == 1
    assert research.evidence[0].page == 12

    judge = JudgeOutput(
        evidence_analysis="Both sides used evidence.",
        overall_analysis="Solid debate.",
    )
    # Optional list fields default to empty, not required.
    assert judge.pro_unsupported_claims == []


def test_schemas_reject_missing_required_field():
    from src.models.schemas import EvidenceItem
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvidenceItem(evidence="...", source="x.pdf")  # missing required "claim"


def test_schemas_json_roundtrip():
    from src.models.schemas import JudgeOutput

    judge = JudgeOutput(evidence_analysis="a", overall_analysis="b", key_disagreements=["x"])
    parsed = JudgeOutput.model_validate_json(judge.model_dump_json())
    assert parsed == judge


def test_build_research_prompt_includes_topic_and_evidence():
    from src.agents.research_agent import build_research_prompt

    prompt = build_research_prompt("Should nuclear energy be expanded?", "[1] Source: ipcc.pdf, Page: 12\nSample text.")
    assert "Should nuclear energy be expanded?" in prompt
    assert "ipcc.pdf" in prompt


def test_research_agent_raises_on_empty_topic():
    from src.agents.research_agent import run_research_agent

    with pytest.raises(ValueError):
        run_research_agent("   ")


def test_research_agent_full_pipeline_mocked():
    """Mocks retrieval + LLM so this test needs no API key, but still
    exercises the real orchestration logic in run_research_agent."""
    from unittest.mock import patch, MagicMock
    from langchain_core.documents import Document
    from src.agents.research_agent import run_research_agent
    from src.models.schemas import ResearchOutput, EvidenceItem

    fake_chunks = [Document(page_content="Nuclear supplies 10% of electricity.",
                              metadata={"document": "ipcc.pdf", "page": 12})]
    fake_output = ResearchOutput(evidence=[
        EvidenceItem(claim="Nuclear supplies 10% of global electricity", evidence="...", source="ipcc.pdf", page=12)
    ])

    with patch("src.agents.research_agent.retrieve_evidence", return_value=fake_chunks):
        with patch("src.agents.research_agent.get_llm") as mock_get_llm:
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = fake_output
            mock_get_llm.return_value.with_structured_output.return_value = mock_structured

            result = run_research_agent("Should nuclear energy be expanded?")

    assert isinstance(result, ResearchOutput)
    assert len(result.evidence) == 1
    assert result.evidence[0].source == "ipcc.pdf"


def test_research_agent_empty_evidence_short_circuits():
    from unittest.mock import patch
    from src.agents.research_agent import run_research_agent

    with patch("src.agents.research_agent.retrieve_evidence", return_value=[]):
        result = run_research_agent("Some obscure topic with no matching documents")
    assert result.evidence == []


@pytest.mark.skipif(not config.GOOGLE_API_KEY, reason="Requires GOOGLE_API_KEY for LLM call")
def test_research_agent_produces_grounded_evidence():
    from src.agents.research_agent import run_research_agent

    result = run_research_agent("Should nuclear energy be expanded?")
    assert len(result.evidence) > 0
    for item in result.evidence:
        assert item.source  # must cite a real source, never blank


@pytest.mark.skipif(not config.GOOGLE_API_KEY, reason="Requires GOOGLE_API_KEY for embeddings")
def test_retrieve_evidence_returns_documents():
    vector_store = get_vector_store()
    results = retrieve_evidence("nuclear energy costs", top_k=3, vector_store=vector_store)
    assert len(results) > 0
    for doc in results:
        assert "document" in doc.metadata
        assert "page" in doc.metadata


def _sample_research_output():
    from src.models.schemas import ResearchOutput, EvidenceItem

    return ResearchOutput(evidence=[
        EvidenceItem(claim="Nuclear supplies 10% of global electricity", evidence="...", source="ipcc.pdf", page=12),
        EvidenceItem(claim="Construction cost overruns average 120%", evidence="...", source="ipcc.pdf", page=45),
    ])


def test_pro_prompt_argues_in_favor_and_cites_evidence():
    from src.agents.pro_agent import build_pro_prompt

    prompt = build_pro_prompt("Should nuclear energy be expanded?", _sample_research_output())
    assert "IN FAVOR" in prompt
    assert "ipcc.pdf" in prompt


def test_con_prompt_argues_against_and_cites_evidence():
    from src.agents.con_agent import build_con_prompt

    prompt = build_con_prompt("Should nuclear energy be expanded?", _sample_research_output())
    assert "AGAINST" in prompt
    assert "ipcc.pdf" in prompt


def test_pro_and_con_prompts_handle_empty_evidence():
    from src.agents.pro_agent import build_pro_prompt
    from src.models.schemas import ResearchOutput

    prompt = build_pro_prompt("Some topic", ResearchOutput(evidence=[]))
    assert "No evidence" in prompt


def test_pro_agent_raises_on_empty_topic():
    from src.agents.pro_agent import run_pro_agent

    with pytest.raises(ValueError):
        run_pro_agent("   ", _sample_research_output())


def test_con_agent_raises_on_empty_topic():
    from src.agents.con_agent import run_con_agent

    with pytest.raises(ValueError):
        run_con_agent("   ", _sample_research_output())


def test_pro_agent_pipeline_mocked():
    from unittest.mock import patch, MagicMock
    from src.agents.pro_agent import run_pro_agent
    from src.models.schemas import DebateSideOutput, ArgumentItem

    fake_output = DebateSideOutput(arguments=[
        ArgumentItem(argument="Nuclear cuts emissions", evidence="...", source="ipcc.pdf", page=1)
    ])
    with patch("src.agents.pro_agent.get_llm") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_output
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        result = run_pro_agent("Should nuclear energy be expanded?", _sample_research_output())

    assert isinstance(result, DebateSideOutput)
    assert result.arguments[0].argument == "Nuclear cuts emissions"


def test_con_agent_pipeline_mocked():
    from unittest.mock import patch, MagicMock
    from src.agents.con_agent import run_con_agent
    from src.models.schemas import DebateSideOutput, ArgumentItem

    fake_output = DebateSideOutput(arguments=[
        ArgumentItem(argument="Costs overrun budgets", evidence="...", source="ipcc.pdf", page=45)
    ])
    with patch("src.agents.con_agent.get_llm") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_output
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        result = run_con_agent("Should nuclear energy be expanded?", _sample_research_output())

    assert isinstance(result, DebateSideOutput)
    assert result.arguments[0].argument == "Costs overrun budgets"


def _sample_debate_side(argument_text: str):
    from src.models.schemas import DebateSideOutput, ArgumentItem

    return DebateSideOutput(arguments=[ArgumentItem(argument=argument_text, source="ipcc.pdf", page=1)])


def test_rebuttal_prompt_targets_opponent_argument():
    from src.agents.rebuttal_agent import build_rebuttal_prompt

    con_args = _sample_debate_side("Nuclear costs overrun budgets")
    prompt = build_rebuttal_prompt(
        "Should nuclear energy be expanded?", "PRO", "CON", con_args, _sample_research_output()
    )
    assert "Nuclear costs overrun budgets" in prompt
    assert "PRO" in prompt and "CON" in prompt


def test_rebuttal_prompt_handles_empty_opponent_arguments():
    from src.agents.rebuttal_agent import build_rebuttal_prompt
    from src.models.schemas import DebateSideOutput

    prompt = build_rebuttal_prompt("X", "CON", "PRO", DebateSideOutput(arguments=[]), _sample_research_output())
    assert "no arguments were made" in prompt


def test_pro_rebuttal_raises_on_empty_topic():
    from src.agents.rebuttal_agent import run_pro_rebuttal

    with pytest.raises(ValueError):
        run_pro_rebuttal("   ", _sample_debate_side("x"), _sample_research_output())


def test_con_rebuttal_raises_on_empty_topic():
    from src.agents.rebuttal_agent import run_con_rebuttal

    with pytest.raises(ValueError):
        run_con_rebuttal("   ", _sample_debate_side("x"), _sample_research_output())


def test_pro_rebuttal_pipeline_mocked():
    from unittest.mock import patch, MagicMock
    from src.agents.rebuttal_agent import run_pro_rebuttal
    from src.models.schemas import RebuttalOutput, RebuttalItem

    fake_output = RebuttalOutput(rebuttals=[
        RebuttalItem(opponent_claim="Nuclear costs overrun budgets",
                     response="Costs have declined with standardized designs",
                     evidence="...", source="ipcc.pdf", page=1)
    ])
    with patch("src.agents.rebuttal_agent.get_llm") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_output
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        result = run_pro_rebuttal(
            "Should nuclear energy be expanded?",
            _sample_debate_side("Nuclear costs overrun budgets"),
            _sample_research_output(),
        )

    assert isinstance(result, RebuttalOutput)
    assert result.rebuttals[0].opponent_claim == "Nuclear costs overrun budgets"


def test_con_rebuttal_pipeline_mocked():
    from unittest.mock import patch, MagicMock
    from src.agents.rebuttal_agent import run_con_rebuttal
    from src.models.schemas import RebuttalOutput, RebuttalItem

    fake_output = RebuttalOutput(rebuttals=[
        RebuttalItem(opponent_claim="Nuclear cuts emissions",
                     response="True, but waste storage remains unresolved",
                     evidence="...", source="ipcc.pdf", page=1)
    ])
    with patch("src.agents.rebuttal_agent.get_llm") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_output
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        result = run_con_rebuttal(
            "Should nuclear energy be expanded?",
            _sample_debate_side("Nuclear cuts emissions"),
            _sample_research_output(),
        )

    assert isinstance(result, RebuttalOutput)
    assert result.rebuttals[0].opponent_claim == "Nuclear cuts emissions"


def test_judge_prompt_includes_all_debate_components():
    from src.agents.judge_agent import build_judge_prompt
    from src.models.schemas import RebuttalOutput, RebuttalItem

    pro_rebuttal = RebuttalOutput(rebuttals=[
        RebuttalItem(opponent_claim="Costly to build", response="Costs falling", source="ipcc.pdf", page=1)
    ])
    con_rebuttal = RebuttalOutput(rebuttals=[
        RebuttalItem(opponent_claim="Cuts emissions", response="Waste unresolved", source="ipcc.pdf", page=1)
    ])
    prompt = build_judge_prompt(
        "Should nuclear energy be expanded?",
        _sample_research_output(),
        _sample_debate_side("Cuts emissions"),
        _sample_debate_side("Costly to build"),
        pro_rebuttal,
        con_rebuttal,
    )
    assert "Cuts emissions" in prompt
    assert "Costly to build" in prompt
    assert "Costs falling" in prompt
    assert "do not declare a winner" in prompt.lower() or "Do not declare a winner" in prompt


def test_judge_agent_raises_on_empty_topic():
    from src.agents.judge_agent import run_judge_agent
    from src.models.schemas import DebateSideOutput, RebuttalOutput

    with pytest.raises(ValueError):
        run_judge_agent("   ", _sample_research_output(), _sample_debate_side("x"),
                         _sample_debate_side("y"), RebuttalOutput(rebuttals=[]), RebuttalOutput(rebuttals=[]))


def test_judge_agent_pipeline_mocked():
    from unittest.mock import patch, MagicMock
    from src.agents.judge_agent import run_judge_agent
    from src.models.schemas import JudgeOutput, RebuttalOutput

    fake_output = JudgeOutput(evidence_analysis="Both sides cited sources.", overall_analysis="Substantive debate.")
    with patch("src.agents.judge_agent.get_llm") as mock_llm:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_output
        mock_llm.return_value.with_structured_output.return_value = mock_structured
        result = run_judge_agent(
            "Should nuclear energy be expanded?", _sample_research_output(),
            _sample_debate_side("x"), _sample_debate_side("y"),
            RebuttalOutput(rebuttals=[]), RebuttalOutput(rebuttals=[]),
        )

    assert isinstance(result, JudgeOutput)
    assert result.overall_analysis == "Substantive debate."


def test_debate_graph_compiles_with_expected_structure():
    from src.graph.debate_graph import build_debate_graph

    app = build_debate_graph()
    g = app.get_graph()
    node_names = set(g.nodes.keys())
    expected = {"__start__", "research", "pro", "con", "pro_rebuttal", "con_rebuttal", "judge", "__end__"}
    assert expected.issubset(node_names)

    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("__start__", "research") in edge_pairs
    assert ("research", "pro") in edge_pairs
    assert ("pro", "con") in edge_pairs
    assert ("con", "pro_rebuttal") in edge_pairs
    assert ("pro_rebuttal", "con_rebuttal") in edge_pairs
    assert ("con_rebuttal", "judge") in edge_pairs
    assert ("judge", "__end__") in edge_pairs


def test_debate_graph_raises_on_empty_topic():
    from src.graph.debate_graph import run_debate

    with pytest.raises(ValueError):
        run_debate("   ")


def test_retrieve_evidence_uses_runtime_config_top_k():
    """Regression test: top_k must be read from config.TOP_K at CALL time,
    not bound as a stale function default at import time."""
    from unittest.mock import MagicMock

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = []

    config.TOP_K = 3
    retrieve_evidence("some topic", vector_store=fake_store)
    assert fake_store.similarity_search.call_args.kwargs["k"] == 3

    config.TOP_K = 15
    retrieve_evidence("some topic", vector_store=fake_store)
    assert fake_store.similarity_search.call_args.kwargs["k"] == 15

    config.TOP_K = 5  # restore default for other tests


def test_debate_graph_streams_nodes_in_order_and_forwards_top_k():
    from unittest.mock import patch
    from src.graph.debate_graph import build_debate_graph
    from src.models.schemas import (
        ResearchOutput, EvidenceItem, DebateSideOutput, ArgumentItem,
        RebuttalOutput, RebuttalItem, JudgeOutput,
    )

    fake_research = ResearchOutput(evidence=[EvidenceItem(claim="x", evidence="y", source="ipcc.pdf", page=1)])
    fake_side = DebateSideOutput(arguments=[ArgumentItem(argument="a", source="ipcc.pdf", page=1)])
    fake_rebuttal = RebuttalOutput(rebuttals=[RebuttalItem(opponent_claim="a", response="b", source="ipcc.pdf", page=1)])
    fake_judge = JudgeOutput(evidence_analysis="x", overall_analysis="y")

    captured = {}

    def fake_research_agent(topic, top_k=None):
        captured["top_k"] = top_k
        return fake_research

    with patch("src.graph.debate_graph.run_research_agent", side_effect=fake_research_agent), \
         patch("src.graph.debate_graph.run_pro_agent", return_value=fake_side), \
         patch("src.graph.debate_graph.run_con_agent", return_value=fake_side), \
         patch("src.graph.debate_graph.run_pro_rebuttal", return_value=fake_rebuttal), \
         patch("src.graph.debate_graph.run_con_rebuttal", return_value=fake_rebuttal), \
         patch("src.graph.debate_graph.run_judge_agent", return_value=fake_judge):

        app = build_debate_graph()
        initial_state = {"topic": "Should nuclear energy be expanded?", "top_k": 8}

        node_order = []
        for update in app.stream(initial_state):
            node_order.extend(update.keys())

    assert node_order == ["research", "pro", "con", "pro_rebuttal", "con_rebuttal", "judge"]
    assert captured["top_k"] == 8


def test_llm_provider_switches_to_ollama():
    """Verifies LLM_PROVIDER=ollama routes to ChatOllama/OllamaEmbeddings
    instead of Gemini classes, without requiring a running Ollama server."""
    import importlib
    from unittest.mock import patch
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    with patch("src.config.LLM_PROVIDER", "ollama"):
        import src.llm
        import src.rag.embeddings
        importlib.reload(src.llm)
        importlib.reload(src.rag.embeddings)

        llm = src.llm.get_llm()
        emb = src.rag.embeddings.get_embedding_model()

        assert isinstance(llm, ChatOllama)
        assert isinstance(emb, OllamaEmbeddings)

    # Reload back to restore default (gemini) provider for subsequent tests.
    importlib.reload(src.llm)
    importlib.reload(src.rag.embeddings)


def test_validate_config_rejects_unknown_provider():
    import importlib
    from unittest.mock import patch

    with patch("src.config.LLM_PROVIDER", "bogus"):
        with pytest.raises(ValueError):
            config.validate_config()


def test_debate_graph_full_execution_mocked():
    """Runs the ENTIRE graph START->END with all 6 agent calls mocked.

    This is the key test for Step 9 (test complete workflow): it proves
    state actually flows correctly between nodes -- in particular, that
    PRO's rebuttal responds to CON's argument (not its own), and vice
    versa, which is easy to get backwards when wiring a graph by hand.
    """
    from unittest.mock import patch
    from src.graph.debate_graph import run_debate
    from src.models.schemas import (
        ResearchOutput, EvidenceItem, DebateSideOutput, ArgumentItem,
        RebuttalOutput, RebuttalItem, JudgeOutput,
    )

    fake_research = ResearchOutput(evidence=[
        EvidenceItem(claim="Nuclear supplies 10% of electricity", evidence="...", source="ipcc.pdf", page=12)
    ])
    fake_pro = DebateSideOutput(arguments=[ArgumentItem(argument="Cuts emissions", source="ipcc.pdf", page=12)])
    fake_con = DebateSideOutput(arguments=[ArgumentItem(argument="Costly to build", source="ipcc.pdf", page=45)])
    fake_pro_rebuttal = RebuttalOutput(rebuttals=[
        RebuttalItem(opponent_claim="Costly to build", response="Costs falling", source="ipcc.pdf", page=12)
    ])
    fake_con_rebuttal = RebuttalOutput(rebuttals=[
        RebuttalItem(opponent_claim="Cuts emissions", response="Waste unresolved", source="ipcc.pdf", page=45)
    ])
    fake_judge = JudgeOutput(evidence_analysis="Both sides cited sources.", overall_analysis="Substantive debate.")

    with patch("src.graph.debate_graph.run_research_agent", return_value=fake_research), \
         patch("src.graph.debate_graph.run_pro_agent", return_value=fake_pro), \
         patch("src.graph.debate_graph.run_con_agent", return_value=fake_con), \
         patch("src.graph.debate_graph.run_pro_rebuttal", return_value=fake_pro_rebuttal), \
         patch("src.graph.debate_graph.run_con_rebuttal", return_value=fake_con_rebuttal), \
         patch("src.graph.debate_graph.run_judge_agent", return_value=fake_judge):

        final_state = run_debate("Should nuclear energy be expanded?")

    expected_fields = {"topic", "research", "pro_argument", "con_argument",
                        "pro_rebuttal", "con_rebuttal", "judge_analysis", "sources"}
    assert expected_fields.issubset(final_state.keys())

    # Critical: verify each rebuttal targets the OPPONENT's argument, not its own side's.
    assert final_state["pro_rebuttal"].rebuttals[0].opponent_claim == "Costly to build"
    assert final_state["con_rebuttal"].rebuttals[0].opponent_claim == "Cuts emissions"
    assert "ipcc.pdf, Page 12" in final_state["sources"]
    assert final_state["judge_analysis"].overall_analysis == "Substantive debate."
