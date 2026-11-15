"""
Pydantic schemas for structured LLM outputs.

Why Pydantic here specifically: LLMs return text, but our agents need to
pass structured data (arguments + their sources) to each other and to
the UI. Defining schemas up front means we can use LangChain's
`with_structured_output()` to force the LLM to return valid JSON matching
these shapes, instead of us regex-parsing free text (fragile) or hoping
the model formats things consistently (unreliable).

These schemas are the "contract" between agents.
"""
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A single piece of evidence connecting a claim to a specific source."""

    claim: str = Field(description="A short factual claim relevant to the debate topic.")
    evidence: str = Field(description="The supporting text/quote from the retrieved document.")
    source: str = Field(description="The source document name this evidence came from.")
    page: int | None = Field(default=None, description="Page number in the source document, if known.")


class ResearchOutput(BaseModel):
    """Output of the Research/RAG Agent: organized, source-grounded evidence."""

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="List of evidence items extracted from the retrieved documents.",
    )


class ArgumentItem(BaseModel):
    """A single argument made by the PRO or CON agent."""

    argument: str = Field(description="The argument being made.")
    evidence: str | None = Field(default=None, description="Supporting evidence for this argument, if any.")
    source: str | None = Field(default=None, description="Source document for the evidence, if any.")
    page: int | None = Field(default=None, description="Page number for the source, if known.")


class DebateSideOutput(BaseModel):
    """Output of either the PRO or CON debate agent."""

    arguments: list[ArgumentItem] = Field(
        default_factory=list,
        description="List of arguments constructed by this side.",
    )


class RebuttalItem(BaseModel):
    """A single rebuttal responding to a specific opposing claim."""

    opponent_claim: str = Field(description="The opposing argument being responded to.")
    response: str = Field(description="This side's response to the opposing claim.")
    evidence: str | None = Field(default=None, description="Supporting evidence for the rebuttal, if any.")
    source: str | None = Field(default=None, description="Source document for the evidence, if any.")
    page: int | None = Field(default=None, description="Page number for the source, if known.")


class RebuttalOutput(BaseModel):
    """Output of a rebuttal agent (PRO or CON responding to the other side)."""

    rebuttals: list[RebuttalItem] = Field(
        default_factory=list,
        description="List of rebuttals responding to the opposing side's arguments.",
    )


class JudgeOutput(BaseModel):
    """Output of the Judge/Analysis Agent: structured analysis, not a winner score."""

    evidence_analysis: str = Field(description="Assessment of how well each side used the available evidence.")
    pro_unsupported_claims: list[str] = Field(
        default_factory=list, description="PRO claims that appear unsupported by the supplied evidence."
    )
    con_unsupported_claims: list[str] = Field(
        default_factory=list, description="CON claims that appear unsupported by the supplied evidence."
    )
    key_disagreements: list[str] = Field(
        default_factory=list, description="Points where PRO and CON genuinely disagree."
    )
    unresolved_questions: list[str] = Field(
        default_factory=list, description="Questions that cannot be settled from the current evidence."
    )
    overall_analysis: str = Field(description="Overall synthesis of the debate quality and substance.")
