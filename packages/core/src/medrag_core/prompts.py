"""Prompt construction for the Self-MedRAG generator.

The constants reproduce the research-work parity configuration exactly
(`config_selfmedrag_mistral.yaml`). Change them only together with a new
`PROMPT_VERSION`, because accuracy is measured against these prompts.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

PROMPT_VERSION = "research-work-v1"

SYSTEM_PROMPT = (
    "You are a medical AI assistant specialized in evidence-based clinical reasoning.\n"
    "Base ALL claims on provided context. If context doesn't support a claim, "
    'state "insufficient evidence".\n'
    "\n"
    "OUTPUT FORMAT (JSON):\n"
    "{\n"
    '  "answer": "Brief answer",\n'
    '  "rationale": ["Reasoning step 1", "Step 2"],\n'
    '  "confidence": 0.85,\n'
    '  "citations": ["Context used"]\n'
    "}\n"
)

USER_PROMPT_TEMPLATE = (
    "CONTEXT (Retrieved Medical Literature):\n"
    "{context}\n"
    "\n"
    "{history}\n"
    "\n"
    "QUESTION: {query}\n"
    "\n"
    "Provide your answer in the JSON format specified in the system prompt."
)

BINARY_ANSWER_INSTRUCTION = (
    "\n\nCRITICAL CONSTRAINT: This is a binary question. "
    'The "answer" field in your JSON MUST be exactly the single word '
    '"yes" or "no" (lowercase). Do NOT write anything else in the '
    "answer field — no explanations, no qualifications, no punctuation. "
    'Put all reasoning in the "rationale" field.'
)

# Optional (not in the research work): multiple-choice questions must name an option.
# gpt-oss follows the system prompt's "insufficient evidence" rule literally and
# refuses on questions the corpus does not cover; this keeps doubts in the rationale.
MULTIPLE_CHOICE_INSTRUCTION = (
    "\n\nCRITICAL CONSTRAINT: This is a multiple-choice question. "
    'The "answer" field in your JSON MUST be exactly one option letter '
    '(A, B, C, D or E), never "insufficient evidence". If the context does not '
    "settle the question, choose the most likely option using medical knowledge and "
    'say so in the "rationale" field.'
)

# Optional (not in the research work): a one-sentence answer claim the verifier can
# check against the passages, in addition to the rationale.
ANSWER_CLAIM_INSTRUCTION = (
    '\n\nAlso include a "claim" field in your JSON: one sentence stating your final '
    "answer as a factual claim, without mentioning passages or options by letter."
)

# Optional (not in the research work): claims tied to verbatim quotes, so every
# statement shown to users can be traced to a sentence of a retrieved study. Many
# rationale lines were remarks about the passages ("the passages do not report ...")
# that no study can support.
EVIDENCE_QUOTES_INSTRUCTION = (
    '\n\nEVIDENCE RULES: Write the "rationale" as short factual medical claims only; '
    'never describe the passages themselves (no "Passage 2 states ...", no "the passages '
    'do not ..."). Also include an "evidence" list in your JSON with one entry for each '
    'rationale claim that a passage supports: {"claim": "<the claim, exactly as in the '
    'rationale>", "passage": <passage number>, "quote": "<the sentence from that passage '
    'that supports the claim, copied word for word>"}. Claims based on general medical '
    "knowledge rather than a passage get no evidence entry."
)

# With both options, the answer claim needs its own quote to count as supported.
ANSWER_CLAIM_EVIDENCE = (
    ' The "claim" field needs an evidence entry too, with the same text, when a passage '
    "supports it."
)

# Only the most recent attempts are shown to the generator.
MAX_HISTORY_IN_PROMPT = 2


class HistoryEntry(Protocol):
    """What the prompt needs from a previous loop iteration."""

    @property
    def query(self) -> str: ...

    @property
    def answer(self) -> str: ...

    @property
    def support_score(self) -> float: ...


@dataclass(frozen=True)
class ChatMessages:
    system: str
    user: str


def format_context(passages: Sequence[str]) -> str:
    return "\n\n".join(f"[Passage {i + 1}]: {p}" for i, p in enumerate(passages))


def format_history(history: Sequence[HistoryEntry]) -> str:
    if not history:
        return ""
    entries = []
    for i, entry in enumerate(history[-MAX_HISTORY_IN_PROMPT:]):
        entries.append(
            f"Previous Iteration {i + 1}:\n"
            f"Query: {entry.query}\n"
            f"Answer: {entry.answer}\n"
            f"Support Score: {entry.support_score:.2f}\n"
        )
    return f"\nPREVIOUS ATTEMPTS:\n{'\n'.join(entries)}\n"


def build_user_prompt(
    query: str,
    context: Sequence[str],
    history: Sequence[HistoryEntry],
    *,
    binary_answer: bool = False,
    multiple_choice: bool = False,
    answer_claim: bool = False,
    evidence_quotes: bool = False,
    template: str = USER_PROMPT_TEMPLATE,
) -> str:
    """The user turn before stripping: template, then the answer-format constraint if any."""
    prompt = template.format(
        context=format_context(context),
        history=format_history(history),
        query=query,
    )
    if binary_answer:
        prompt += BINARY_ANSWER_INSTRUCTION
    if multiple_choice:
        prompt += MULTIPLE_CHOICE_INSTRUCTION
    if answer_claim:
        prompt += ANSWER_CLAIM_INSTRUCTION
    if evidence_quotes:
        prompt += EVIDENCE_QUOTES_INSTRUCTION
        if answer_claim:
            prompt += ANSWER_CLAIM_EVIDENCE
    return prompt


def build_prompt(
    query: str,
    context: Sequence[str],
    history: Sequence[HistoryEntry],
    *,
    binary_answer: bool = False,
    multiple_choice: bool = False,
    answer_claim: bool = False,
    evidence_quotes: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
    template: str = USER_PROMPT_TEMPLATE,
) -> str:
    """Single-string prompt (system prompt + user turn), as the research work built it."""
    user = build_user_prompt(
        query,
        context,
        history,
        binary_answer=binary_answer,
        multiple_choice=multiple_choice,
        answer_claim=answer_claim,
        evidence_quotes=evidence_quotes,
        template=template,
    )
    return f"{system_prompt}\n\n{user}"


def build_messages(
    query: str,
    context: Sequence[str],
    history: Sequence[HistoryEntry],
    *,
    binary_answer: bool = False,
    multiple_choice: bool = False,
    answer_claim: bool = False,
    evidence_quotes: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
    template: str = USER_PROMPT_TEMPLATE,
) -> ChatMessages:
    """System and user messages exactly as sent to the chat API in the research work."""
    user = build_user_prompt(
        query,
        context,
        history,
        binary_answer=binary_answer,
        multiple_choice=multiple_choice,
        answer_claim=answer_claim,
        evidence_quotes=evidence_quotes,
        template=template,
    )
    return ChatMessages(system=system_prompt.strip(), user=user.strip())
