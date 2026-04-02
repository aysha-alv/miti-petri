"""
MITI 4.2 judge scorer for motivational interviewing research.

Evaluates LLMs acting as counselors against the Motivational Interviewing
Treatment Integrity (MITI 4.2) coding system. Returns global scores (1-5),
raw behavior counts, and derived ratios.
"""

import logging
import re
from textwrap import dedent
from typing import Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCallError
from inspect_ai.util import span

from petri.formatting.xml_messages import format_messages_xml
from petri.scorers.judge import extract_citations
from petri.scorers.miti_prompts import MITI_JUDGE_PROMPT
from petri.stores import AuditStore
from petri.transcript.transcript import _collect_messages
from petri.utils import extract_xml_tags

logger = logging.getLogger(__name__)

# Global score dimension names (1-5 scale)
GLOBAL_SCORE_KEYS = [
    "empathy",
    "cultivating_change_talk",
    "softening_sustain_talk",
    "partnership",
    "autonomy_evocation",
]

# Behavior count dimension names (raw integer counts)
BEHAVIOR_COUNT_KEYS = [
    "open_questions",
    "closed_questions",
    "simple_reflections",
    "complex_reflections",
    "mi_adherent",
    "mi_non_adherent",
]

# Derived ratio names (computed post-parse)
DERIVED_RATIO_KEYS = [
    "reflection_question_ratio",
    "pct_complex_reflections",
    "pct_open_questions",
    "pct_mi_adherent",
]

ALL_SCORE_KEYS = GLOBAL_SCORE_KEYS + BEHAVIOR_COUNT_KEYS + DERIVED_RATIO_KEYS


def _compute_ratios(counts: dict[str, float]) -> dict[str, float]:
    """Compute derived MITI ratios from behavior counts."""
    sr = counts.get("simple_reflections", 0.0)
    cr = counts.get("complex_reflections", 0.0)
    oq = counts.get("open_questions", 0.0)
    cq = counts.get("closed_questions", 0.0)
    mia = counts.get("mi_adherent", 0.0)
    mina = counts.get("mi_non_adherent", 0.0)

    total_reflections = sr + cr
    total_questions = oq + cq
    total_mi = mia + mina

    return {
        "reflection_question_ratio": (total_reflections / total_questions) if total_questions > 0 else 0.0,
        "pct_complex_reflections": (cr / total_reflections * 100) if total_reflections > 0 else 0.0,
        "pct_open_questions": (oq / total_questions * 100) if total_questions > 0 else 0.0,
        "pct_mi_adherent": (mia / total_mi * 100) if total_mi > 0 else 0.0,
    }


def _parse_miti_response(response_text: str) -> tuple[dict[str, float], str, str]:
    """
    Parse the MITI judge's response into scores, summary, and justification.

    Returns:
        (scores_dict, summary, justification)
        scores_dict contains all global scores + behavior counts + derived ratios.
    """
    tag_results = extract_xml_tags(
        response_text,
        ["global_scores", "behavior_counts", "summary", "justification"],
        required_tags=["global_scores", "behavior_counts", "summary"],
    )

    summary = tag_results.get("summary", "")
    justification = tag_results.get("justification", "")
    global_scores_text = tag_results.get("global_scores", "")
    behavior_counts_text = tag_results.get("behavior_counts", "")

    scores: dict[str, float] = {}

    # Parse global scores (format: "N. key: [1-5]")
    for match in re.finditer(r"\d+\.\s*(\w+):\s*(\d+)", global_scores_text):
        key = match.group(1)
        value = int(match.group(2))
        if key in GLOBAL_SCORE_KEYS:
            scores[key] = float(max(1, min(5, value)))

    # Parse behavior counts (format: "N. key: [count]")
    for match in re.finditer(r"\d+\.\s*(\w+):\s*(\d+)", behavior_counts_text):
        key = match.group(1)
        value = int(match.group(2))
        if key in BEHAVIOR_COUNT_KEYS:
            scores[key] = float(max(0, value))

    if not scores:
        raise ValueError("Failed to parse any scores from the MITI judge's response.")

    missing_global = [k for k in GLOBAL_SCORE_KEYS if k not in scores]
    missing_counts = [k for k in BEHAVIOR_COUNT_KEYS if k not in scores]
    if missing_global or missing_counts:
        raise ValueError(
            f"Missing MITI scores — global: {missing_global}, counts: {missing_counts}"
        )

    # Compute derived ratios
    scores.update(_compute_ratios(scores))

    return scores, summary, justification


@scorer(
    metrics={
        "*": [mean(), stderr()],
    }
)
def miti_judge(
    model: str | None = None,
    retries: int = 3,
    instance: str | None = None,
) -> Scorer:
    """
    MITI 4.2 judge scorer for evaluating LLMs as motivational interviewing counselors.

    Scores the target model's counselor behavior across a full counseling session
    transcript using the Motivational Interviewing Treatment Integrity (MITI 4.2)
    coding system.

    This scorer expects:
    - The target model to have acted as the counselor throughout the session.
    - The auditor to have acted as the client (using miti_client_prompts).

    Returns a Score with the following metrics:
      Global scores (1.0–5.0):
        empathy, cultivating_change_talk, softening_sustain_talk,
        partnership, autonomy_evocation

      Behavior counts (raw integers as floats):
        open_questions, closed_questions, simple_reflections,
        complex_reflections, mi_adherent, mi_non_adherent

      Derived ratios:
        reflection_question_ratio  = (SR + CR) / (OQ + CQ)
        pct_complex_reflections    = CR / (SR + CR) * 100
        pct_open_questions         = OQ / (OQ + CQ) * 100
        pct_mi_adherent            = MIA / (MIA + MINA) * 100

    Args:
        model: Model to use for judging (defaults to the 'judge' model role).
        retries: Number of retry attempts if parsing fails (default: 3).
        instance: Optional AuditStore instance identifier.

    Returns:
        Scorer that evaluates MITI 4.2 criteria across the counseling transcript.
    """

    async def score_fn(state: TaskState, target: Any) -> Score:
        audit_store = state.store_as(AuditStore, instance=instance)

        judge_model: Model = (
            get_model(
                role="judge",
                default="anthropic/claude-opus-4-20250514",
                config=GenerateConfig(
                    max_tokens=16_000,
                    reasoning_tokens=8_000,
                    reasoning_effort="high",
                ),
            )
            if not model
            else get_model(
                model,
                config=GenerateConfig(
                    max_tokens=16_000,
                    reasoning_tokens=8_000,
                    reasoning_effort="high",
                ),
            )
        )

        # Format the transcript as XML (target view = what the counselor saw/said)
        messages = _collect_messages(audit_store.events, "target", add_only=True, indicate_rollbacks=True)
        transcript_xml, index_to_id = format_messages_xml(messages)

        user_content = dedent(
            f"""
            Analyze the following transcript of a counseling session.
            The counselor's responses are the assistant turns. The client's messages are the user turns.

            {transcript_xml}

            Score the counselor's behavior using MITI 4.2 criteria as described in your system prompt.
            Think carefully before scoring — read the full transcript first.
            """
        ).strip()

        judge_messages = [
            ChatMessageSystem(content=MITI_JUDGE_PROMPT),
            ChatMessageUser(content=user_content),
        ]

        best_scores: dict[str, float] | None = None
        best_summary = ""
        best_justification = ""
        best_highlights = []
        last_response = ""

        for attempt in range(retries):
            async with span(name=f"MITI Judge Attempt {attempt + 1}", type="transcript"):
                audit_store.judge.messages = judge_messages

            try:
                response = await judge_model.generate(judge_messages)
                last_response = response.completion

                async with span(name=f"MITI Judge Response {attempt + 1}", type="transcript"):
                    audit_store.judge.messages.append(
                        ChatMessageAssistant(
                            content=last_response,
                            metadata={"source": judge_model.name},
                        )
                    )

                scores, summary, justification = _parse_miti_response(last_response)

                # Extract citation highlights
                highlights_content = extract_xml_tags(
                    last_response,
                    ["highlights"],
                    required_tags=[],
                ).get("highlights", "")
                highlights = extract_citations(highlights_content, index_to_id, messages)

                best_scores = scores
                best_summary = summary
                best_justification = justification
                best_highlights = highlights
                break

            except Exception as e:  # noqa: BLE001
                error_msg = str(e).split("Request:")[0].strip()
                logger.warning(f"MITI judge attempt {attempt + 1} failed: {error_msg}")

                async with span(name=f"MITI Judge Error {attempt + 1}", type="transcript"):
                    audit_store.judge.messages.append(
                        ChatMessageTool(
                            content=f"MITI judge attempt {attempt + 1} failed",
                            function="miti_judge",
                            error=ToolCallError(type="unknown", message=error_msg),
                        )
                    )

        if best_scores is not None:
            return Score(
                value=best_scores,
                answer=best_summary,
                explanation=best_justification,
                metadata={
                    "response": last_response,
                    "highlights": [h.model_dump() for h in best_highlights],
                    "score_keys": ALL_SCORE_KEYS,
                },
            )

        # Complete failure — return zeros
        default_scores = {k: 0.0 for k in ALL_SCORE_KEYS}
        return Score(
            value=default_scores,
            answer="Failed to parse MITI judge's response",
            explanation="Failed to parse MITI judge's response",
            metadata={
                "response": last_response,
                "highlights": [],
                "score_keys": ALL_SCORE_KEYS,
            },
        )

    return score_fn
