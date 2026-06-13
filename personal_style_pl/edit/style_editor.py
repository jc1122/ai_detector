"""StyleSuggestionEngine + conservative deterministic editor."""

from __future__ import annotations

import numpy as np

from ..features.surface_features import SurfaceFeatureExtractor, SURFACE_FEATURE_NAMES
from ..profile.similarity import score_text
from .rules import EditSuggestion, normalize_whitespace

_PROTECTED_NOTE = ("Never changes names, dates, numbers, legal/medical/financial claims, "
                   "citations, quotes, URLs, or code.")


def _feature(profile, name: str) -> float:
    idx = profile.feature_names.index(name)
    return float(profile.center[idx])


def suggest_edits(profile, text: str) -> list[EditSuggestion]:
    feats = dict(zip(SURFACE_FEATURE_NAMES,
                     SurfaceFeatureExtractor().fit_transform([text])[0]))
    out: list[EditSuggestion] = []

    cand_len = feats["avg_sentence_len_tokens"]
    prof_len = _feature(profile, "avg_sentence_len_tokens")
    if cand_len > prof_len * 1.4 and prof_len > 0:
        out.append(EditSuggestion(
            issue="Long sentences",
            reason=f"Avg sentence {cand_len:.0f} tokens vs your usual {prof_len:.0f}.",
            suggestion="Split the longest sentences at safe punctuation (. ; :).",
            severity="warn"))
    elif cand_len < prof_len * 0.6 and prof_len > 0:
        out.append(EditSuggestion(
            issue="Choppy sentences",
            reason=f"Avg sentence {cand_len:.0f} tokens vs your usual {prof_len:.0f}.",
            suggestion="Merge closely related short sentences."))

    if feats["comma_per_sentence"] < _feature(profile, "comma_per_sentence") * 0.5:
        out.append(EditSuggestion(
            issue="Comma rhythm",
            reason="Comma rate is well below your usual rhythm.",
            suggestion="Review punctuation rhythm — don't blindly add commas."))

    if feats["generic_boilerplate_count"] > max(_feature(profile, "generic_boilerplate_count"), 0) + 0.5:
        out.append(EditSuggestion(
            issue="Boilerplate phrases",
            reason="More generic/boilerplate phrases than your profile.",
            suggestion="Delete or replace generic phrases (e.g. 'warto zauważyć').",
            severity="warn"))

    if feats["transition_phrase_count"] > max(_feature(profile, "transition_phrase_count"), 0) + 0.5:
        out.append(EditSuggestion(
            issue="Transition density",
            reason="More transition phrases than your usual density.",
            suggestion="Reduce or vary transitions."))
    return out


def suggestions_to_markdown(profile, text: str, mode: str) -> str:
    result = score_text(profile, text)
    suggestions = suggest_edits(profile, text)
    lines = [
        "---",
        "machine_assisted_style_edit: true",
        f"profile_used: {profile.profile_id}",
        f"mode: {mode}",
        "---",
        "",
        f"# Style suggestions (score {result.style_match_score}/100, {result.label})",
        "",
        f"_{_PROTECTED_NOTE}_",
        "",
        "## Top divergences",
    ]
    for mm in result.top_mismatches[:5]:
        lines.append(f"- `{mm['feature']}`: {mm['effect']} "
                     f"(you≈{mm['profile_mean']}, draft={mm['candidate_value']})")
    lines.append("")
    lines.append("## Suggested edits")
    if not suggestions:
        lines.append("- No conservative suggestions; the draft is close to your style.")
    for s in suggestions:
        lines.append(f"### {s.issue} ({s.severity})")
        lines.append(f"- Reason: {s.reason}")
        lines.append(f"- Suggestion: {s.suggestion}")
    lines.append("")
    lines.append("## Warnings")
    for w in result.warnings:
        lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def conservative_edit(profile, text: str, mode: str) -> str:
    """Deterministic, meaning-preserving edits. Currently: whitespace normalization.
    Sentence splitting/merging are emitted as SUGGESTIONS only (suggest-edits), not
    applied automatically, to guarantee meaning preservation."""
    return normalize_whitespace(text)
