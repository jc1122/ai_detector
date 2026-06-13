"""Serialize ScoreResult to dict/JSON."""

from __future__ import annotations

from dataclasses import asdict

from .similarity import ScoreResult


def score_result_to_dict(result: ScoreResult) -> dict:
    return asdict(result)


def profile_to_markdown(profile) -> str:
    names = profile.feature_names
    center = {n: float(profile.center[i]) for i, n in enumerate(names)}

    def g(key, default=0.0):
        return center.get(key, default)

    lines = [
        f"# Style profile {profile.profile_id}",
        "",
        f"- Created: {profile.created_at}",
        f"- Trained on: {profile.training_sample_count} samples, "
        f"{profile.training_chunk_count} chunks, {profile.total_tokens} tokens",
        f"- Genres: {', '.join(profile.genres) or 'unspecified'}",
        f"- StyloMetrix used: {profile.config.get('use_stylometrix')}",
        "",
        "## Sentence length habits",
        f"- Avg tokens/sentence: {g('avg_sentence_len_tokens'):.1f} "
        f"(std {g('std_sentence_len_tokens'):.1f}, cv {g('sentence_len_cv'):.2f})",
        "## Paragraph length habits",
        f"- Avg sentences/paragraph: {g('avg_paragraph_len_sentences'):.1f}",
        "## Punctuation habits",
        f"- Commas/sentence: {g('comma_per_sentence'):.2f}; "
        f"dashes/sentence: {g('dash_per_sentence'):.2f}; "
        f"punctuation density: {g('punctuation_density'):.3f}",
        "## Lexical diversity",
        f"- Type-token ratio: {g('type_token_ratio'):.2f}; hapax ratio: {g('hapax_ratio'):.2f}",
        "## Common transitions and function words",
        f"- Transition phrases/chunk: {g('transition_phrase_count'):.2f}; "
        f"clause markers/sentence: {g('average_clause_marker_count'):.2f}",
        "## Formulaic phrase frequency",
        f"- Boilerplate phrases/chunk: {g('generic_boilerplate_count'):.2f}; "
        f"hedges/chunk: {g('hedge_count'):.2f}",
        "## What this profile was trained on",
        f"- {profile.training_sample_count} documents in genres: "
        f"{', '.join(profile.genres) or 'unspecified'}",
        "## Limitations",
    ]
    for w in profile.warnings:
        lines.append(f"- {w}")
    if profile.config.get("use_stylometrix"):
        lines.append("- StyloMetrix feature families included (172 PL features).")
    else:
        lines.append("- StyloMetrix unavailable/disabled: surface features only.")
    return "\n".join(lines) + "\n"
