"""Configurable Polish phrase lists and scoring thresholds.

Phrases are stored folded (lowercase, diacritics stripped) so matching against
heuristic_detector._fold_text(text) is consistent. These are STYLE indicators,
not proof of AI authorship.
"""

from __future__ import annotations

from heuristic_detector import _fold_text, AI_PHRASES_PL, AI_WORDS_PL

_TRANSITIONS_RAW = (
    "co więcej", "ponadto", "poza tym", "z jednej strony", "z drugiej strony",
    "jednocześnie", "tym samym", "w efekcie", "w rezultacie", "dlatego",
)
_BOILERPLATE_RAW = (
    "warto zauważyć", "należy podkreślić", "trzeba pamiętać", "kluczowe znaczenie",
    "istotnym elementem", "dynamicznie zmieniającym się", "w dzisiejszych czasach",
    "podsumowując",
)
_HEDGES_RAW = (
    "wydaje się", "można powiedzieć", "raczej", "prawdopodobnie", "być może",
    "w pewnym sensie",
)
_FIRST_PERSON_SINGULAR_RAW = ("ja", "mnie", "mi", "mną", "moim", "moja", "moje", "mój")
_FIRST_PERSON_PLURAL_RAW = ("my", "nas", "nam", "nami", "nasz", "nasza", "nasze", "naszych")
# Clause markers (subordinators/conjunctions) — folded.
_CLAUSE_MARKERS_RAW = (
    "że", "który", "która", "które", "ponieważ", "gdyż", "aby", "żeby", "jeśli",
    "jeżeli", "gdy", "kiedy", "chociaż", "mimo", "dlatego", "więc",
)


def _fold_all(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        folded = _fold_text(item)
        if folded and folded not in seen:
            seen.append(folded)
    return tuple(seen)


# Seed transitions/boilerplate from both the spec lists and heuristic_detector.AI_PHRASES_PL.
TRANSITION_PHRASES = _fold_all(_TRANSITIONS_RAW)
BOILERPLATE_PHRASES = _fold_all(_BOILERPLATE_RAW)
HEDGE_PHRASES = _fold_all(_HEDGES_RAW)
FIRST_PERSON_SINGULAR = _fold_all(_FIRST_PERSON_SINGULAR_RAW)
FIRST_PERSON_PLURAL = _fold_all(_FIRST_PERSON_PLURAL_RAW)
CLAUSE_MARKERS = _fold_all(_CLAUSE_MARKERS_RAW)

# Extra AI-marker phrases available to the bridge (already folded upstream).
AI_MARKER_PHRASES = tuple(AI_PHRASES_PL)
AI_MARKER_WORDS = tuple(AI_WORDS_PL)

# Chunking defaults.
DEFAULT_CHUNK_SENTENCES = 8
DEFAULT_MIN_CHUNK_TOKENS = 120

# Scoring.
Z_CLIP = 8.0
BLEND_Z = 0.7
BLEND_COSINE = 0.3
LABEL_THRESHOLDS = {"close_to_my_style": 80, "mixed": 55}
MIN_CANDIDATE_TOKENS = 40

WEAK_PROFILE_WARNING = (
    "Profile is weak: provide at least 10–20 writing samples or 5,000+ words."
)
STYLE_NOT_AUTHORSHIP_WARNING = (
    "This is style similarity, not proof of authorship."
)
