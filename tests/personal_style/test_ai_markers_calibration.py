"""v2.1 Polish-register recalibration of the AI-leaning overlay.

Data-driven (audit 2026-06-14, artifacts/doktorat_style): on Polish scientific prose
mattr/em_dash/repeated_4gram are domain-unreliable (polarity flips or typography/term
artifacts) and must NOT count toward the PL score; perplexity is advisory via a threshold.
"""
from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl import ai_markers
from personal_style_pl.ai_markers import (
    ai_marker_report, PL_UNRELIABLE_MARKERS, PL_PERPLEXITY_AI_THRESHOLD,
)

_DASHY = ("Warto zauważyć, że to istotne — naprawdę — dla wszystkich. "
          "Należy podkreślić kluczowe znaczenie — bez wątpienia. ") * 5


def _pl_profile():
    base = ("Wczoraj poszedłem do sklepu. Kupiłem chleb. Wróciłem do domu. "
            "Zrobiłem herbatę i usiadłem przy oknie spokojnie. ") * 4
    return build_profile([SampleDoc(doc_id=f"d{i}", text=base) for i in range(4)],
                         use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)


def test_pl_unreliable_set_documented():
    assert {"mattr", "em_dash_per_1k", "repeated_4gram_ratio"} <= set(PL_UNRELIABLE_MARKERS)


def test_pl_unreliable_markers_advisory_only():
    rep = ai_marker_report(_DASHY, profile=_pl_profile())
    by = {r["feature"]: r for r in rep["markers"]}
    for f in ("em_dash_per_1k", "mattr", "repeated_4gram_ratio"):
        assert by[f]["counted"] is False
        assert by[f]["leaning"] == "advisory_only"
    # A reliable marker stays counted and can flag.
    assert by["boilerplate_per_1k"]["counted"] is True


def test_pl_score_counts_only_reliable_markers():
    rep = ai_marker_report(_DASHY, profile=_pl_profile())
    counted = [r for r in rep["markers"] if r.get("counted")]
    assert counted, "expected at least one counted marker"
    assert all(r["feature"] not in PL_UNRELIABLE_MARKERS for r in counted)
    assert rep["ai_leaning_score"] is None or 0 <= rep["ai_leaning_score"] <= 100


def test_english_input_counts_em_dash():
    english = ("This is a long English passage — with many em-dashes — and it should "
               "count em-dash overuse because that is a genuine English AI tell. ") * 6
    rep = ai_marker_report(english, profile=_pl_profile())
    assert rep["language"] == "en"
    by = {r["feature"]: r for r in rep["markers"]}
    assert by["em_dash_per_1k"]["counted"] is True


def test_perplexity_flag_advisory(monkeypatch):
    monkeypatch.setattr(
        ai_markers, "_all_metrics",
        lambda text, wp: {**ai_markers.rich_metrics_for_text(text),
                          "lm_perplexity": 20.0, "sent_perplexity_cv": 0.5})
    rep = ai_marker_report("Tekst po polsku. " * 30, profile=_pl_profile(),
                           with_perplexity=True)
    assert "perplexity_flag" in rep
    assert rep["perplexity_flag"]["threshold"] == PL_PERPLEXITY_AI_THRESHOLD
    assert rep["perplexity_flag"]["flag"] == "leans_AI_low_perplexity"
    assert rep["ood_or_unreliable"] is True  # advisory; PL still abstains


def test_perplexity_flag_human_range(monkeypatch):
    monkeypatch.setattr(
        ai_markers, "_all_metrics",
        lambda text, wp: {**ai_markers.rich_metrics_for_text(text),
                          "lm_perplexity": 140.0, "sent_perplexity_cv": 1.0})
    rep = ai_marker_report("Tekst po polsku. " * 30, profile=_pl_profile(),
                           with_perplexity=True)
    assert rep["perplexity_flag"]["flag"] == "within_or_above_human_range"


def test_no_perplexity_flag_without_flag_requested():
    rep = ai_marker_report("Tekst po polsku. " * 30, profile=_pl_profile())
    assert "perplexity_flag" not in rep
