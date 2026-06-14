from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.ai_markers import ai_marker_report, AI_MARKERS


def _profile():
    base = ("Wczoraj poszedłem do sklepu. Kupiłem chleb. Wróciłem do domu. "
            "Zrobiłem herbatę i usiadłem przy oknie spokojnie. ") * 4
    return build_profile([SampleDoc(doc_id=f"d{i}", text=base) for i in range(4)],
                         use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)


def test_report_structure_and_pl_abstain():
    rep = ai_marker_report("Tekst po polsku. " * 30)
    assert "markers" in rep and "warnings" in rep
    assert rep["language"] == "pl"
    assert rep["ood_or_unreliable"] is True
    assert any("not proof of authorship" in w.lower() for w in rep["warnings"])
    assert any("polish" in w.lower() for w in rep["warnings"])


def test_markers_cover_known_signals():
    feats = {m.feature for m in AI_MARKERS}
    for req in ("burstiness_coeff", "em_dash_per_1k", "boilerplate_per_1k", "repeated_4gram_ratio"):
        assert req in feats


def test_overlay_with_profile_flags_dashy_boilerplate():
    profile = _profile()
    dashy = ("Warto zauważyć, że to istotne — naprawdę — dla wszystkich. "
             "Należy podkreślić kluczowe znaczenie — bez wątpienia. ") * 5
    rep = ai_marker_report(dashy, profile=profile)
    assert isinstance(rep["ai_leaning_score"], (int, float))
    flagged = {r["feature"] for r in rep["markers"] if r.get("leaning") == "AI-leaning"}
    assert "boilerplate_per_1k" in flagged or "em_dash_per_1k" in flagged
