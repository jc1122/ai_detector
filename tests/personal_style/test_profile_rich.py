from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.profile.similarity import score_text
from personal_style_pl.features.rich_metrics import RICH_METRIC_NAMES


def _docs():
    base = ("Wczoraj poszedłem do sklepu. Kupiłem chleb. Wróciłem do domu. "
            "Zrobiłem herbatę i usiadłem przy oknie spokojnie. ") * 4
    return [SampleDoc(doc_id=f"d{i}", text=base) for i in range(5)]


def test_profile_includes_rich_by_default_and_scores():
    p = build_profile(_docs(), use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)
    assert p.config["include_rich"] is True
    for name in ("mattr", "burstiness_coeff", "em_dash_per_1k", "boilerplate_per_1k"):
        assert name in p.feature_names
    # scorer must reproduce identical dimensions (no broadcast/hstack error)
    r = score_text(p, " ".join(d.text for d in _docs()[:1]))
    assert 0 <= r.style_match_score <= 100


def test_rich_can_be_disabled():
    p = build_profile(_docs(), use_stylometrix=False, include_rich=False,
                      chunk_sentences=2, min_chunk_tokens=15)
    assert p.config["include_rich"] is False
    assert "mattr" not in p.feature_names
