import numpy as np

from personal_style_pl.features.ngram_features import NgramFeatureExtractor
from personal_style_pl.bridge import attach_heuristics


def test_ngram_preserves_polish_and_fits():
    texts = ["źdźbło trawy źdźbło", "kot pies kot pies", "źdźbło kot pies trawy"]
    ext = NgramFeatureExtractor(min_df=1)
    X = ext.fit_transform(texts)
    assert X.shape[0] == 3
    assert any("ź" in name for name in ext.get_feature_names_out())


def test_attach_heuristics_adds_ai_likeness_block():
    payload = {"style_match_score": 70.0, "label": "mixed"}
    merged = attach_heuristics(payload, "Warto zauważyć, że to jest tekst po polsku. "
                                        "Drugie zdanie tutaj jest również obecne.")
    assert "ai_likeness" in merged
    assert 0.0 <= merged["ai_likeness"]["ai_probability"] <= 1.0
    assert merged["style_match_score"] == 70.0
