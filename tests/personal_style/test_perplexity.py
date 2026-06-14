import pytest


def test_missing_model_raises_clear_error(monkeypatch):
    import personal_style_pl.features.perplexity_features as m
    monkeypatch.setattr(m, "_load_model", lambda name: (_ for _ in ()).throw(ImportError("no model")))
    with pytest.raises(RuntimeError, match="perplexity"):
        m.perplexity_features_for_text("Ala ma kota i psa.")


def test_real_perplexity():
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if try_to_load_from_cache("dkleczek/papuGaPT2", "config.json") is None:
        pytest.skip("papuGaPT2 not cached (model-gated; avoids network download in tests)")
    from personal_style_pl.features.perplexity_features import (
        perplexity_features_for_text, PerplexityExtractor, PERPLEXITY_FEATURE_NAMES)
    feats = perplexity_features_for_text(
        "Wczoraj poszedłem do urzędu. Czekałem w kolejce dość długo i nudno.")
    assert feats["lm_perplexity"] > 0
    assert set(PERPLEXITY_FEATURE_NAMES) <= set(feats)
    X = PerplexityExtractor().fit_transform(["Ala ma kota.", "Drugie zdanie tutaj jest tutaj."])
    assert X.shape == (2, len(PERPLEXITY_FEATURE_NAMES))
