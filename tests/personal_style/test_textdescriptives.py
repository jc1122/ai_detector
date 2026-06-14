import numpy as np
import pytest


def test_disabled_returns_empty():
    from personal_style_pl.features.textdescriptives_features import TextDescriptivesExtractor
    ext = TextDescriptivesExtractor(enabled=False)
    X = ext.fit_transform(["Ala ma kota."])
    assert X.shape == (1, 0)


def test_missing_dependency_raises_clear_error(monkeypatch):
    import personal_style_pl.features.textdescriptives_features as m
    monkeypatch.setattr(m, "_load_pipe", lambda lang: (_ for _ in ()).throw(ImportError("no td")))
    ext = m.TextDescriptivesExtractor(enabled=True)
    with pytest.raises(RuntimeError, match="TextDescriptives"):
        ext.fit_transform(["Ala ma kota."])


def test_real_extraction():
    pytest.importorskip("textdescriptives")
    import spacy
    if not spacy.util.is_package("pl_nask"):
        pytest.skip("pl_nask model not installed (model-gated; runs only in full env)")
    from personal_style_pl.features.textdescriptives_features import TextDescriptivesExtractor
    ext = TextDescriptivesExtractor(enabled=True)
    X = ext.fit_transform(["Wczoraj poszedłem do urzędu i czekałem w kolejce dość długo, bo "
                           "formularz trzeba było poprawić aż dwa razy."])
    assert X.shape[0] == 1 and X.shape[1] == len(ext.get_feature_names_out())
    assert not np.isnan(X).any()           # NaN -> 0.0 guard works
    feats = dict(zip(ext.get_feature_names_out(), X[0]))
    assert feats["dependency_distance_mean"] > 0   # real parser ran (pl_nask)
