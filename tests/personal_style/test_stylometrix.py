import numpy as np
import pytest

from personal_style_pl.features.stylometrix_features import StyloMetrixFeatureExtractor


def test_disabled_returns_empty_block():
    ext = StyloMetrixFeatureExtractor(enabled=False)
    X = ext.fit_transform(["Ala ma kota."])
    assert X.shape == (1, 0)
    assert list(ext.get_feature_names_out()) == []


def test_missing_dependency_raises_clear_error_when_enabled(monkeypatch):
    import personal_style_pl.features.stylometrix_features as m
    monkeypatch.setattr(m, "_import_stylo_metrix", lambda: (_ for _ in ()).throw(
        ImportError("no stylo_metrix")))
    ext = StyloMetrixFeatureExtractor(enabled=True)
    with pytest.raises(RuntimeError, match="StyloMetrix"):
        ext.fit_transform(["Ala ma kota."])


def test_real_stylometrix_extraction():
    pytest.importorskip("stylo_metrix")
    try:
        ext = StyloMetrixFeatureExtractor(enabled=True)
        X = ext.fit_transform(["Wczoraj poszedłem do urzędu i czekałem w kolejce."])
    except RuntimeError:
        pytest.skip("StyloMetrix present but pl_nask model unavailable")
    assert X.shape[0] == 1 and X.shape[1] >= 150
    assert len(ext.get_feature_names_out()) == X.shape[1]
