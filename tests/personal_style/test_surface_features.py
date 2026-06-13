import numpy as np

from personal_style_pl.features.surface_features import (
    SurfaceFeatureExtractor,
    SURFACE_FEATURE_NAMES,
)

PL = ("Wczoraj poszedłem do urzędu, bo odkładałem wymianę dokumentu. "
      "Kolejka była krótka, ale formularz poprawiałem dwa razy. "
      "Pani cierpliwie pokazała mi błąd.")


def test_feature_names_are_stable_and_complete():
    ext = SurfaceFeatureExtractor().fit([PL])
    assert list(ext.get_feature_names_out()) == list(SURFACE_FEATURE_NAMES)
    assert len(SURFACE_FEATURE_NAMES) == len(set(SURFACE_FEATURE_NAMES))
    for required in ("avg_sentence_len_tokens", "comma_per_sentence",
                     "type_token_ratio", "hapax_ratio", "transition_phrase_count"):
        assert required in SURFACE_FEATURE_NAMES


def test_transform_shape_and_diacritics_counted():
    ext = SurfaceFeatureExtractor()
    X = ext.fit_transform([PL])
    assert isinstance(X, np.ndarray)
    assert X.shape == (1, len(SURFACE_FEATURE_NAMES))
    row = dict(zip(SURFACE_FEATURE_NAMES, X[0]))
    assert row["sentence_count"] == 3
    assert row["comma_per_sentence"] > 0
    assert row["first_person_singular_count"] >= 1  # "mi"


def test_empty_text_is_safe():
    ext = SurfaceFeatureExtractor()
    X = ext.fit_transform([""])
    assert X.shape == (1, len(SURFACE_FEATURE_NAMES))
    assert not np.isnan(X).any()
