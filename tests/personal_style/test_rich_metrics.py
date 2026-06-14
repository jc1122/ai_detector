import numpy as np
from personal_style_pl.features.rich_metrics import (
    rich_metrics_for_text, RichMetricsExtractor, RICH_METRIC_NAMES,
)

PLAIN = ("Wczoraj poszedłem do sklepu. Kupiłem chleb i mleko. Wróciłem do domu. "
         "Zrobiłem herbatę i usiadłem przy oknie. ") * 3
DASHY = ("Sztuczna inteligencja — co warto wiedzieć — odgrywa rolę. "
         "Warto zauważyć, że to istotne — naprawdę istotne — dla wszystkich. ") * 3


def test_names_stable_and_complete():
    assert list(RichMetricsExtractor().fit([PLAIN]).get_feature_names_out()) == list(RICH_METRIC_NAMES)
    for req in ("mattr", "mtld", "burstiness_coeff", "em_dash_per_1k", "repeated_4gram_ratio"):
        assert req in RICH_METRIC_NAMES


def test_densities_length_normalized_and_emdash_detected():
    plain = rich_metrics_for_text(PLAIN)
    dashy = rich_metrics_for_text(DASHY)
    assert dashy["em_dash_per_1k"] > plain["em_dash_per_1k"]   # em-dash tell
    assert dashy["boilerplate_per_1k"] >= 0.0
    assert 0.0 <= plain["mattr"] <= 1.0


def test_empty_text_safe():
    X = RichMetricsExtractor().fit_transform([""])
    assert X.shape == (1, len(RICH_METRIC_NAMES))
    assert not np.isnan(X).any()
