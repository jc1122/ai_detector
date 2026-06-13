import numpy as np

from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile, StyleProfile


def _docs():
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę. "
            "Potem kupiłem chleb i wróciłem do domu spokojnie. "
            "Wieczorem zapisałem numer sprawy na kartce. ")
    return [SampleDoc(doc_id=f"d{i}", text=base * 3) for i in range(4)]


def test_build_profile_returns_style_profile():
    profile = build_profile(_docs(), use_stylometrix=False, chunk_sentences=2,
                            min_chunk_tokens=20)
    assert isinstance(profile, StyleProfile)
    assert profile.language == "pl"
    assert profile.training_chunk_count >= 4
    assert len(profile.feature_names) == len(profile.center)
    assert len(profile.center) == len(profile.scale)
    assert profile.scale.min() > 0  # no zero scale (floored)


def test_weak_profile_warning_for_tiny_corpus():
    profile = build_profile([SampleDoc(doc_id="d0", text="Krótki tekst tutaj jest.")],
                            use_stylometrix=False, chunk_sentences=8, min_chunk_tokens=120)
    assert any("Profile is weak" in w for w in profile.warnings)
