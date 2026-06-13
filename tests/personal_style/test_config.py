from personal_style_pl import config


def test_phrase_lists_are_folded_and_nonempty():
    for name in ("TRANSITION_PHRASES", "BOILERPLATE_PHRASES", "HEDGE_PHRASES"):
        phrases = getattr(config, name)
        assert phrases, f"{name} empty"
        for p in phrases:
            # folded: lowercase, no diacritics, no uppercase
            assert p == p.lower()
            assert all(ch not in p for ch in "ąćęłńóśźż")


def test_seeded_from_heuristic_detector():
    # transitions include canonical spec phrases (folded)
    assert "co wiecej" in config.TRANSITION_PHRASES
    assert "ponadto" in config.TRANSITION_PHRASES
    assert "podsumowujac" in config.BOILERPLATE_PHRASES


def test_score_thresholds():
    assert config.LABEL_THRESHOLDS["close_to_my_style"] == 80
    assert config.LABEL_THRESHOLDS["mixed"] == 55
