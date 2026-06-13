from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.profile.similarity import score_text, ScoreResult


def _profile():
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. "
            "Potem kupiłem chleb i wróciłem do domu wieczorem. "
            "Zapisałem numer sprawy na kartce przy komputerze. ")
    docs = [SampleDoc(doc_id=f"d{i}", text=base * 5) for i in range(6)]
    return build_profile(docs, use_stylometrix=False, chunk_sentences=2, min_chunk_tokens=15)


def test_score_self_is_high():
    profile = _profile()
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. ") * 4
    result = score_text(profile, base)
    assert isinstance(result, ScoreResult)
    assert 0 <= result.style_match_score <= 100
    assert result.label in {"close_to_my_style", "mixed", "far_from_my_style"}
    assert result.style_match_score >= 55


def test_insufficient_text():
    profile = _profile()
    result = score_text(profile, "Za krótkie.")
    assert result.label == "insufficient_text"
    assert any("short" in w.lower() for w in result.warnings)


def test_dissimilar_text_scores_lower_than_self():
    profile = _profile()
    self_text = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
                 "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. ") * 4
    odd = ("WARTO ZAUWAŻYĆ!!! Lista:\n- punkt\n- punkt\n- punkt\n"
           "Tekst — z wieloma — myślnikami — naprawdę — dużo.") * 4
    assert score_text(profile, self_text).style_match_score >= \
        score_text(profile, odd).style_match_score
