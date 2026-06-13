from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.profile.similarity import score_text


def test_ngram_profile_scores_without_dimension_error():
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę. ") * 4
    docs = [SampleDoc(doc_id=f"d{i}", text=base) for i in range(6)]
    profile = build_profile(docs, use_stylometrix=False, include_ngrams=True,
                            chunk_sentences=2, min_chunk_tokens=15)
    assert profile.ngram_extractor is not None
    result = score_text(profile, base)  # must not raise on hstack/broadcast
    assert 0 <= result.style_match_score <= 100
