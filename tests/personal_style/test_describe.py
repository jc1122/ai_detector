import joblib

from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.cli import main


def test_describe_profile_writes_markdown(tmp_path):
    docs = [SampleDoc(doc_id=f"d{i}",
                      text=("Zdanie jedno tutaj jest. Zdanie dwa tutaj jest. " * 8))
            for i in range(4)]
    profile = build_profile(docs, use_stylometrix=False, chunk_sentences=2,
                            min_chunk_tokens=10)
    p = tmp_path / "p.joblib"
    joblib.dump(profile, p)
    out = tmp_path / "summary.md"
    rc = main(["describe-profile", "--profile", str(p), "--output", str(out)])
    assert rc == 0 and out.exists()
    text = out.read_text(encoding="utf-8")
    for heading in ("Sentence length", "Punctuation", "Lexical diversity", "Limitations"):
        assert heading in text
