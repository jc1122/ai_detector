import joblib

from personal_style_pl.io import SampleDoc
from personal_style_pl.profile.build_profile import build_profile
from personal_style_pl.cli import main


def _profile(tmp_path):
    docs = [SampleDoc(doc_id=f"d{i}",
                      text=("Krótkie zdanie. Drugie zdanie tutaj. " * 8))
            for i in range(4)]
    profile = build_profile(docs, use_stylometrix=False, chunk_sentences=2,
                            min_chunk_tokens=10)
    p = tmp_path / "p.joblib"
    joblib.dump(profile, p)
    return p


def test_suggest_edits_writes_markdown_with_metadata(tmp_path):
    p = _profile(tmp_path)
    draft = tmp_path / "draft.txt"
    draft.write_text("Warto zauważyć, że to zdanie jest dość długie i zawiera wiele "
                     "fragmentów połączonych przecinkami, co bywa męczące dla czytelnika "
                     "i odbiega od zwykłego stylu pisania autora w tym profilu.",
                     encoding="utf-8")
    out = tmp_path / "sug.md"
    rc = main(["suggest-edits", "--profile", str(p), "--text-file", str(draft),
               "--output", str(out), "--mode", "light"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "machine_assisted_style_edit: true" in text
    assert "mode: light" in text


def test_edit_preserves_numbers_urls_and_normalizes_whitespace(tmp_path):
    p = _profile(tmp_path)
    draft = tmp_path / "draft.txt"
    draft.write_text("Zobacz https://example.com/x oraz 12 345 zł.\n\n\nKoniec   tekstu.",
                     encoding="utf-8")
    out = tmp_path / "edited.txt"
    rc = main(["edit", "--profile", str(p), "--text-file", str(draft),
               "--output", str(out), "--mode", "light"])
    assert rc == 0
    edited = out.read_text(encoding="utf-8")
    assert "https://example.com/x" in edited
    assert "12 345" in edited
    assert "   " not in edited  # collapsed runs of spaces
