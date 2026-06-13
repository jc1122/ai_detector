import csv
import json

from personal_style_pl.cli import main


def _write_samples(tmp_path):
    d = tmp_path / "samples"
    d.mkdir()
    base = ("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
            "Formularz poprawiałem dwa razy, bo pomyliłem datę urodzenia. "
            "Potem kupiłem chleb i wróciłem do domu wieczorem. ") * 4
    for i in range(5):
        (d / f"s{i}.txt").write_text(base, encoding="utf-8")
    return d


def test_build_score_rank_smoke(tmp_path, capsys):
    samples = _write_samples(tmp_path)
    profile = tmp_path / "p.joblib"
    rc = main(["build-profile", "--samples-dir", str(samples),
               "--output", str(profile), "--no-stylometrix",
               "--chunk-sentences", "2", "--min-chunk-tokens", "15"])
    assert rc == 0 and profile.exists()

    draft = tmp_path / "draft.txt"
    draft.write_text(("Wczoraj poszedłem do urzędu i czekałem w kolejce spokojnie. "
                      "Formularz poprawiałem dwa razy. ") * 4, encoding="utf-8")
    rc = main(["score", "--profile", str(profile), "--text-file", str(draft), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"style_match_score", "label", "confidence",
                            "warnings", "summary", "top_matches", "top_mismatches",
                            "chunk_scores"}

    cand = tmp_path / "cands"
    cand.mkdir()
    (cand / "a.txt").write_text(draft.read_text(encoding="utf-8"), encoding="utf-8")
    (cand / "b.txt").write_text("Inny tekst. " * 30, encoding="utf-8")
    ranking = tmp_path / "rank.csv"
    rc = main(["rank", "--profile", str(profile), "--candidates-dir", str(cand),
               "--output", str(ranking)])
    assert rc == 0 and ranking.exists()
    rows = list(csv.DictReader(ranking.open(encoding="utf-8")))
    assert {"filename", "style_match_score", "label", "confidence",
            "word_count", "warnings"} <= set(rows[0])
