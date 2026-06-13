import csv
import json

from personal_style_pl.io import load_samples_from_dir, load_samples_from_csv, SampleDoc
from personal_style_pl.utils.json import dump_json


def test_load_samples_from_dir(tmp_path):
    (tmp_path / "a.txt").write_text("Pierwszy tekst. Łódź.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Drugi tekst po polsku.", encoding="utf-8")
    docs = load_samples_from_dir(tmp_path)
    assert len(docs) == 2
    assert all(isinstance(d, SampleDoc) for d in docs)
    assert {d.doc_id for d in docs} == {"a", "b"}
    assert "Łódź" in next(d.text for d in docs if d.doc_id == "a")


def test_load_samples_from_csv(tmp_path):
    path = tmp_path / "s.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "source", "date", "genre"])
        w.writerow(["Mój tekst pierwszy.", "blog", "2024-01-01", "essay"])
        w.writerow(["Mój tekst drugi.", "email", "2024-02-10", "email"])
    docs = load_samples_from_csv(path, text_col="text")
    assert len(docs) == 2
    assert docs[0].genre == "essay"


def test_dump_json_preserves_polish(tmp_path):
    p = tmp_path / "o.json"
    dump_json({"k": "źdźbło"}, p)
    assert "źdźbło" in p.read_text(encoding="utf-8")
    assert json.loads(p.read_text(encoding="utf-8"))["k"] == "źdźbło"
