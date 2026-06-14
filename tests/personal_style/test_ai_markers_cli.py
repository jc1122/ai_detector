import json
from personal_style_pl.cli import main


def test_ai_markers_cli_json(tmp_path, capsys):
    draft = tmp_path / "d.txt"
    draft.write_text(("Warto zauważyć, że to istotne — naprawdę. " * 20), encoding="utf-8")
    rc = main(["ai-markers", "--text-file", str(draft), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"language", "markers", "warnings", "ood_or_unreliable"} <= set(payload)
