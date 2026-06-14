import json
import subprocess
import sys
import pytest


def test_rich_flag_adds_block(tmp_path):
    pytest.importorskip("lexicalrichness")
    f = tmp_path / "t.txt"
    f.write_text("Warto zauważyć, że to istotne — naprawdę. " * 20, encoding="utf-8")
    out = subprocess.run([sys.executable, "heuristic_detector.py", "--text-file", str(f),
                          "--json", "--rich"], capture_output=True, text=True)
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert "rich_metrics" in payload["experts"]["heuristic"]
    assert "ai_leaning" in payload["experts"]["heuristic"]
