"""Tests for the fast local heuristic detector."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import heuristic_detector


POLISH_TEMPLATE_AI_MARKERS = (
    "Warto zauważyć, że w dzisiejszym świecie sztuczna inteligencja odgrywa "
    "kluczową rolę w edukacji. W kontekście dynamicznie rozwijających się "
    "technologii należy podkreślić, że tego typu rozwiązania pozwalają na "
    "efektywne wspieranie uczniów. Co więcej, kompleksowe podejście otwiera "
    "nowe możliwości dla nauczycieli i studentów. Podsumowując, jest to "
    "niezwykle istotny aspekt nowoczesnego kształcenia."
)

POLISH_PLAIN_HUMANISH = (
    "Wczoraj po pracy poszedłem do urzędu, bo od tygodnia odkładałem wymianę "
    "dokumentu. Kolejka była krótka, ale formularz musiałem poprawiać dwa razy. "
    "Pani w okienku cierpliwie pokazała mi błąd. Potem kupiłem chleb, wróciłem "
    "do domu i zapisałem sobie numer sprawy na kartce przy komputerze."
)

ENGLISH_TEMPLATE_AI_MARKERS = (
    "It is worth noting that artificial intelligence plays a crucial role in "
    "today's digital landscape. In this context, a comprehensive and holistic "
    "approach can streamline workflows and unlock new opportunities. On the "
    "other hand, organizations should navigate these changes carefully. In "
    "conclusion, this transformative technology is a testament to innovation."
)


class HeuristicAnalysisTests(unittest.TestCase):
    def test_polish_ai_marker_sample_matches_original_high_score_shape(self) -> None:
        result = heuristic_detector.analyze_text(POLISH_TEMPLATE_AI_MARKERS)

        self.assertEqual(result["language"], "pl")
        self.assertEqual(result["ai_probability_percent"], 99)
        self.assertGreaterEqual(result["metrics"]["ai_phrase_count"], 10)
        self.assertGreater(result["metrics"]["ai_word_density"], 10)

    def test_plain_polish_sample_stays_low_like_original_smoke(self) -> None:
        result = heuristic_detector.analyze_text(POLISH_PLAIN_HUMANISH)

        self.assertEqual(result["language"], "pl")
        self.assertLessEqual(result["ai_probability_percent"], 35)
        self.assertEqual(result["metrics"]["ai_phrase_count"], 0)

    def test_english_ai_marker_sample_matches_original_high_score_shape(self) -> None:
        result = heuristic_detector.analyze_text(ENGLISH_TEMPLATE_AI_MARKERS)

        self.assertEqual(result["language"], "en")
        self.assertEqual(result["ai_probability_percent"], 99)
        self.assertGreaterEqual(result["metrics"]["ai_phrase_count"], 7)
        self.assertGreater(result["metrics"]["ai_word_density"], 10)

    def test_short_text_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as exc_info:
            heuristic_detector.analyze_text("To jest bardzo krótki tekst.")

        self.assertIn("at least 10 words", str(exc_info.exception))

    def test_payload_uses_operator_contract_sections(self) -> None:
        payload = heuristic_detector.build_payload(POLISH_TEMPLATE_AI_MARKERS, threshold=0.5)

        for key in ("text_preview", "weights", "experts", "ensemble", "calibration", "device"):
            self.assertIn(key, payload)
        self.assertEqual(payload["weights"], {"heuristic": 1.0})
        self.assertEqual(set(payload["experts"]), {"heuristic"})
        heuristic = payload["experts"]["heuristic"]
        for key in ("ai_score", "human_score", "ai_probability", "human_probability", "chunks", "loaded"):
            self.assertIn(key, heuristic)
        self.assertIn("site_metrics", heuristic)
        site_metrics = heuristic["site_metrics"]
        self.assertEqual(site_metrics["variation"]["label"], "Zmienność tekstu")
        self.assertEqual(
            site_metrics["variation"]["ai_probability_percent"],
            100 - heuristic["categories"]["variation"]["score"],
        )
        self.assertIn("Odch. std. zdań", site_metrics["variation"]["detail"])
        self.assertIn("Sygnatury AI", site_metrics["signatures"]["label"])
        self.assertEqual(payload["ensemble"]["label"], "ai")
        self.assertFalse(payload["calibration"]["calibrated"])


class HeuristicCLITests(unittest.TestCase):
    def test_cli_json_text_mode(self) -> None:
        with patch("sys.stdout", new=io.StringIO()) as stdout:
            heuristic_detector.main(["--json", "--text", POLISH_TEMPLATE_AI_MARKERS])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "heuristic")
        self.assertEqual(payload["ensemble"]["ai_probability"], 0.99)

    def test_cli_text_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            path = Path(workspace) / "input.txt"
            path.write_text(POLISH_PLAIN_HUMANISH, encoding="utf-8")

            with patch("sys.stdout", new=io.StringIO()) as stdout:
                heuristic_detector.main(["--json", "--text-file", str(path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["experts"]["heuristic"]["language"], "pl")
        self.assertLessEqual(payload["ensemble"]["ai_probability"], 0.35)

    def test_cli_stdin_mode(self) -> None:
        with patch("sys.stdin", new=io.StringIO(ENGLISH_TEMPLATE_AI_MARKERS)), patch(
            "sys.stdout", new=io.StringIO()
        ) as stdout:
            heuristic_detector.main(["--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["experts"]["heuristic"]["language"], "en")
        self.assertEqual(payload["ensemble"]["ai_probability"], 0.99)


if __name__ == "__main__":
    unittest.main()
