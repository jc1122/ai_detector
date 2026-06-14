#!/usr/bin/env python3
"""Fast local heuristic scorer for AI-like text markers."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


PL_FUNCTION_WORDS = {
    "i",
    "w",
    "na",
    "z",
    "do",
    "nie",
    "sie",
    "to",
    "jak",
    "ale",
    "ze",
    "jest",
    "za",
    "co",
    "po",
    "od",
    "o",
    "tak",
    "ten",
    "tym",
    "tego",
    "tej",
    "jego",
    "jej",
    "ich",
    "byc",
    "moze",
    "tylko",
    "juz",
    "jeszcze",
    "tez",
    "bardzo",
    "gdzie",
    "kiedy",
    "ktory",
    "ktora",
    "ktore",
    "bo",
    "czy",
    "przez",
    "przy",
    "dla",
    "bez",
    "nad",
    "pod",
    "przed",
    "miedzy",
}

EN_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "from",
    "by",
    "of",
    "and",
    "or",
    "but",
    "if",
    "then",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "not",
    "no",
}

AI_PHRASES_PL = (
    "warto zauwazyc",
    "warto podkreslic",
    "nalezy podkreslic",
    "nalezy zauwazyc",
    "warto wspomniec",
    "warto dodac",
    "warto zwrocic uwage",
    "warto pamietac",
    "w kontekscie",
    "w dzisiejszym swiecie",
    "w dobie",
    "w erze cyfrowej",
    "w erze sztucznej inteligencji",
    "w obliczu",
    "w swietle powyzszych",
    "kluczowy aspekt",
    "kluczowym aspektem",
    "kluczowe znaczenie",
    "kluczowe zastrzezenie",
    "odgrywa kluczowa role",
    "odgrywa istotna role",
    "odgrywa wazna role",
    "nie jest to tajemnica",
    "nie jest tajemnica",
    "nie ulega watpliwosci",
    "nie mozna zapominac",
    "nie mozna tez zapominac",
    "nie mozna pominac",
    "nie sposob pominac",
    "trudno przecenic",
    "nie sposob nie docenic",
    "podsumowujac",
    "reasumujac",
    "konkludujac",
    "przede wszystkim",
    "w pierwszej kolejnosci",
    "na samym poczatku",
    "z drugiej strony",
    "z jednej strony",
    "majac na uwadze",
    "biorac pod uwage",
    "w gruncie rzeczy",
    "w rezultacie",
    "co wiecej",
    "ponadto",
    "dodatkowo",
    "w tym kontekscie",
    "co istotne",
    "co wazne",
    "co ciekawe",
    "niezwykle istotne",
    "niezwykle wazne",
    "szczegolnie istotne",
    "szczegolnie wazne",
    "dynamicznie rozwijaj",
    "dynamicznie zmieniaj",
    "stanowi fundament",
    "stanowi podstawe",
    "stanowi klucz",
    "jest to szczegolnie",
    "jest to niezwykle",
    "jest to o tyle istotne",
    "wieloaspektowy",
    "wielowymiarowy",
    "wieloplaszczyznowy",
    "tego typu rozwiazania",
    "tego rodzaju podejscie",
    "pozwala na efektywne",
    "pozwalaja na efektywne",
    "umozliwia skuteczne",
    "pozwala na lepsze",
    "otwiera drzwi",
    "otwiera mozliwosci",
    "otwiera nowe mozliwosci",
    "w znacznym stopniu",
    "w duzej mierze",
    "w pewnym sensie",
    "to nie tylko",
    "nie tylko przyjemnosc",
    "nie tylko kwestia",
    "oczywiscie pod warunkiem",
    "pod warunkiem ze",
    "o ile nie",
    "na uwage zasluguje",
    "istotnym aspektem",
    "istotny aspekt",
    "istotne jest to",
    "dzieki czemu",
    "dzieki temu",
    "w efekcie czego",
    "potrafi zdzialac",
    "moze pozytywnie wplywac",
    "pozytywnie wplywac na",
    "kazdy znajdzie",
    "kazdy moze",
    "kazdy z nas",
    "dobrze udokumentowane",
    "powszechnie wiadomo",
    "jak wiadomo",
    "jednoczesnie warto",
    "jednoczesnie nalezy",
)

AI_WORDS_PL = (
    "kluczowy",
    "istotny",
    "fundamentalny",
    "kompleksowy",
    "holistyczny",
    "innowacyjny",
    "przelomowy",
    "rewolucyjny",
    "transformacyjny",
    "wieloaspektowy",
    "wielowymiarowy",
    "wieloplaszczyznowy",
    "niewatpliwie",
    "bezsprzecznie",
    "niezaprzeczalnie",
    "efektywny",
    "optymalny",
    "zintegrowany",
    "strategiczny",
    "dynamiczny",
    "synergiczny",
    "transparentny",
    "niezastapiony",
    "subiektywny",
    "metaboliczny",
    "aromatyczny",
    "intensywny",
    "sprzymierzeniec",
    "aspekt",
    "kontekst",
    "implementacja",
    "optymalizacja",
    "dedykowany",
    "skoncentrowany",
    "ukierunkowany",
    "jednoznaczny",
    "wyspecjalizowany",
    "komplementarny",
)

AI_PHRASES_EN = (
    "it's worth noting",
    "it is worth noting",
    "it's important to note",
    "it is important to note",
    "in today's",
    "in the realm of",
    "in this context",
    "in conclusion",
    "in summary",
    "in essence",
    "a testament to",
    "deep dive",
    "game changer",
    "treasure trove",
    "unique blend",
    "ever-evolving",
    "cutting-edge",
    "state-of-the-art",
    "it should be noted",
    "on the other hand",
    "having said that",
    "at the end of the day",
    "when it comes to",
    "in terms of",
    "the fact that",
    "it goes without saying",
    "needless to say",
    "serves as a",
    "plays a crucial role",
    "plays a vital role",
    "holistic approach",
)

AI_WORDS_EN = (
    "delve",
    "underscore",
    "meticulous",
    "commendable",
    "showcase",
    "intricate",
    "tapestry",
    "symphony",
    "realm",
    "prowess",
    "noteworthy",
    "groundbreaking",
    "leverage",
    "unveil",
    "pivotal",
    "bolster",
    "holistic",
    "elevate",
    "unwavering",
    "transformative",
    "enigma",
    "embark",
    "invaluable",
    "testament",
    "nuance",
    "mitigate",
    "multifaceted",
    "navigate",
    "unravel",
    "streamline",
    "intersection",
    "harness",
    "seamless",
    "foster",
    "comprehensive",
    "burgeon",
    "aptly",
    "demystify",
    "spearhead",
    "underpin",
)


def _fold_text(text: str) -> str:
    text = text.replace("ł", "l").replace("Ł", "L")
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", text)
        if len(sentence.strip()) > 3
    ]


def _extract_words(text: str) -> list[str]:
    cleaned = re.sub(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s'-]+", " ", text.casefold())
    return [word for word in re.split(r"\s+", cleaned) if word]


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph for paragraph in re.split(r"\n\s*\n", text) if len(paragraph.strip()) > 20]


def _stddev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sigmoid(value: float, center: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (value - center)))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def detect_language(text: str) -> str:
    pl_chars = len(re.findall(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", text))
    words = [_fold_text(word) for word in _extract_words(text)]
    pl_function_count = sum(1 for word in words[:100] if word in PL_FUNCTION_WORDS)
    return "pl" if pl_chars > 2 or pl_function_count > 5 else "en"


def _char_entropy(text: str) -> float:
    chars = [char.casefold() for char in text if not char.isspace()]
    if not chars:
        return 0.0
    counts = Counter(chars)
    total = len(chars)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _word_bigram_entropy(words: list[str]) -> float:
    if len(words) < 3:
        return 0.0
    bigrams = Counter(f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1))
    total = sum(bigrams.values())
    return -sum((count / total) * math.log2(count / total) for count in bigrams.values())


def _sentence_length_variation(sentences: list[str]) -> dict[str, object]:
    lengths = [len(_extract_words(sentence)) for sentence in sentences]
    return {
        "std_dev": _stddev(lengths),
        "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "lengths": lengths,
    }


def _type_token_ratio(words: list[str]) -> float:
    sample = words[:200]
    return len(set(sample)) / len(sample) if sample else 0.0


def _hapax_ratio(words: list[str]) -> float:
    counts = Counter(words)
    if not counts:
        return 0.0
    return sum(1 for count in counts.values() if count == 1) / len(counts)


def _max_ngram_frequency(words: list[str], size: int) -> int:
    if len(words) < size + 1:
        return 0
    counts = Counter(tuple(words[index : index + size]) for index in range(len(words) - size + 1))
    return max(counts.values(), default=0)


def _detect_ai_phrases(text: str, lang: str) -> list[dict[str, object]]:
    folded = _fold_text(text)
    phrases = AI_PHRASES_PL if lang == "pl" else AI_PHRASES_EN
    found: list[dict[str, object]] = []
    for phrase in phrases:
        start = 0
        while True:
            index = folded.find(phrase, start)
            if index == -1:
                break
            found.append({"phrase": phrase, "index": index, "length": len(phrase), "type": "phrase"})
            start = index + len(phrase)
    return found


def _detect_ai_words(words: list[str], lang: str) -> dict[str, object]:
    ai_words = AI_WORDS_PL if lang == "pl" else AI_WORDS_EN
    stems = [word[: max(4, math.ceil(len(word) * 0.8))] for word in ai_words]
    count = 0
    found_words: list[str] = []
    for word in words:
        folded = _fold_text(word)
        for stem in stems:
            if len(folded) >= len(stem) and folded.startswith(stem):
                count += 1
                found_words.append(word)
                break
    return {
        "count": count,
        "density": count / len(words) * 100 if words else 0.0,
        "words": found_words,
    }


def _em_dash_count(text: str) -> int:
    return len(re.findall(r"[—–]", text))


def _paragraph_variance(paragraphs: list[str]) -> float:
    if len(paragraphs) < 2:
        return 0.0
    return _stddev(len(_extract_words(paragraph)) for paragraph in paragraphs)


def _sentence_opening_diversity(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 1.0
    first_words = {
        _fold_text(words[0])
        for sentence in sentences
        if (words := sentence.strip().split())
    }
    return len(first_words) / len(sentences)


def _comma_density(text: str, sentences: list[str]) -> float:
    return text.count(",") / len(sentences) if sentences else 0.0


def _punctuation_variety(text: str) -> int:
    return len(set(re.findall(r"""[.,;:!?()""''„"«»\-–—…\[\]{}]""", text)))


def _average_sentence_length(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    return sum(len(_extract_words(sentence)) for sentence in sentences) / len(sentences)


def _count_syllables(word: str) -> int:
    matches = re.findall(r"[aeiouyąęó]", word.casefold())
    return len(matches) if matches else 1


def _readability_variance(paragraphs: list[str]) -> float:
    if len(paragraphs) < 2:
        return 0.0
    scores = []
    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        words = _extract_words(paragraph)
        if not sentences or not words:
            continue
        avg_sentence_len = len(words) / len(sentences)
        avg_syllables = sum(_count_syllables(word) for word in words) / len(words)
        scores.append(206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables)
    return _stddev(scores)


def _consecutive_sentence_diff(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 0.0
    lengths = [len(_extract_words(sentence)) for sentence in sentences]
    diffs = [abs(lengths[index] - lengths[index - 1]) for index in range(1, len(lengths))]
    return sum(diffs) / len(diffs)


def _function_word_entropy(words: list[str], lang: str) -> float:
    function_words = PL_FUNCTION_WORDS if lang == "pl" else EN_FUNCTION_WORDS
    counts: Counter[str] = Counter()
    total = 0
    for word in words:
        folded = _fold_text(word)
        if folded in function_words:
            counts[folded] += 1
            total += 1
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _detect_list_structure(text: str) -> dict[str, int | float]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return {"ratio": 0.0, "dash_lines": 0, "short_lines": 0, "total_lines": len(lines)}
    dash_lines = 0
    short_lines = 0
    for line in lines:
        stripped = line.strip()
        if re.search(r"[—–]", stripped):
            dash_lines += 1
        if 3 < len(stripped) < 60:
            short_lines += 1
    return {
        "ratio": dash_lines / len(lines) if lines else 0.0,
        "dash_lines": dash_lines,
        "short_lines": short_lines,
        "total_lines": len(lines),
    }


def _compute_score(metrics: dict[str, object]) -> dict[str, object]:
    sent_std_score = _sigmoid(float(metrics["sent_len_std_dev"]), 5.0, 0.7)
    para_var_score = (
        _sigmoid(float(metrics["paragraph_variance"]), 8.0, 0.15)
        if float(metrics["paragraph_variance"]) > 0
        else 0.5
    )
    open_div_score = _sigmoid(float(metrics["sentence_opening_diversity"]), 0.45, 6.0)
    consec_diff_score = _sigmoid(float(metrics["consecutive_diff"]), 4.0, 0.4)
    variation_score = (
        sent_std_score * 0.35
        + para_var_score * 0.20
        + open_div_score * 0.25
        + consec_diff_score * 0.20
    )

    ttr_score = _sigmoid(float(metrics["ttr"]), 0.65, 12.0)
    hapax_score = _sigmoid(float(metrics["hapax"]), 0.52, 8.0)
    vocabulary_score = ttr_score * 0.55 + hapax_score * 0.45

    bigram_score = 1.0 - _sigmoid(float(metrics["max_bigram"]), 4.0, 0.8)
    trigram_score = 1.0 - _sigmoid(float(metrics["max_trigram"]), 2.5, 1.2)
    repetition_score = bigram_score * 0.5 + trigram_score * 0.5

    char_entropy_score = _sigmoid(float(metrics["char_entropy"]), 4.0, 2.0)
    bigram_entropy_score = (
        _sigmoid(float(metrics["word_bigram_entropy"]), 6.0, 0.5)
        if float(metrics["word_bigram_entropy"]) > 0
        else 0.5
    )
    entropy_score = char_entropy_score * 0.5 + bigram_entropy_score * 0.5

    phrase_density = int(metrics["ai_phrase_count"]) / max(int(metrics["word_count"]) / 100.0, 1.0)
    phrase_score = 1.0 - _sigmoid(phrase_density, 1.5, 2.0)
    if int(metrics["em_dash_count"]) == 0:
        em_dash_score = 0.5
    else:
        em_dash_density = int(metrics["em_dash_count"]) / max(int(metrics["char_count"]) / 1000.0, 1.0)
        em_dash_score = 1.0 - _sigmoid(em_dash_density, 1.5, 1.5)
    ai_word_score = 1.0 - _sigmoid(float(metrics["ai_word_density"]), 3.0, 0.6)
    signature_score = phrase_score * 0.45 + em_dash_score * 0.20 + ai_word_score * 0.35

    readability_score = (
        _sigmoid(float(metrics["readability_variance"]), 10.0, 0.15)
        if float(metrics["readability_variance"]) > 0
        else 0.5
    )
    punctuation_score = _sigmoid(float(metrics["punctuation_variety"]), 5.0, 0.5)
    comma_score = _sigmoid(float(metrics["comma_density"]), 0.8, 2.0)
    function_entropy_score = (
        _sigmoid(float(metrics["function_word_entropy"]), 2.5, 1.5)
        if float(metrics["function_word_entropy"]) > 0
        else 0.5
    )
    structure_score = (
        readability_score * 0.30
        + punctuation_score * 0.25
        + comma_score * 0.25
        + function_entropy_score * 0.20
    )

    human_score = (
        variation_score * 0.18
        + vocabulary_score * 0.08
        + repetition_score * 0.12
        + entropy_score * 0.12
        + signature_score * 0.35
        + structure_score * 0.15
    )
    ai_probability = round((1.0 - human_score) * 100)

    if phrase_density > 8:
        ai_probability = max(ai_probability, 90)
    elif phrase_density > 5:
        ai_probability = max(ai_probability, 78)
    elif phrase_density > 3:
        ai_probability = max(ai_probability, 60)

    if phrase_density > 2 and float(metrics["ai_word_density"]) > 2:
        ai_probability = min(ai_probability + 12, 99)
    if float(metrics["ai_word_density"]) > 6:
        ai_probability = min(ai_probability + 5, 99)

    dash_density = int(metrics["em_dash_count"]) / max(int(metrics["word_count"]) / 100.0, 1.0)
    if dash_density > 4:
        ai_probability = min(ai_probability + 15, 99)
    elif dash_density > 2:
        ai_probability = min(ai_probability + 10, 99)
    elif int(metrics["em_dash_count"]) > 2:
        ai_probability = min(ai_probability + 5, 99)

    list_structure = metrics["list_structure"]
    assert isinstance(list_structure, dict)
    if int(list_structure["dash_lines"]) >= 3:
        ai_probability = min(ai_probability + 12, 99)
    elif int(list_structure["dash_lines"]) >= 2 and float(list_structure["ratio"]) > 0.3:
        ai_probability = min(ai_probability + 8, 99)

    if int(metrics["ai_phrase_count"]) >= 1 and float(metrics["sent_len_std_dev"]) < 4:
        ai_probability = max(ai_probability, 45)
    if int(metrics["ai_phrase_count"]) >= 1 and int(metrics["em_dash_count"]) >= 1:
        ai_probability = min(ai_probability + 10, 99)
    if float(metrics["ai_word_density"]) > 1 and float(metrics["sent_len_std_dev"]) < 5:
        ai_probability = min(ai_probability + 8, 99)

    if (
        int(metrics["ai_phrase_count"]) == 0
        and float(metrics["ai_word_density"]) < 0.3
        and int(metrics["em_dash_count"]) == 0
        and variation_score > 0.65
    ):
        ai_probability = min(ai_probability, 25)

    ai_probability = int(_clamp(ai_probability, 0, 100))

    return {
        "ai_probability_percent": ai_probability,
        "categories": {
            "variation": {
                "score": round(variation_score * 100),
                "details": {
                    "sent_std": round(float(metrics["sent_len_std_dev"]), 1),
                    "paragraph_variance": round(float(metrics["paragraph_variance"]), 1),
                    "opening_diversity": round(float(metrics["sentence_opening_diversity"]) * 100),
                    "consecutive_diff": round(float(metrics["consecutive_diff"]), 1),
                },
            },
            "vocabulary": {
                "score": round(vocabulary_score * 100),
                "details": {
                    "ttr": round(float(metrics["ttr"]) * 100),
                    "hapax": round(float(metrics["hapax"]) * 100),
                },
            },
            "repetition": {
                "score": round(repetition_score * 100),
                "details": {
                    "max_bigram": metrics["max_bigram"],
                    "max_trigram": metrics["max_trigram"],
                },
            },
            "entropy": {
                "score": round(entropy_score * 100),
                "details": {
                    "char_entropy": round(float(metrics["char_entropy"]), 2),
                    "word_bigram_entropy": round(float(metrics["word_bigram_entropy"]), 2),
                },
            },
            "signatures": {
                "score": round(signature_score * 100),
                "details": {
                    "phrases": metrics["ai_phrase_count"],
                    "dash_count": metrics["em_dash_count"],
                    "ai_word_density": round(float(metrics["ai_word_density"]), 1),
                },
            },
            "structure": {
                "score": round(structure_score * 100),
                "details": {
                    "readability_variance": round(float(metrics["readability_variance"]), 1),
                    "punctuation_variety": metrics["punctuation_variety"],
                    "comma_density": round(float(metrics["comma_density"]), 1),
                },
            },
        },
    }


def _site_metrics(categories: dict[str, object]) -> dict[str, dict[str, object]]:
    variation = categories["variation"]["details"]
    vocabulary = categories["vocabulary"]["details"]
    entropy = categories["entropy"]["details"]
    repetition = categories["repetition"]["details"]
    signatures = categories["signatures"]["details"]
    structure = categories["structure"]["details"]

    definitions = (
        (
            "variation",
            "Zmienność tekstu",
            f"Odch. std. zdań: {variation['sent_std']:.1f} | "
            f"Zróżnicowanie początków: {variation['opening_diversity']}%",
        ),
        (
            "vocabulary",
            "Słownictwo",
            f"TTR: {vocabulary['ttr']}% | Hapax legomena: {vocabulary['hapax']}%",
        ),
        (
            "entropy",
            "Entropia",
            f"Znakowa: {entropy['char_entropy']:.2f} bit | "
            f"Bigramowa: {entropy['word_bigram_entropy']:.2f} bit",
        ),
        (
            "repetition",
            "Powtarzalność",
            f"Maks. bigram: {repetition['max_bigram']}x | "
            f"Maks. trigram: {repetition['max_trigram']}x",
        ),
        (
            "signatures",
            "Sygnatury AI",
            f"Frazy AI: {signatures['phrases']} | Em dash: {signatures['dash_count']} | "
            f"Słowa AI: {signatures['ai_word_density']:.1f}%",
        ),
        (
            "structure",
            "Struktura",
            f"Czytelność (wariancja): {structure['readability_variance']:.1f} | "
            f"Interpunkcja: {structure['punctuation_variety']} typów",
        ),
    )

    return {
        key: {
            "label": label,
            "ai_probability_percent": 100 - int(categories[key]["score"]),
            "human_score_percent": int(categories[key]["score"]),
            "detail": detail,
            "details": categories[key]["details"],
        }
        for key, label, detail in definitions
    }


def analyze_text(text: str) -> dict[str, object]:
    lang = detect_language(text)
    sentences = _split_sentences(text)
    words = _extract_words(text)
    paragraphs = _split_paragraphs(text)

    if len(words) < 10:
        raise RuntimeError("Enter at least 10 words to run heuristic analysis.")
    if not sentences:
        raise RuntimeError("Input text must contain at least one sentence.")

    sentence_variation = _sentence_length_variation(sentences)
    ai_phrases = _detect_ai_phrases(text, lang)
    ai_words = _detect_ai_words(words, lang)

    metrics: dict[str, object] = {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
        "char_entropy": _char_entropy(text),
        "word_bigram_entropy": _word_bigram_entropy(words),
        "sent_len_std_dev": sentence_variation["std_dev"],
        "sent_len_mean": sentence_variation["mean"],
        "word_len_std_dev": _stddev(len(word) for word in words),
        "ttr": _type_token_ratio(words),
        "hapax": _hapax_ratio(words),
        "max_bigram": _max_ngram_frequency(words, 2),
        "max_trigram": _max_ngram_frequency(words, 3),
        "ai_phrase_count": len(ai_phrases),
        "ai_word_density": ai_words["density"],
        "em_dash_count": _em_dash_count(text),
        "paragraph_variance": _paragraph_variance(paragraphs),
        "sentence_opening_diversity": _sentence_opening_diversity(sentences),
        "comma_density": _comma_density(text, sentences),
        "punctuation_variety": _punctuation_variety(text),
        "avg_sentence_length": _average_sentence_length(sentences),
        "readability_variance": _readability_variance(paragraphs),
        "consecutive_diff": _consecutive_sentence_diff(sentences),
        "function_word_entropy": _function_word_entropy(words, lang),
        "list_structure": _detect_list_structure(text),
    }

    score = _compute_score(metrics)
    ai_probability = score["ai_probability_percent"] / 100.0
    human_probability = 1.0 - ai_probability
    return {
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "ai_probability_percent": score["ai_probability_percent"],
        "human_probability_percent": round(human_probability * 100),
        "language": lang,
        "categories": score["categories"],
        "site_metrics": _site_metrics(score["categories"]),
        "signals": {
            "ai_phrases": ai_phrases,
            "ai_words": ai_words,
            "em_dash_count": metrics["em_dash_count"],
        },
        "metrics": metrics,
    }


def build_payload(text: str, *, threshold: float, rich: bool = False) -> dict[str, object]:
    result = analyze_text(text)
    ai_probability = float(result["ai_probability"])
    human_probability = 1.0 - ai_probability
    label = "ai" if ai_probability >= threshold else "human"
    expert_payload = {
        "ai_score": ai_probability,
        "human_score": human_probability,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "chunks": 1,
        "loaded": True,
        "language": result["language"],
        "ai_probability_percent": result["ai_probability_percent"],
        "categories": result["categories"],
        "site_metrics": result["site_metrics"],
        "signals": result["signals"],
        "metrics": result["metrics"],
    }
    if rich:
        try:
            from personal_style_pl.features.rich_metrics import rich_metrics_for_text
            from personal_style_pl.ai_markers import ai_marker_report
            expert_payload["rich_metrics"] = rich_metrics_for_text(text)
            expert_payload["ai_leaning"] = ai_marker_report(text)
        except Exception as exc:  # personal_style_pl deps not installed
            expert_payload["rich_metrics"] = {"error": str(exc)}
            expert_payload["ai_leaning"] = {"error": str(exc)}
    return {
        "text_preview": text[:250],
        "weights": {"heuristic": 1.0},
        "experts": {"heuristic": expert_payload},
        "ensemble": {
            "ai_score": ai_probability,
            "human_score": human_probability,
            "ai_probability": ai_probability,
            "human_probability": human_probability,
            "threshold": threshold,
            "label": label,
        },
        "calibration": {
            "status": "uncalibrated_heuristic",
            "calibrated": False,
            "message": "Fast heuristic scores are raw rule-based signals, not calibrated probabilities.",
        },
        "device": "cpu",
        "mode": "heuristic",
    }


def _parse_probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1]")
    return parsed


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text.strip()
    if args.text_file is not None:
        text_path = Path(args.text_file)
        try:
            return text_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Cannot read --text-file '{text_path}': {exc.strerror or exc}") from exc
    return sys.stdin.read().strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast local heuristic AI-text scoring.")
    parser.add_argument("--text", help="Input text as CLI argument.")
    parser.add_argument("--text-file", dest="text_file", help="Read input text from a file.")
    parser.add_argument(
        "--threshold",
        type=_parse_probability,
        default=0.5,
        help="Decision threshold on heuristic AI probability.",
    )
    parser.add_argument("--json", action="store_true", help="Print compact JSON output.")
    parser.add_argument("--rich", action="store_true",
                        help="Attach rich metrics + AI-leaning overlay (lazy-imports personal_style_pl).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        text = _read_text(args)
        if not text:
            raise RuntimeError("No input text provided.")
        payload = build_payload(text, threshold=args.threshold, rich=args.rich)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("Heuristic AI probability:", f"{payload['ensemble']['ai_probability']:.6f}")
    print("Decision:", payload["ensemble"]["label"])
    print("Language:", payload["experts"]["heuristic"]["language"])
    print(
        "Counts:",
        f"words={payload['experts']['heuristic']['metrics']['word_count']}",
        f"sentences={payload['experts']['heuristic']['metrics']['sentence_count']}",
    )
    print("Site metrics:")
    for metric in payload["experts"]["heuristic"]["site_metrics"].values():
        print(f"- {metric['label']}: {metric['ai_probability_percent']}% AI ({metric['detail']})")


if __name__ == "__main__":
    main()
