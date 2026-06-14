"""argparse CLI for personal_style_pl. Entry: `python -m personal_style_pl.cli`."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib

from .io import load_samples_from_dir, load_samples_from_csv, ensure_parent
from .profile.build_profile import build_profile
from .profile.similarity import score_text
from .profile.report import score_result_to_dict
from .utils.json import dumps_json
from heuristic_detector import _extract_words


def _cmd_build_profile(args) -> int:
    if args.csv:
        docs = load_samples_from_csv(args.csv, text_col=args.text_col)
    else:
        docs = load_samples_from_dir(args.samples_dir)
    profile = build_profile(
        docs, use_stylometrix=not args.no_stylometrix,
        include_ngrams=args.include_ngrams,
        include_rich=not args.no_rich,
        include_perplexity=args.with_perplexity,
        chunk_sentences=args.chunk_sentences, min_chunk_tokens=args.min_chunk_tokens)
    joblib.dump(profile, ensure_parent(args.output))
    if args.report:
        from .utils.json import dump_json
        dump_json({
            "profile_id": profile.profile_id, "created_at": profile.created_at,
            "training_sample_count": profile.training_sample_count,
            "training_chunk_count": profile.training_chunk_count,
            "total_tokens": profile.total_tokens, "genres": profile.genres,
            "config": profile.config, "warnings": profile.warnings,
        }, args.report)
    print(f"Profile written to {args.output} "
          f"({profile.training_chunk_count} chunks, {profile.total_tokens} tokens)",
          file=sys.stderr)
    for w in profile.warnings:
        print(f"  warning: {w}", file=sys.stderr)
    return 0


def _cmd_score(args) -> int:
    profile = joblib.load(args.profile)
    text = Path(args.text_file).read_text(encoding="utf-8")
    result = score_text(profile, text)
    payload = score_result_to_dict(result)
    if args.with_heuristics:
        from .bridge import attach_heuristics
        payload = attach_heuristics(payload, text)
    if args.json:
        print(dumps_json(payload, indent=2))
    else:
        print(f"Style match: {result.style_match_score}/100 ({result.label}), "
              f"confidence {result.confidence}")
        print(result.summary)
    return 0


def _cmd_rank(args) -> int:
    profile = joblib.load(args.profile)
    rows = []
    for path in sorted(Path(args.candidates_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        result = score_text(profile, text)
        rows.append({
            "filename": path.name, "style_match_score": result.style_match_score,
            "label": result.label, "confidence": result.confidence,
            "word_count": len(_extract_words(text)),
            "warnings": "; ".join(result.warnings),
        })
    rows.sort(key=lambda r: r["style_match_score"], reverse=True)
    with ensure_parent(args.output).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "style_match_score", "label", "confidence",
            "word_count", "warnings"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ranking written to {args.output} ({len(rows)} candidates)")
    return 0


def _cmd_describe_profile(args) -> int:
    from .profile.report import profile_to_markdown
    profile = joblib.load(args.profile)
    ensure_parent(args.output).write_text(profile_to_markdown(profile), encoding="utf-8")
    print(f"Profile summary written to {args.output}", file=sys.stderr)
    return 0


def _cmd_suggest_edits(args) -> int:
    from .edit.style_editor import suggestions_to_markdown
    profile = joblib.load(args.profile)
    text = Path(args.text_file).read_text(encoding="utf-8")
    ensure_parent(args.output).write_text(
        suggestions_to_markdown(profile, text, args.mode), encoding="utf-8")
    print(f"Suggestions written to {args.output}", file=sys.stderr)
    return 0


def _cmd_edit(args) -> int:
    from .edit.style_editor import conservative_edit
    profile = joblib.load(args.profile)
    text = Path(args.text_file).read_text(encoding="utf-8")
    ensure_parent(args.output).write_text(
        conservative_edit(profile, text, args.mode), encoding="utf-8")
    print(f"Edited text written to {args.output}", file=sys.stderr)
    return 0


def _cmd_train_supervised(args) -> int:
    from .models.supervised import train_supervised, save_model
    model = train_supervised(args.csv, text_col=args.text_col, label_col=args.label_col)
    save_model(model, args.output)
    print(f"Supervised model written to {args.output} "
          f"(cv_accuracy={model.cv_accuracy})", file=sys.stderr)
    return 0


def _cmd_ai_markers(args) -> int:
    import joblib
    from .ai_markers import ai_marker_report
    from .utils.json import dumps_json
    text = Path(args.text_file).read_text(encoding="utf-8")
    profile = joblib.load(args.profile) if args.profile else None
    report = ai_marker_report(text, profile=profile, with_perplexity=args.with_perplexity)
    if args.json:
        print(dumps_json(report, indent=2))
    else:
        print(f"language={report['language']} ai_leaning_score={report['ai_leaning_score']} "
              f"confidence={report['confidence']}")
        for r in report["markers"]:
            print(f"- {r['feature']}={r['value']} ({r.get('leaning','-')})")
        for w in report["warnings"]:
            print(f"  ! {w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="personal-style-pl",
                                     description="Polish personal writing-style similarity.")
    sub = parser.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build-profile", help="Build a style profile from samples.")
    bp.add_argument("--samples-dir")
    bp.add_argument("--csv")
    bp.add_argument("--text-col", default="text")
    bp.add_argument("--output", required=True)
    bp.add_argument("--report")
    bp.add_argument("--min-chunk-tokens", type=int, default=120)
    bp.add_argument("--chunk-sentences", type=int, default=8)
    bp.add_argument("--include-ngrams", action="store_true")
    bp.add_argument("--no-stylometrix", action="store_true")
    bp.add_argument("--no-rich", action="store_true",
                    help="Disable the always-on rich-metrics baseline block.")
    bp.add_argument("--with-perplexity", action="store_true",
                    help="Add papuGaPT2 perplexity features to the baseline (heavy; gated).")
    bp.set_defaults(func=_cmd_build_profile)

    sc = sub.add_parser("score", help="Score one text against a profile.")
    sc.add_argument("--profile", required=True)
    sc.add_argument("--text-file", required=True)
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--with-heuristics", action="store_true")
    sc.set_defaults(func=_cmd_score)

    rk = sub.add_parser("rank", help="Rank candidate drafts.")
    rk.add_argument("--profile", required=True)
    rk.add_argument("--candidates-dir", required=True)
    rk.add_argument("--output", required=True)
    rk.set_defaults(func=_cmd_rank)

    dp = sub.add_parser("describe-profile", help="Summarize a profile as Markdown.")
    dp.add_argument("--profile", required=True)
    dp.add_argument("--output", required=True)
    dp.set_defaults(func=_cmd_describe_profile)

    se = sub.add_parser("suggest-edits", help="Suggest conservative edits (Markdown).")
    se.add_argument("--profile", required=True)
    se.add_argument("--text-file", required=True)
    se.add_argument("--output", required=True)
    se.add_argument("--mode", choices=["light", "medium", "strong"], default="light")
    se.set_defaults(func=_cmd_suggest_edits)

    ed = sub.add_parser("edit", help="Apply conservative deterministic edits.")
    ed.add_argument("--profile", required=True)
    ed.add_argument("--text-file", required=True)
    ed.add_argument("--output", required=True)
    ed.add_argument("--mode", choices=["light", "medium", "strong"], default="light")
    ed.set_defaults(func=_cmd_edit)

    tsup = sub.add_parser("train-supervised", help="Train a mine-vs-other classifier.")
    tsup.add_argument("--csv", required=True)
    tsup.add_argument("--text-col", default="text")
    tsup.add_argument("--label-col", default="label")
    tsup.add_argument("--output", required=True)
    tsup.set_defaults(func=_cmd_train_supervised)

    am = sub.add_parser("ai-markers", help="Interpretable AI-leaning marker report.")
    am.add_argument("--text-file", required=True)
    am.add_argument("--profile")
    am.add_argument("--with-perplexity", action="store_true")
    am.add_argument("--json", action="store_true")
    am.set_defaults(func=_cmd_ai_markers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-profile" and not (args.samples_dir or args.csv):
            parser.error("build-profile requires --samples-dir or --csv")
        return args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
