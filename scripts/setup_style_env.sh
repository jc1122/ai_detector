#!/usr/bin/env bash
# Reproducible single-env setup for ai_detector + personal_style_pl + StyloMetrix.
# Requires `uv` (https://astral.sh/uv). Builds .venv on Python 3.12 (StyloMetrix ceiling:
# stylo_metrix pins spacy==3.7.2, whose thinc stack has no wheels above cp312).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv python install 3.12
rm -rf .venv
uv venv --python 3.12 .venv

# Editable install of the detector + style + stylometrix + test extras.
# numpy<2 is mandatory: thinc 8.2.x was built against numpy 1.x (numpy 2 -> ABI crash).
# pandas<2.2 keeps a clean numpy 1.26 pairing.
uv pip install --python .venv/bin/python -e ".[test,style,style-stylometrix]" "numpy<2" "pandas<2.2"

# pl_nask model: install --no-deps from the TLS-valid HF mirror. The model declares
# spacy<3.6 which would otherwise drag an unbuildable thinc 8.1.x; the official IPI PAN
# host (mozart.ipipan.waw.pl) has an EXPIRED TLS cert, so we use the HF mirror.
uv pip install --python .venv/bin/python --no-deps \
  "https://huggingface.co/ipipan/pl_nask/resolve/main/pl_nask-0.0.7.tar.gz"

echo "Done. Verify:"
echo "  .venv/bin/python -c \"import stylo_metrix as sm; print(sm.StyloMetrix('pl').transform(['Ala ma kota.']).shape)\""
echo "  .venv/bin/python -m pytest -q"
