.PHONY: error-atlas error-tsne env

PY?=.venv/bin/python

env:
	@test -x .venv/bin/python || python3 -m venv .venv
	@.venv/bin/pip install --quiet -U pip
	@.venv/bin/pip install --quiet -r requirements.txt

# Rebuild unified errors parquet/csv and generate t-SNE map (incorrect only).
error-atlas:
	$(PY) scripts/build_error_frame.py
	$(PY) scripts/run_tsne.py --only-incorrect

# Just recompute t-SNE (uses existing parquet).
error-tsne:
	$(PY) scripts/run_tsne.py --only-incorrect
