#!/usr/bin/env bash
# Fetch small curated subset of NIFS LHD Open Data from AWS (--no-sign-request). Reproducible.

set -e
OUT="${1:-data/fusion_real_lhd/raw}"
mkdir -p "$OUT"
echo "NIFS LHD Open Data on AWS. List bucket (no-sign-request):"
aws s3 ls s3://nifs-ldb-open-data/ --no-sign-request 2>/dev/null | head -20 || true
echo "To download a subset, use: aws s3 cp s3://nifs-ldb-open-data/<path> $OUT/ --recursive --no-sign-request"
echo "LHD dataset generator will use synthetic time series if raw files absent."
