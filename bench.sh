#!/bin/bash

set -x

RUSTFLAGS="-C opt-level=3 -C target-feature=+avx2" cargo run --release -- v1.csv
RUSTFLAGS="-C opt-level=3 -C no-vectorize-loops -C no-vectorize-slp" cargo run --release -- v2.csv
