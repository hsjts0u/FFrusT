#!/bin/bash

set -x

RUSTFLAGS="-C target-feature=+avx2 -C opt-level=3" cargo run --release -- v1.csv
RUSTFLAGS="-C no-vectorize-loops -C no-vectorize-slp -C opt-level=3" cargo run --release -- v2.csv
