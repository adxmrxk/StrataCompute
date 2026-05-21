#!/usr/bin/env bash
# =============================================================================
#  StrataCompute container entrypoint.
#
#    test        ctest offline suite (P1/2/4) + ONNX baseline test (P3)
#    phase5      run the C++ pipeline benchmark + PyTorch driver -> docs
#    cachegrind  REAL L1/LL cache-miss counts via Valgrind Cachegrind
#                (the metric the spec wanted that Windows could not provide)
#    all         test -> phase5 -> cachegrind   (default)
#
#  Mount a host dir at /app/out to collect generated reports.
# =============================================================================
set -euo pipefail
cd /app

ORT_LIB="$(dirname "$(find build-onnx/_ort -name 'libonnxruntime.so' | head -n1)")"
export LD_LIBRARY_PATH="${ORT_LIB}:${LD_LIBRARY_PATH:-}"

collect() {
    if [ -d /app/out ]; then
        mkdir -p /app/out
        cp -f results/*.csv /app/out/ 2>/dev/null || true
        cp -f results/cachegrind.txt /app/out/ 2>/dev/null || true
        cp -f docs/benchmarks.md /app/out/ 2>/dev/null || true
        echo "  (reports copied to mounted /app/out)"
    fi
}

run_test() {
    echo "=== offline suite (Phases 1, 2, 4) — GCC -Werror build ==="
    ctest --test-dir build --output-on-failure
    echo "=== ONNX Runtime baseline (Phase 3) ==="
    ./build-onnx/strata_onnx_test /app/models
}

run_phase5() {
    echo "=== Phase 5: C++ pipeline benchmark (ORT vs StrataCompute) ==="
    mkdir -p results
    ./build-onnx/bench_pipeline /app/models results/latency_cpp.csv
    echo "=== Phase 5: PyTorch baseline + report ==="
    python scripts/run_phase5.py
    collect
}

run_cachegrind() {
    echo "=== Cachegrind: real L1/LL data-cache misses ==="
    echo "(whole-process, low iter count; per-inference ~ totals / iters)"
    mkdir -p results
    valgrind --tool=cachegrind --cachegrind-out-file=/tmp/cg.out \
        ./build-onnx/bench_pipeline /app/models results/latency_cpp.csv 50 500 \
        2> results/cachegrind.txt || true
    grep -E 'D1 |LLd |D   refs|I   refs' results/cachegrind.txt \
        | sed 's/^==[0-9]*== /  /' | tee -a /dev/stdout
    echo "  full log: results/cachegrind.txt"
    collect
}

case "${1:-all}" in
    test)       run_test ;;
    phase5)     run_phase5 ;;
    cachegrind) run_cachegrind ;;
    all)        run_test; run_phase5; run_cachegrind ;;
    *) echo "usage: {test|phase5|cachegrind|all}" >&2; exit 2 ;;
esac

echo
echo "DONE: ${1:-all}"
