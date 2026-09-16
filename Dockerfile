# =============================================================================
#  StrataCompute — Linux container
# -----------------------------------------------------------------------------
#  Dockerizing is more than packaging here: a Linux image lets the project
#  build with the spec's *original* intended toolchain — GCC with
#  -O3 -march=native -flto -Wall -Wextra -Wpedantic -Werror, C++20 — via the
#  non-MSVC branch already in CMakeLists.txt (the Windows host could only use
#  the MSVC substitution). It also bundles Valgrind, so the **real** L1/LL
#  cache-miss counts the spec asked for — impossible to capture on Windows —
#  can finally be measured (Cachegrind).
#
#  The offline test suite (Phases 1, 2, 4) is built AND run during
#  `docker build`, so the image fails to build if anything regresses under
#  GCC -Werror — the build itself is the verification gate.
#
#  Build:  docker build -t stratacompute .
#  Run:    docker run --rm stratacompute            # = test + phase5 + cachegrind
#          docker run --rm stratacompute test
#          docker run --rm -v "$PWD/out:/app/out" stratacompute phase5
# =============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        curl \
        valgrind \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the Phase 5 PyTorch baseline (Ubuntu 24.04 is PEP 668, so a
# venv). torch comes from the CPU wheel index; numpy/onnx from PyPI.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir "numpy<2" onnx \
 && pip install --no-cache-dir --extra-index-url \
        https://download.pytorch.org/whl/cpu torch

WORKDIR /app
COPY . .

# ---- Phase 1/2/4: build with the spec's GCC toolchain AND run the suite.
#      `ctest` here is the hard gate — image build fails on any regression.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" \
 && ctest --test-dir build --output-on-failure

# ---- Phase 3/5: ONNX Runtime baseline + pipeline benchmark (fetches the
#      prebuilt linux-x64 ORT; compiling bench_pipeline gates GCC issues too).
RUN cmake -S . -B build-onnx -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_BUILD_ONNX=ON -DSTRATA_BUILD_TESTS=ON \
 && cmake --build build-onnx -j"$(nproc)" --target strata_onnx_test bench_pipeline

RUN chmod +x scripts/docker-entrypoint.sh
ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["all"]
