#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bin="${LLAMA_BIN_DIR:?set LLAMA_BIN_DIR to the llama build bin directory}"
model="${MODEL:?set MODEL to a Q4_0 GGUF model}"
guard="${K200_GUARD:-k200-thermal-guard}"
export KUNLUN_LOG_DIR="${KUNLUN_LOG_DIR:-$PWD/local-results}"
mkdir -p "$KUNLUN_LOG_DIR"
run() {
    local name="$1"; shift
    "$guard" --name "$name" --seconds 300 -- "$@" >"$KUNLUN_LOG_DIR/$name.txt" 2>&1 || { tail -40 "$KUNLUN_LOG_DIR/$name.txt"; return 1; }
    tail -8 "$KUNLUN_LOG_DIR/$name.txt"
}
export GGML_KUNLUN_REQUANT_I8=0
run backend-contract "$bin/kunlun-backend-contract"
run adapter-ops "$bin/test-backend-ops" -b KUNLUN0 -o SET_ROWS,RMS_NORM,SILU,ADD,MUL,SCALE
for pd in 0 1; do
    for type in 0 1 2; do run "adapter-exact-$pd-$type" "$bin/kunlun-gemv-bench" 256 64 1 0 "$type" "$pd"; done
done
export GGML_KUNLUN_REQUANT_I8=1 GGML_KUNLUN_TEST_CHUNKS=1
run adapter-fast-tail "$bin/kunlun-gemv-bench" 256 65 1 0 2
for pd in 0 1; do
    for batch in 1 4 8 32; do
        run "adapter-fast-split-$pd-$batch" "$bin/kunlun-gemv-bench" 1024 512 "$batch" 1 2 "$pd"
    done
done
export GGML_KUNLUN_TEST_CLEAR=1
for fraction in 0 0.25 1; do
    export GGML_KUNLUN_TEST_SPLIT_FRACTION="$fraction" GGML_KUNLUN_REQUANT_I8=0
    for type in 0 1 2; do
        run "adapter-clear-$fraction-$type" "$bin/kunlun-gemv-bench" 256 65 4 1 "$type"
    done
    export GGML_KUNLUN_REQUANT_I8=1
    run "adapter-clear-fast-$fraction" "$bin/kunlun-gemv-bench" 256 65 4 1 2
done
unset GGML_KUNLUN_TEST_CHUNKS GGML_KUNLUN_TEST_CLEAR GGML_KUNLUN_TEST_SPLIT_FRACTION GGML_KUNLUN_TEACHER_TOKENS
run model-cpu "$bin/kunlun-llm-smoke" "$model" cpu 8 "$KUNLUN_LOG_DIR/model-cpu.bin"
export GGML_KUNLUN_TEACHER_TOKENS="$(python3 - "$KUNLUN_LOG_DIR/model-cpu.txt" <<'PY'
import re,sys
from pathlib import Path
print(re.search(r'TOKENS mode=cpu((?: \d+)+)',Path(sys.argv[1]).read_text())[1].strip())
PY
)"
for mode in none layer row; do
    run "model-$mode" "$bin/kunlun-llm-smoke" "$model" "$mode" 8 "$KUNLUN_LOG_DIR/model-$mode.bin"
    python3 "$here/compare-logits.py" "$KUNLUN_LOG_DIR/model-cpu" "$KUNLUN_LOG_DIR/model-$mode" --teacher-forced | tee "$KUNLUN_LOG_DIR/compare-model-$mode.txt"
done
