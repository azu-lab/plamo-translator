#!/usr/bin/env bash
# vLLM 翻訳サーバーを起動する（README「セットアップ」参照）
#
#   ./serve.sh                       # ./model-fp8 を起動
#   MODEL=./model-0.8b-fp8 ./serve.sh
#   GPU_UTIL=0.6 ./serve.sh          # VRAM の確保量を減らす
#
# 起動後、別のターミナルで python app.py を実行する（自動でこのサーバーを使う）。
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-./model-fp8}"
PORT="${PORT:-8000}"
# vLLM は起動時に「GPU メモリ全体 × GPU_UTIL」を先に確保する。
# 8GB の GPU で 2B FP8 なら 0.7（約 5.6GB）で、KV キャッシュに約 1.2GB 残る。
GPU_UTIL="${GPU_UTIL:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

if [ ! -d "$MODEL" ]; then
  echo "モデルが見つかりません: $MODEL（README の手順でダウンロードしてください）" >&2
  exit 1
fi
if [ ! -x .venv-vllm/bin/vllm ]; then
  echo ".venv-vllm がありません（README の手順で作成してください）" >&2
  exit 1
fi

# FlashInfer のサンプラーは初回に CUDA コンパイラ (nvcc) で JIT ビルドしようとする。
# 翻訳は温度 0（貪欲法）で使わないので無効にして、nvcc 無しでも起動できるようにする。
export VLLM_USE_FLASHINFER_SAMPLER=0

exec .venv-vllm/bin/vllm serve "$MODEL" \
  --trust-remote-code \
  --served-model-name plamo-3-translate \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS"
