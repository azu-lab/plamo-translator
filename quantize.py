"""PLaMoモデルを量子化して保存する（初回だけ実行）

使い方:
    python quantize.py            # 4bit (NF4) -> ./model-4bit
    python quantize.py --bits 8   # 8bit       -> ./model-8bit
    python quantize.py --src ./model-0.8b --dst ./model-0.8b-4bit
"""

import argparse
import gc
import os
import shutil

# bitsandbytes の import でコンパイル用ワーカーが大量に起動するのを防ぐ（app.py 参照）
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 量子化モデルを保存した後も、推論に必要なファイル（カスタムコード・テンプレート）をコピーする
EXTRA_FILES = [
    "modeling_plamo.py",
    "tokenization_plamo.py",
    "chat_template.jinja",
    "tokenizer.jsonl",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "LICENSE",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--src", default="./model", help="元のモデル（BF16）のディレクトリ")
    parser.add_argument("--dst", default=None, help="保存先（省略時は ./model-<bits>bit）")
    args = parser.parse_args()
    SRC = args.src

    if not torch.cuda.is_available():
        raise SystemExit("bitsandbytes量子化にはCUDA GPUが必要です。")

    dst = args.dst or f"./model-{args.bits}bit"

    if args.bits == 4:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # 出力層は量子化すると訳質が落ちやすいので元の精度のまま
            llm_int8_skip_modules=["lm_head"],
        )
    else:
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"],
        )

    print(f"loading {SRC} ...")
    model = AutoModelForCausalLM.from_pretrained(
        SRC,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization_config=qconfig,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

    print(f"saving to {dst} ...")
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)

    for name in EXTRA_FILES:
        src_path = os.path.join(SRC, name)
        if not os.path.exists(src_path):
            continue
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(dst, name), dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst)

    print(f"VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
