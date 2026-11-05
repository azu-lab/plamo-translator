"""Unlimited-OCR で PDF のページを読み取る（.venv-ocr で動かす）

アプリ（.venv）とは transformers の版が合わないため、別の仮想環境の別プロセスとして起動する。

    .venv-ocr/bin/python ocr_worker.py MODEL_DIR PDF_PATH PAGES_JSON [--dpi 200]

PAGES_JSON は 0 始まりのページ番号のリスト（例: "[0, 3, 4]"）。
標準出力には 1 行 1 件の JSON だけを書く（モデルが出すログは標準エラーへ回す）。

    {"event": "loaded", "sec": 3.1}
    {"event": "page", "page": 0, "text": "<|det|>title [..]<|/det|>...", "sec": 37.2}
    {"event": "done"}
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("pdf_path")
    parser.add_argument("pages")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out = sys.stdout
    sys.stdout = sys.stderr      # モデルの print が JSON に混ざらないようにする

    def emit(**kwargs):
        out.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
        out.flush()

    import pymupdf
    import torch
    from transformers import AutoModel, AutoTokenizer

    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    emit(event="loaded", sec=round(time.time() - t, 1))

    doc = pymupdf.open(args.pdf_path)
    with tempfile.TemporaryDirectory(prefix="plamo_ocr_") as tmp:
        for page_no in json.loads(args.pages):
            t = time.time()
            image_path = os.path.join(tmp, f"page_{page_no}.png")
            doc[page_no].get_pixmap(dpi=args.dpi).save(image_path)
            with contextlib.redirect_stdout(sys.stderr):
                # 8GB の GPU に収まる base 設定（ページを分割せず 1024px で読む）。
                # 長いページで同じ内容を繰り返さないよう、モデル README の値で n-gram の繰り返しを禁止する
                text = model.infer(
                    tokenizer, prompt="<image>document parsing.", image_file=image_path,
                    output_path=tmp, base_size=1024, image_size=1024, crop_mode=False,
                    eval_mode=True, max_length=8192,
                    no_repeat_ngram_size=35, ngram_window=128,
                )
            emit(event="page", page=page_no, text=text, sec=round(time.time() - t, 1))
            torch.cuda.empty_cache()
    emit(event="done")


if __name__ == "__main__":
    main()
