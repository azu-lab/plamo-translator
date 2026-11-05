import argparse
import atexit
import os
import re
import signal
import sys
import threading
import zipfile
import tempfile

import gradio as gr

import paper
from backends import TransformersBackend, VllmBackend, VllmServer, find_vllm_models

# ------------------------------------------------------------
# 翻訳エンジンの選択（--backend auto のとき、上から順に試す）
#   1. vLLM サーバーがすでに起動している（./serve.sh を手動で起動した） -> それを使う
#   2. .venv-vllm と vLLM 用のモデル（./model-fp8 など）がある
#        -> アプリが vLLM サーバーを起動する。GUI でモデルを切り替えられる
#   3. それ以外 -> Transformers（model-4bit があればそれ、無ければ model）
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
                    help="vLLM サーバーの URL")
parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL"),
                    help="アプリが vLLM サーバーを起動するときに最初に使うモデル（既定: ./model-fp8）")
parser.add_argument("--model", default=os.environ.get("PLAMO_MODEL"),
                    help="Transformers で使うモデルのディレクトリ")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--batch-size", type=int, default=8,
                    help="Transformers で一度に訳す段落数（VRAMが足りなければ下げる）")
args = parser.parse_args()

# アプリが起動・管理する vLLM サーバー（手動起動のサーバーや Transformers のときは None）
vllm_server = None
vllm_models = []          # [(表示名, パス)]
_backend = None


def create_backend():
    global vllm_server, vllm_models

    if args.backend != "transformers":
        if VllmBackend.is_available(args.vllm_url):
            print(f"起動済みの vLLM サーバー（{args.vllm_url}）を使います。モデルの切り替えはできません。")
            return VllmBackend(args.vllm_url)

        vllm_models = find_vllm_models(".")
        if os.access(".venv-vllm/bin/vllm", os.X_OK) and vllm_models:
            paths = [p for _, p in vllm_models]
            initial = args.vllm_model or ("./model-fp8" if "./model-fp8" in paths else paths[0])
            vllm_server = VllmServer(args.vllm_url)
            vllm_server.model_path = initial   # 画面の初期表示用（起動はスレッドで行う）
            # 終了時（Ctrl+C を含む）に vLLM サーバーも止めて VRAM を解放する
            atexit.register(vllm_server.stop)
            threading.Thread(target=_start_vllm, args=(initial,), daemon=True).start()
            return None   # 起動が終わったら _start_vllm がセットする

        if args.backend == "vllm":
            raise SystemExit(
                "vLLM を使えません。.venv-vllm と ./model-fp8 を用意するか、./serve.sh を起動してください。"
            )

    model_path = args.model or next(
        (p for p in ("./model-4bit", "./model") if os.path.isdir(p)), None
    )
    if model_path is None:
        raise SystemExit(
            "翻訳エンジンが見つかりません。README の手順で vLLM の環境とモデルを用意するか、"
            "「Transformers で動かす」の手順で ./model を用意してください。"
        )
    return TransformersBackend(model_path, batch_size=args.batch_size)


def _start_vllm(model_path):
    global _backend
    print(f"vLLM サーバーを起動中: {model_path}")
    try:
        vllm_server.start(model_path)
    except Exception as e:
        print(f"vLLM サーバーを起動できませんでした: {e}")
        return
    _backend = VllmBackend(args.vllm_url)
    print("backend:", _backend.name)


def get_backend():
    """翻訳エンジンを返す。vLLM サーバーの起動中・切り替え中はエラーにする"""
    if vllm_server is not None and not vllm_server.ready:
        raise gr.Error("翻訳モデルを起動中です。画面上部の表示が「翻訳できます」になってから実行してください。")
    return _backend


def engine_status():
    if vllm_server is not None:
        return vllm_server.status_text()
    return f"✅ `{_backend.name}` で翻訳できます"


def switch_model(model_path):
    """GUI からのモデル切り替え（vLLM サーバーの再起動）"""
    global _backend
    if vllm_server is None:
        raise gr.Error("この起動方法ではモデルを切り替えられません。")
    if model_path == vllm_server.model_path and vllm_server.ready:
        return engine_status(), gr.update(value=model_path)
    _backend = None
    try:
        vllm_server.switch(model_path)
    except Exception as e:
        gr.Warning(vllm_server.message or str(e))
    if vllm_server.ready:
        _backend = VllmBackend(args.vllm_url)
    return engine_status(), gr.update(value=vllm_server.model_path)


def _exit_on_sigterm(signum, frame):
    sys.exit(0)   # atexit を走らせて vLLM サーバーを止める


signal.signal(signal.SIGTERM, _exit_on_sigterm)
# Ctrl+C（SIGINT）で KeyboardInterrupt を起こし、atexit で vLLM サーバーを止める
# （バックグラウンド起動などで SIGINT が無視に設定されている場合も元に戻す）
signal.signal(signal.SIGINT, signal.default_int_handler)

_initial_backend = create_backend()
if _initial_backend is not None:
    _backend = _initial_backend
    print("backend:", _backend.name)


# 言語コードは model/README_ja.md の「対応言語」を参照
LANGUAGES = [
    ("自動判定", "auto"),
    ("英語", "en"),
    ("日本語", "ja"),
    ("やさしい日本語", "ja(easy)"),
    ("中国語（簡体字）", "zh-hans"),
    ("中国語（繁体字）", "zh-hant"),
    ("韓国語", "ko"),
    ("フランス語", "fr"),
    ("ドイツ語", "de"),
    ("スペイン語", "es"),
    ("イタリア語", "it"),
    ("ポルトガル語", "pt"),
    ("ロシア語", "ru"),
    ("ベトナム語", "vi"),
    ("タイ語", "th"),
    ("インドネシア語", "id"),
]
TARGET_LANGUAGES = [lang for lang in LANGUAGES if lang[1] != "auto"]

FORMS = [("常体（だ・である）", "plain"), ("敬体（です・ます）", "polite")]


def split_text(text, max_chars=2200):
    """段落を維持しながら文章を分割する"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        if len(current) + len(paragraph) > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current += "\n\n" + paragraph if current else paragraph

    if current:
        chunks.append(current)

    return chunks


# ------------------------------------------------------------
# テキスト翻訳
# ------------------------------------------------------------
def translate_input_text(text, source_lang, target_lang, form,
                         progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("翻訳する文章を入力してください。")
    if source_lang == target_lang:
        raise gr.Error("翻訳元と翻訳先が同じ言語です。")

    chunks = split_text(text)
    results = get_backend().translate_many(
        chunks, source_lang, target_lang, form,
        progress=lambda done, total: progress(done / total, desc=f"翻訳中 ({done}/{total})"),
    )
    return "\n\n".join(results)


def swap_languages(source_lang, target_lang, text, translated):
    # 自動判定は翻訳先にできないので、そのときは入れ替えない
    if source_lang == "auto":
        return source_lang, target_lang, text, translated
    return target_lang, source_lang, translated, text


# ------------------------------------------------------------
# PDF翻訳（章ごとの全文訳を Markdown / HTML で出力）
# ------------------------------------------------------------
def translate_pdfs(files, source_lang, target_lang, form, formats,
                   bilingual, with_images, translate_refs,
                   progress=gr.Progress()):
    if not files:
        raise gr.Error("PDFを選択してください。")
    if not formats:
        raise gr.Error("出力形式を1つ以上選んでください。")
    backend = get_backend()

    output_dir = tempfile.mkdtemp(prefix="plamo_")
    output_files = []
    preview = ""

    for n, pdf_path in enumerate(files):
        name = os.path.basename(pdf_path)
        stem = os.path.splitext(name)[0]
        prefix = f"[{n + 1}/{len(files)}] {name}"

        progress(0, desc=f"{prefix}: 本文を抽出中")
        md_source = paper.pdf_to_markdown(pdf_path, with_images=with_images)
        blocks = paper.parse_blocks(md_source)

        if not any(b.text for b in blocks if b.kind not in ("table", "image")):
            raise gr.Error(
                f"{name}: テキストを抽出できませんでした（スキャン画像のPDFはOCRが必要です）。"
            )

        def on_progress(done, total):
            progress(done / total, desc=f"{prefix}: 翻訳中 ({done}/{total} 段落)")

        paper.translate_blocks(
            blocks,
            lambda texts, cb: backend.translate_many(
                texts, source_lang, target_lang, form, progress=cb
            ),
            translate_references=translate_refs,
            progress=on_progress,
        )

        md_text = paper.render_markdown(blocks, name, bilingual=bilingual)
        title = next((b.translated or b.text for b in blocks if b.kind == "title"), stem)

        if "Markdown" in formats:
            path = os.path.join(output_dir, f"{stem}_ja.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(md_text)
            output_files.append(path)

        if "HTML" in formats:
            path = os.path.join(output_dir, f"{stem}_ja.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(paper.render_html(md_text, title))
            output_files.append(path)

        if not preview:
            # 画面のプレビューは画像を除いて軽くする
            preview = re.sub(r"!\[[^\]]*\]\(data:[^)]*\)", "*(図)*", md_text)

    if len(files) > 1:
        zip_path = os.path.join(output_dir, "translated_papers.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for path in output_files:
                z.write(path, os.path.basename(path))
        output_files = [zip_path] + output_files

    return output_files, preview


def language_controls(default_source="en", default_target="ja"):
    with gr.Row():
        source = gr.Dropdown(LANGUAGES, value=default_source, label="翻訳元")
        target = gr.Dropdown(TARGET_LANGUAGES, value=default_target, label="翻訳先")
        form = gr.Radio(FORMS, value="plain", label="文体（翻訳先が日本語のとき）")
    return source, target, form


with gr.Blocks(title="PLaMo Translator") as demo:

    gr.Markdown(
        """
        # PLaMo 3 Translate
        ### Local Translator

        ローカル環境で文章・PDF論文を翻訳します。
        """
    )

    with gr.Row(equal_height=True):
        can_switch = vllm_server is not None and len(vllm_models) > 1
        model_select = gr.Dropdown(
            vllm_models,
            value=vllm_server.model_path if vllm_server else None,
            label="翻訳モデル（切り替えには 1〜2 分かかります）",
            interactive=can_switch,
            visible=vllm_server is not None,
            scale=3,
        )
        switch_button = gr.Button(
            "モデルを切り替え", interactive=can_switch,
            visible=vllm_server is not None, scale=1,
        )
    engine_md = gr.Markdown(engine_status())
    # 起動・切り替えの進み具合を表示する（翻訳の処理を待たずに更新できるよう同時実行の制限を外す）
    gr.Timer(2).tick(engine_status, outputs=engine_md,
                     concurrency_limit=None, show_progress="hidden")

    with gr.Tab("テキスト翻訳"):
        t_source, t_target, t_form = language_controls("auto", "ja")

        with gr.Row():
            text_input = gr.Textbox(
                label="原文",
                lines=14,
                placeholder="翻訳したい文章を入力してください"
            )
            text_output = gr.Textbox(
                label="訳文",
                lines=14,
                buttons=["copy"]
            )

        with gr.Row():
            swap_button = gr.Button("⇄ 言語を入れ替え")
            text_button = gr.Button("翻訳", variant="primary")

        text_button.click(
            fn=translate_input_text,
            inputs=[text_input, t_source, t_target, t_form],
            outputs=[text_output]
        )
        swap_button.click(
            fn=swap_languages,
            inputs=[t_source, t_target, text_input, text_output],
            outputs=[t_source, t_target, text_input, text_output]
        )

    with gr.Tab("PDF翻訳"):
        gr.Markdown(
            "論文の章立て（1 / 1.1 / I. / A. など）を保ったまま全文訳を出力します。"
            "著者・所属と参考文献は原文のまま残します。"
        )
        p_source, p_target, p_form = language_controls("en", "ja")

        with gr.Row():
            p_formats = gr.CheckboxGroup(
                ["HTML", "Markdown"], value=["HTML", "Markdown"], label="出力形式"
            )
            p_bilingual = gr.Checkbox(False, label="段落ごとに原文を併記（折りたたみ）")
            p_images = gr.Checkbox(True, label="図を埋め込む")
            p_refs = gr.Checkbox(False, label="参考文献も翻訳する")

        pdf_input = gr.File(
            label="翻訳するPDFを選択",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath"
        )

        translate_button = gr.Button(
            "翻訳開始",
            variant="primary"
        )

        output_file = gr.File(
            label="翻訳結果",
            file_count="multiple"
        )

        with gr.Accordion("プレビュー（1本目）", open=False):
            preview_md = gr.Markdown()

        translate_button.click(
            fn=translate_pdfs,
            inputs=[pdf_input, p_source, p_target, p_form, p_formats,
                    p_bilingual, p_images, p_refs],
            outputs=[output_file, preview_md]
        )


    # モデルの切り替え: 実行中は翻訳ボタンを押せないようにして、終わったら戻す
    _locked = [model_select, switch_button, text_button, translate_button]

    def _set_interactive(flag):
        return [gr.update(interactive=flag and can_switch)] * 2 + [gr.update(interactive=flag)] * 2

    switch_button.click(
        lambda: _set_interactive(False), outputs=_locked, queue=False,
    ).then(
        switch_model, inputs=model_select, outputs=[engine_md, model_select],
    ).then(
        lambda: _set_interactive(True), outputs=_locked, queue=False,
    )


# Transformers はGPUを同時に使うと VRAM が足りなくなるので、リクエストは1件ずつ処理する
# （vLLM の場合も、1件の中で段落を並列に投げるのでこれで十分）
demo.queue(default_concurrency_limit=1)

demo.launch(
    server_name="0.0.0.0",
    server_port=args.port
)
