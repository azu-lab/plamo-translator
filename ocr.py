"""スキャンした PDF（文字を持たないページ）を Unlimited-OCR で読み取る

OCR は .venv-ocr の別プロセス（ocr_worker.py）で動かす。ここでは
- OCR が必要なページを見つける（find_scanned_pages）
- ワーカーを起動して結果を受け取る（run_ocr）
- OCR の出力を paper.py が読める Markdown に直す（ocr_to_markdown）
を行う。

OCR の出力は、ブロックごとに種類と位置（0〜999 に正規化した座標）が付いた形:

    <|det|>title [137, 65, 844, 132]<|/det|>Deadlock- and Starvation-free ...
    <|det|>text [75, 244, 492, 336]<|/det|>Abstract—Robot middleware is ...
    <|det|>table [244, 106, 681, 913]<|/det|><table><tr><td>...</td></tr></table>
    <|det|>equation [580, 742, 845, 760]<|/det|>\\[ ... \\]
    <|det|>image [510, 66, 920, 86]<|/det|>
"""

import base64
import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent
OCR_PYTHON = _ROOT / ".venv-ocr" / "bin" / "python"
OCR_MODEL = _ROOT / "model-ocr"
WORKER = _ROOT / "ocr_worker.py"

# OCR に必要な GPU の空きメモリ（base 設定で測ったピーク 6.8 GiB に少し余裕を持たせた値）
REQUIRED_FREE_MIB = 7000


def ocr_available() -> bool:
    return OCR_PYTHON.exists() and (OCR_MODEL / "config.json").is_file()


def _is_scanned(page) -> bool:
    import pymupdf

    if len(page.get_text().strip()) < 30:
        return bool(page.get_images() or page.get_drawings())
    # 出版社のダウンロード透かし（余白の "Downloaded from ... by [IP]" など）だけ文字を持ち、
    # 本文はページ全体を覆う画像、というスキャン PDF がある。大きな画像の上に文字がほとんど
    # 無ければスキャンしたページとみなす
    area = page.rect.width * page.rect.height
    big = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()
           if pymupdf.Rect(info["bbox"]).get_area() >= area * 0.3]
    if not big:
        return False
    on_image = off_image = 0
    for w in page.get_text("words"):
        center = pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        if any(center in r for r in big):
            on_image += len(w[4])
        else:
            off_image += len(w[4])
    # 画像の外に本文がある（大きな図と本文が並んだページ）なら、文字のあるページとして扱う
    return on_image < 30 and off_image < 200


def find_scanned_pages(pdf_path: str, pages: Optional[Sequence[int]] = None) -> List[int]:
    """本文が画像になっているページ（スキャンしたページ）を返す"""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [i for i in (pages if pages is not None else range(doc.page_count))
                if _is_scanned(doc[i])]


def gpu_free_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run_ocr(pdf_path: str, pages: Sequence[int],
            progress: Optional[Callable[[int, int], None]] = None,
            log_path: str = "logs/ocr.log", dpi: int = 200) -> Dict[int, str]:
    """ワーカーを起動し、{ページ番号: OCR の出力} を返す。GPU の空きは呼び出し側で用意する"""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    results: Dict[int, str] = {}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [str(OCR_PYTHON), str(WORKER), str(OCR_MODEL), pdf_path,
             json.dumps(list(pages)), "--dpi", str(dpi)],
            stdout=subprocess.PIPE, stderr=log, text=True,
        )
        try:
            if progress:
                progress(0, len(pages))
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "page":
                    results[event["page"]] = event["text"]
                    if progress:
                        progress(len(results), len(pages))
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if proc.returncode != 0 or len(results) != len(pages):
        tail = Path(log_path).read_text(errors="replace").splitlines()[-5:]
        raise RuntimeError(f"OCR が途中で終了しました（詳細: {log_path}）\n" + "\n".join(tail))
    return results


# ------------------------------------------------------------
# OCR の出力 -> Markdown
# ------------------------------------------------------------
_BLOCK_RE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]*\])?\s*<\|/det\|>")
_DROP = {"header", "page_number", "footer", "page_header", "page_footer"}


def _split_blocks(text: str):
    """[(種類, [x0, y0, x1, y1] or None, 中身), ...]"""
    matches = list(_BLOCK_RE.finditer(text))
    blocks = []
    for m, nxt in zip(matches, matches[1:] + [None]):
        body = text[m.end():nxt.start() if nxt else len(text)].strip()
        try:
            box = json.loads(m.group(2)) if m.group(2) else None
        except ValueError:
            box = None
        blocks.append((m.group(1), box, body))
    if not matches and text.strip():
        blocks.append(("text", None, text.strip()))
    return blocks


def html_table_to_markdown(table_html: str) -> str:
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.DOTALL | re.IGNORECASE)
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("|", "\\|").replace("\n", " ").strip()
                 for c in cells]
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["|" + "|".join(rows[0]) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines += ["|" + "|".join(r) + "|" for r in rows[1:]]
    return "\n".join(lines)


def _looks_like_code(body: str) -> bool:
    lines = body.splitlines()
    if len(lines) < 3:
        return False
    signs = sum(1 for l in lines if re.match(r"^\s{2,}\S", l) or re.search(r"[{};]\s*$", l))
    return signs >= len(lines) / 3


def _crop_image(page, box, dpi: int) -> Optional[str]:
    """正規化座標（0〜999）の範囲をページから切り出し、data URI にする"""
    import pymupdf

    r = page.rect
    x0, y0, x1, y1 = box
    clip = pymupdf.Rect(r.x0 + r.width * x0 / 999, r.y0 + r.height * y0 / 999,
                        r.x0 + r.width * x1 / 999, r.y0 + r.height * y1 / 999)
    if clip.width < 8 or clip.height < 8:
        return None
    png = page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def ocr_to_markdown(text: str, page=None, with_images: bool = True, image_dpi: int = 200) -> str:
    """OCR の出力を Markdown にする。page（pymupdf の Page）があれば図を切り出して埋め込む"""
    out = []
    for kind, box, body in _split_blocks(text):
        if kind in _DROP:
            continue
        if kind == "image":
            if with_images and page is not None and box and len(box) == 4:
                uri = _crop_image(page, box, image_dpi)
                if uri:
                    out.append(f"![]({uri})")
            continue
        if not body:
            continue
        if kind == "title":
            out.append("# " + " ".join(body.split()))
        elif kind == "table" and "<table" in body.lower():
            out.append(html_table_to_markdown(body))
        elif kind == "equation":
            # 数式は訳さず、LaTeX のまま残す
            out.append("```latex\n" + body.strip() + "\n```")
        elif _looks_like_code(body):
            out.append("```\n" + body.rstrip() + "\n```")
        else:
            # 段落の中の改行（行の折り返し）はつなげる。ハイフンで切れた単語は戻す
            para = re.sub(r"-\n(?=[a-z])", "", body)
            out.append(re.sub(r"\s*\n\s*", " ", para).strip())
    return "\n\n".join(out)
