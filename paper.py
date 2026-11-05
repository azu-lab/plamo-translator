"""PDF論文を章立てを保ったまま翻訳し、Markdown / HTML に書き出す

流れ:
    PDF --pymupdf4llm--> Markdown --parse_blocks--> ブロック列
        --translate(バッチ)--> 訳文付きブロック --render_markdown / render_html--> 出力

翻訳モデルには依存せず、`translate_many(list[str]) -> list[str]` を外から受け取る。
"""

import html
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pymupdf4llm
from markdown_it import MarkdownIt


# ------------------------------------------------------------
# ブロック
# ------------------------------------------------------------
@dataclass
class Block:
    kind: str                  # title / heading / para / list / quote / table / image
    text: str = ""             # 原文（装飾を除いたもの）。table / image では Markdown そのまま
    level: int = 0             # heading の深さ (2=章, 3=節, 4=項)
    number: str = ""           # heading の番号 ("1", "2.1", "III", "A")
    marker: str = ""           # list の記号 ("-", "1.")
    translate: bool = True     # False なら原文のまま出す（前付け・参考文献など）
    in_refs: bool = False      # 参考文献の章に属するか
    segments: List[str] = field(default_factory=list)  # 翻訳単位に分割した原文
    translated: Optional[str] = None


# ------------------------------------------------------------
# PDF -> Markdown
# ------------------------------------------------------------
def pdf_to_markdown(pdf_path: str, with_images: bool = True) -> str:
    options = dict(
        use_ocr=False,        # テキストを持つPDFならOCR不要（遅い）
        embed_images=with_images,
        show_progress=False,
    )
    try:
        # pymupdf4llm の解析結果（ページごとのブロック）を受け取り、
        # 2カラムの読み順を直してから Markdown にする
        from pymupdf4llm.helpers import document_layout

        parsed = document_layout.parse_document(pdf_path, force_text=True, **options)
        for page in parsed.pages:
            page.boxes = order_two_columns(page.boxes, page.width)
        return parsed.to_markdown(
            header=False,     # ページヘッダー・フッターを除く
            footer=False,
            embed_images=with_images,
            show_progress=False,
        )
    except (ImportError, AttributeError, TypeError):
        # pymupdf4llm の内部APIが変わった場合は、読み順の補正なしで変換する
        return pymupdf4llm.to_markdown(pdf_path, header=False, footer=False, **options)


_MARGIN_CLASSES = ("page-header", "page-footer")


def order_two_columns(boxes, page_width):
    """2カラムのページで、ブロックを「左カラム → 右カラム」の順に並べ直す

    pymupdf4llm は、左右のカラムの段落を高さ順に交互に並べてしまうことがある
    （論文の1ページ目で、左にAbstract、右にIntroductionがあるときなど）。

    - 横幅いっぱいのブロック（タイトル・著者欄・2カラムにまたがる図表）と、
      それに高さが重なるブロックは「帯」として、上から順・左から順に置く
    - 帯と帯のあいだは、左カラムを上から下へ、続いて右カラムを上から下へ置く
    - 2カラムと判定できないページは元の順番のまま返す
    """
    mid = page_width / 2
    tol = page_width * 0.05

    def is_left(b):
        return b.x1 <= mid + tol

    def is_right(b):
        return b.x0 >= mid - tol

    body = [b for b in boxes if b.boxclass not in _MARGIN_CLASSES]
    headers = [b for b in boxes if b.boxclass == "page-header"]
    footers = [b for b in boxes if b.boxclass == "page-footer"]

    # 左右それぞれに2つ以上ブロックがあるページだけを2カラムとみなす
    if sum(map(is_left, body)) < 2 or sum(map(is_right, body)) < 2:
        return boxes

    # 横幅いっぱいのブロックから「帯」（高さの範囲）を作る
    bands = []
    for b in sorted((b for b in body if not is_left(b) and not is_right(b)),
                    key=lambda b: b.y0):
        if bands and b.y0 <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], b.y1)
        else:
            bands.append([b.y0, b.y1])

    def band_of(b):
        """ブロックが帯と高さで半分以上重なれば、その帯の番号を返す"""
        height = max(b.y1 - b.y0, 1e-6)
        for i, (y0, y1) in enumerate(bands):
            if min(b.y1, y1) - max(b.y0, y0) >= height / 2:
                return i
        return None

    # 帯（i）と、帯 i の直前の区間（"gap", i）にブロックを振り分ける。
    # 最後の帯より下は ("gap", len(bands))
    groups = {}
    for b in body:
        i = band_of(b)
        if i is not None:
            key = (i, 1)
        else:
            gap = sum(1 for y0, _ in bands if y0 <= b.y0)
            key = (gap, 0)
        groups.setdefault(key, []).append(b)

    ordered = []
    for key in sorted(groups):
        members = groups[key]
        if key[1] == 1:   # 帯: 上から順、同じ高さなら左から
            ordered += sorted(members, key=lambda b: (round(b.y0), b.x0))
        else:             # 帯のあいだ: 左カラム → 右カラム
            ordered += sorted((b for b in members if is_left(b)), key=lambda b: b.y0)
            ordered += sorted((b for b in members if not is_left(b)), key=lambda b: b.y0)

    return headers + ordered + footers


# ------------------------------------------------------------
# 装飾の除去
# ------------------------------------------------------------
_TAG_RE = re.compile(r"</?(?:mark|u|b|i|em|strong|span|sub)[^>]*>")
_SUP_RE = re.compile(r"<sup>(.*?)</sup>")


def clean_inline(text: str) -> str:
    """pymupdf4llm が付ける太字・斜体・下線などを除いてプレーンテキストにする"""
    text = text.replace("�", "")                 # ORCIDアイコンなどの化け
    text = _SUP_RE.sub(r"\1", text)
    text = _TAG_RE.sub("", text)
    text = text.replace("**", "").replace("~~", "")
    # _斜体_ だけを外す（snake_case の _ は残す）
    text = re.sub(r"(?<![\w\\])_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ------------------------------------------------------------
# 見出しの判定
# ------------------------------------------------------------
_KNOWN_SECTIONS = re.compile(
    r"^(abstract|introduction|related works?|background|preliminar(y|ies)|"
    r"conclusions?|discussion|evaluation|experiments?|references|bibliography|"
    r"acknowledge?ments?|appendix|appendices|future work|limitations)\b",
    re.IGNORECASE,
)
# 1語だけの見出しは原則として訳さない（固有名詞の誤訳を防ぐ）が、
# 論文でよく使われる一般的な章名は訳す
_COMMON_SINGLE_WORD_SECTIONS = re.compile(
    r"^(results?|methods?|methodology|implementation|design|overview|motivation|"
    r"summary|analysis|setup|evaluation|discussion|conclusions?|introduction|"
    r"background|preliminaries|approach|experiments?)$",
    re.IGNORECASE,
)
_REFERENCES = re.compile(r"^(references|bibliography|参考文献)\s*$", re.IGNORECASE)

_ROMAN = r"(?:X{0,3})(?:IX|IV|V?I{0,3})"


def classify_heading(text: str):
    """見出しなら (level, number, title)、章・節でなければ None"""
    if len(text) > 120:
        return None

    m = re.match(rf"^({_ROMAN})\.\s+(\S.*)$", text)          # IEEE: "III. METHOD"
    if m and m.group(1):
        return 2, m.group(1), m.group(2)

    m = re.match(r"^([A-Z])\.\s+(\S.*)$", text)               # IEEE: "A. Model"
    if m:
        return 3, m.group(1), m.group(2)

    m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Za-z].*)$", text)  # "2.1 Model"
    if m:
        number = m.group(1)
        return min(2 + number.count("."), 5), number, m.group(2)

    m = re.match(r"^(appendix\s+[A-Z0-9]+)[.:]?\s*(.*)$", text, re.IGNORECASE)
    if m:
        return 2, m.group(1), m.group(2) or m.group(1)

    if _KNOWN_SECTIONS.match(text):
        return 2, "", text

    return None


# ------------------------------------------------------------
# Markdown -> ブロック列
# ------------------------------------------------------------
_LIST_RE = re.compile(r"^\s*([-*+•]|\d{1,3}[.)])\s+(.*)$", re.DOTALL)
_END_OF_SENTENCE = re.compile(r"[.!?:;。．！？)\]”\"']\s*$")


def _split_raw_blocks(md: str) -> List[str]:
    """空行区切り。ただし表（| で始まる行の連続）は1ブロックにまとめる"""
    blocks, current = [], []
    for line in md.splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if current and line.lstrip().startswith("|") != current[-1].lstrip().startswith("|"):
            blocks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def parse_blocks(md: str) -> List[Block]:
    blocks: List[Block] = []

    for raw in _split_raw_blocks(md):
        stripped = raw.strip()

        if stripped.startswith("|"):
            blocks.append(Block("table", stripped, translate=False))
            continue
        if stripped.startswith("!["):
            blocks.append(Block("image", stripped, translate=False))
            continue
        if stripped.startswith("```"):
            blocks.append(Block("table", stripped, translate=False))  # コードはそのまま
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped, re.DOTALL)
        if m:
            text = clean_inline(m.group(2))
            if not text or text.lower() == "blank":
                continue
            if len(m.group(1)) == 1 and not any(b.kind == "title" for b in blocks):
                blocks.append(Block("title", text))
                continue
            heading = classify_heading(text)
            if heading:
                level, number, title = heading
                # 1語だけの見出し（Rust, GEDF, MRM など）は文脈なしだと誤訳しやすい
                # （例: "Rust" -> 「錆」）ので原文のままにする
                single_word = (len(title.split()) == 1
                               and not _KNOWN_SECTIONS.match(title)
                               and not _COMMON_SINGLE_WORD_SECTIONS.match(title))
                blocks.append(Block("heading", title, level=level, number=number,
                                    translate=not single_word))
            else:
                # 著者名・「TABLE II」などは見出し扱いしない
                blocks.append(Block("para", text))
            continue

        if stripped.startswith(">"):
            text = clean_inline(re.sub(r"^>\s?", "", stripped, flags=re.MULTILINE))
            if text:
                blocks.append(Block("quote", text))
            continue

        m = _LIST_RE.match(stripped)
        if m:
            text = clean_inline(m.group(2).replace("\n", " "))
            if text:
                marker = "-" if not m.group(1)[0].isdigit() else m.group(1)
                blocks.append(Block("list", text, marker=marker))
            continue

        text = clean_inline(stripped.replace("\n", " "))
        if text:
            _append_paragraph(blocks, text)

    _mark_untranslated(blocks)
    return blocks


def _append_paragraph(blocks: List[Block], text: str):
    """ページ・段組みの境目で切れた段落を前の段落につなげる"""
    if text[0].islower():
        # 図・表・脚注を挟んで切れていることがあるので、それらは飛ばして探す
        for prev in reversed(blocks):
            if prev.kind in ("image", "table", "quote"):
                continue
            if prev.kind in ("para", "list") and not _END_OF_SENTENCE.search(prev.text):
                prev.text = f"{prev.text} {text}"
                return
            break
    blocks.append(Block("para", text))


def _mark_untranslated(blocks: List[Block]):
    """前付け（著者・所属）と参考文献は原文のまま残す"""
    has_sections = sum(b.kind == "heading" for b in blocks) >= 2
    in_front = has_sections        # 論文でなければ全文を訳す
    in_refs = False
    in_abstract = False

    for b in blocks:
        if b.kind == "heading":
            in_front = False
            if b.level <= 2:
                in_refs = bool(_REFERENCES.match(b.text))
            if in_refs and not _REFERENCES.match(b.text):
                b.translate = False
            continue
        if b.kind in ("table", "image"):
            continue
        if b.kind == "title":
            continue
        if in_front:
            # アブストラクト（以降の段落も含む）とキーワードは訳す。著者・所属は原文のまま
            if re.match(r"^(abstract|index terms|keywords)\b", b.text, re.IGNORECASE):
                in_abstract = True
            b.translate = in_abstract
        elif in_refs:
            b.translate = False
            b.in_refs = True


# ------------------------------------------------------------
# 翻訳
# ------------------------------------------------------------
def split_long(text: str, max_chars: int = 1500) -> List[str]:
    """長すぎる段落を文単位で分割する"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?。])\s+(?=[A-Z0-9\[(“\"])", text)
    parts, current = [], ""
    for s in sentences:
        while len(s) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(s[:max_chars])
            s = s[max_chars:]
        if current and len(current) + len(s) + 1 > max_chars:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}" if current else s
    if current:
        parts.append(current)
    return parts


def translate_blocks(
    blocks: List[Block],
    translate_many: Callable[[List[str], Callable[[int, int], None]], List[str]],
    translate_references: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
):
    targets = []
    for b in blocks:
        if b.kind in ("table", "image"):
            continue
        if not b.translate and not (translate_references and b.in_refs):
            continue
        b.segments = split_long(b.text)
        targets.append(b)

    texts = [s for b in targets for s in b.segments]
    results = translate_many(texts, progress) if texts else []

    i = 0
    for b in targets:
        n = len(b.segments)
        b.translated = "".join(results[i:i + n]) if n > 1 else results[i]
        i += n


# ------------------------------------------------------------
# 出力
# ------------------------------------------------------------
def _heading_line(b: Block) -> str:
    title = b.translated or b.text
    if not b.number:
        number = ""
    elif b.number.isdigit() or "." in b.number or b.number.lower().startswith("appendix"):
        number = f"{b.number} "        # "1 ", "2.1 "
    else:
        number = f"{b.number}. "       # IEEE形式 "III. ", "A. "
    # 章名は原文も残しておく（論文を読むときに対応が取りやすい）
    orig = f"（{b.text}）" if b.translated and b.translated != b.text else ""
    return f"{'#' * b.level} {number}{title}{orig}"


def _original_details(text: str) -> str:
    return f"<details><summary>原文</summary>\n\n{text}\n\n</details>"


def render_markdown(blocks: List[Block], source_name: str, bilingual: bool = False) -> str:
    out: List[str] = []
    pending_originals: List[str] = []   # 箇条書きの原文はリストの後ろにまとめる
    prev_kind = ""

    def flush_list_originals():
        if pending_originals:
            out.append(_original_details("\n".join(f"- {t}" for t in pending_originals)))
            pending_originals.clear()

    for b in blocks:
        if b.kind != "list":
            flush_list_originals()

        if b.kind == "title":
            out.append(f"# {b.translated or b.text}")
            if b.translated:
                out.append(f"*{b.text}*")
            out.append(f"> 翻訳元: `{source_name}`（PLaMo 3 Translate による機械翻訳）")
        elif b.kind == "heading":
            out.append(_heading_line(b))
        elif b.kind in ("table", "image"):
            out.append(b.text)
        elif b.kind == "list":
            line = f"{b.marker} {b.translated or b.text}"
            # 連続する箇条書きは空行を挟まず1つのリストにする
            if prev_kind == "list":
                out[-1] += "\n" + line
            else:
                out.append(line)
            if bilingual and b.translated:
                pending_originals.append(b.text)
        elif b.kind == "quote":
            out.append("> " + (b.translated or b.text))
        else:
            out.append(b.translated or b.text)
            if bilingual and b.translated:
                out.append(_original_details(b.text))
        prev_kind = b.kind

    flush_list_originals()
    return "\n\n".join(out) + "\n"


_HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #fdfdfc; --fg: #1f2328; --muted: #656d76; --line: #d8dee4;
  --accent: #0a5bd3; --side: #f5f6f8; --code: #f0f2f4;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #15171a; --fg: #e3e5e8; --muted: #9aa1a9; --line: #30363d;
    --accent: #6ea8fe; --side: #1c1f23; --code: #23272d;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Yu Gothic UI", Meiryo, sans-serif;
  line-height: 1.9; font-size: 16px;
}}
.layout {{ display: flex; align-items: flex-start; }}
nav {{
  position: sticky; top: 0; height: 100vh; overflow-y: auto; width: 300px; flex: none;
  background: var(--side); border-right: 1px solid var(--line); padding: 24px 16px; font-size: 13.5px;
  line-height: 1.6;
}}
nav .toc-title {{ font-weight: 700; margin-bottom: 8px; color: var(--muted); }}
nav a {{ display: block; color: var(--fg); text-decoration: none; padding: 3px 6px; border-radius: 4px; }}
nav a:hover {{ background: var(--line); }}
nav a.l3 {{ padding-left: 20px; color: var(--muted); }}
nav a.l4, nav a.l5 {{ padding-left: 34px; color: var(--muted); font-size: 12.5px; }}
main {{ flex: 1; min-width: 0; max-width: 860px; margin: 0 auto; padding: 32px 24px 96px; }}
h1 {{ font-size: 1.7em; line-height: 1.4; margin: 0 0 4px; }}
h1 + p em {{ color: var(--muted); font-style: normal; }}
h2 {{ font-size: 1.35em; margin-top: 2.4em; padding-bottom: 6px; border-bottom: 2px solid var(--line); }}
h3 {{ font-size: 1.15em; margin-top: 2em; }}
h4, h5 {{ font-size: 1.02em; margin-top: 1.6em; }}
h2, h3, h4, h5 {{ scroll-margin-top: 16px; }}
p {{ margin: 0.9em 0; text-align: justify; }}
blockquote {{ margin: 1em 0; padding: 4px 14px; border-left: 4px solid var(--line); color: var(--muted); font-size: 0.92em; }}
img {{ max-width: 100%; height: auto; display: block; margin: 16px auto; background: #fff; border-radius: 4px; }}
table {{ border-collapse: collapse; margin: 16px 0; font-size: 0.85em; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; }}
th {{ background: var(--side); }}
code {{ background: var(--code); padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
details {{ margin: -0.4em 0 1.2em; font-size: 0.88em; color: var(--muted); }}
details summary {{ cursor: pointer; user-select: none; width: fit-content; }}
details[open] {{ border-left: 3px solid var(--line); padding-left: 12px; }}
.toggle {{ position: fixed; right: 16px; bottom: 16px; padding: 8px 12px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--side); color: var(--fg); cursor: pointer; font-size: 13px; }}
@media (max-width: 900px) {{
  nav {{ display: none; }}
  main {{ padding: 20px 16px 80px; }}
}}
</style>
</head>
<body>
<div class="layout">
<nav><div class="toc-title">目次</div>{toc}</nav>
<main>
{body}
</main>
</div>
{toggle}
</body>
</html>
"""

_TOGGLE = """<button class="toggle" onclick="
  const d=document.querySelectorAll('details'); const open=![...d].every(x=>x.open);
  d.forEach(x=>x.open=open);">原文を全て開閉</button>"""


def render_html(markdown_text: str, title: str) -> str:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = md.parse(markdown_text)

    toc = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag in ("h2", "h3", "h4", "h5"):
            anchor = f"sec-{len(toc)}"
            tok.attrSet("id", anchor)
            text = tokens[i + 1].content
            toc.append(
                f'<a class="l{tok.tag[1]}" href="#{anchor}">{html.escape(text)}</a>'
            )

    body = md.renderer.render(tokens, md.options, {})
    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        toc="\n".join(toc),
        body=body,
        toggle=_TOGGLE if "<details>" in markdown_text else "",
    )
