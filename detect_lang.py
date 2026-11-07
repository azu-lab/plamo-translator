"""原文の言語を判定する（テキスト翻訳で「自動判定」のとき、判定結果を表示し、入れ替えに使う）

翻訳そのものは PLaMo の自動判定に任せる。ここでの判定は、画面に出す言語と、
⇄ で入れ替えるときの翻訳先を決めるためだけに使う。

判定には lingua（Apache-2.0）を使い、アプリが扱う言語だけに候補を絞る。
"""

import re
from typing import Optional

from lingua import Language, LanguageDetectorBuilder

_CODES = {
    Language.ENGLISH: "en", Language.JAPANESE: "ja", Language.CHINESE: "zh",
    Language.KOREAN: "ko", Language.FRENCH: "fr", Language.GERMAN: "de",
    Language.SPANISH: "es", Language.ITALIAN: "it", Language.PORTUGUESE: "pt",
    Language.RUSSIAN: "ru", Language.VIETNAMESE: "vi", Language.THAI: "th",
    Language.INDONESIAN: "id",
}
_detector = None

# 簡体字と繁体字で字形が違う、よく使う字
_SIMPLIFIED = set("这个们来说时会为国学对发与经过还后么开关问题种体点应义现实见电话书长区认让变听买气东车无业产动将样电们图机")
_TRADITIONAL = set("這個們來說時會為國學對發與經過還後麼開關問題種體點應義現實見電話書長區認讓變聽買氣東車無業產動將樣電們圖機")

_KANA = re.compile(r"[぀-ヿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


def _get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(*_CODES).build()
    return _detector


def _clean(text: str) -> str:
    """画像（data URI を含む）・コード・URL を除き、先頭 2000 文字だけにする"""
    text = re.sub(r"```.*?(```|$)", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return text[:2000]


def detect_language(text: str) -> Optional[str]:
    """アプリの言語コード（en / ja / zh-hans / zh-hant など）を返す。判定できなければ None"""
    text = _clean(text)
    if not text.strip():
        return None
    # かなが入っていれば日本語（漢字だけの短い文は中国語と区別できないため、先に見る）
    if _KANA.search(text):
        return "ja"
    values = _get_detector().compute_language_confidence_values(text)
    if not values or values[0].value == 0:
        return None
    top = values[0]
    lang = top.language
    # 短いラテン文字の文（"Hello" など）は確信度が低く、伊・西・葡と取り違えやすい。
    # 確信度が低く、英語も候補の上位にあれば英語とみなす
    if (_LATIN.search(text) and top.value < 0.4 and
            any(v.language == Language.ENGLISH for v in values[:3])):
        lang = Language.ENGLISH
    if lang == Language.CHINESE:
        simp = sum(ch in _SIMPLIFIED for ch in text)
        trad = sum(ch in _TRADITIONAL for ch in text)
        return "zh-hant" if trad > simp else "zh-hans"
    return _CODES[lang]
