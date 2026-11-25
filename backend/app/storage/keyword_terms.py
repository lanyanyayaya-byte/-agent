import json
import re
import unicodedata

# Keep ASCII compounds intact and give common formula operators their own positions.
MATH_OPERATORS = "+−×÷=≠≈≤≥∑∏∫√±∂∇∞^<>"
TOKENIZER = f"unicode61 remove_diacritics 0 tokenchars '-{MATH_OPERATORS}'"
HAN = (
    "\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002ffff\U00030000-\U000323af"
)
SEPARATE = re.compile(f"([{HAN}{re.escape(MATH_OPERATORS)}])")
WIDTH_AND_HYPHENS = str.maketrans({
    **{ord(char) + 0xFEE0: char for char in map(chr, range(0x21, 0x7F))},
    0x3000: " ", 0x2010: "-", 0x2011: "-", 0: " ", 0xE000: " ",
})


def keyword_terms(text: str) -> str:
    """Prepare positional Han tokens while retaining whole Latin words."""
    normalized = unicodedata.normalize("NFC", text.translate(WIDTH_AND_HYPHENS))
    return SEPARATE.sub(r" \1 ", normalized)


def keyword_title_terms(title_context: str) -> str:
    # A reserved phrase boundary prevents matches from spanning different titles.
    return " \ue000 ".join(keyword_terms(title) for title in json.loads(title_context))


def fts_phrase(terms: str) -> str:
    return '"' + terms.replace('"', '""') + '"'
