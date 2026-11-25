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
MATH_OPERATOR_RE = re.compile(f"[{re.escape(MATH_OPERATORS)}]")
WIDTH_AND_HYPHENS = str.maketrans({
    **{ord(char) + 0xFEE0: char for char in map(chr, range(0x21, 0x7F))},
    0x3000: " ", 0x2010: "-", 0x2011: "-", 0: " ", 0xE000: " ",
})

QUOTED_SPAN_RE = re.compile(r'["“]([^"”]{2,64})["”]|《([^》]{2,64})》')
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
ENGLISH_PHRASE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){1,3}")
HAN_RUN_RE = re.compile(f"[{HAN}]{{2,}}")

EN_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "compare", "comparison",
    "describe", "difference", "differences", "do", "does", "explain", "for", "from", "give",
    "how", "in", "into", "is", "it", "its", "me", "of", "on", "or", "please", "prove",
    "show", "summarize", "summary", "tell", "that", "the", "their", "these", "they", "this",
    "those", "to", "use", "using", "vs", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "would",
})

ZH_SPLIT_RE = re.compile(
    r"(?:请问一下|请问|请教一下|请教|请帮我|帮我|给我|能否|可以|麻烦|"
    r"详细讲解一下|深入讲解一下|讲解一下|详细讲解|讲解|讲讲|讲一下|谈谈|说说|"
    r"详细解释一下|解释一下|请解释|总结一下|请总结|介绍一下|请介绍|分析一下|推导一下|证明一下|梳理一下|"
    r"告诉我|什么是|如何理解|怎么理解|有什么优势|有什么优缺点|优缺点|适用场景|主要方法|"
    r"在不同教材里|在不同教材中|不同教材|各本教材|各书|怎么对比|如何对比|对比一下|比较一下|对比|比较|"
    r"有什么区别|有何区别|的区别|的异同|的联系|的原理|的推导|的步骤|的定义|的性质|的证明|的方法|的概念|的公式|的内容|"
    r"是如何|如何逐步|如何|怎么|为什么|是什么|什么|时候|它在|它在求解|在求解|相比|"
    r"根据|关于|主要|以及|或者|与|和|跟|中|里|的|了|吗|呢|啊|呀|把|被|对|将)"
)

ZH_STOPWORDS = frozenset({
    "什么", "如何", "怎么", "为什么", "一下", "教材", "总结", "对比", "比较",
    "原理", "优缺点", "适用", "场景", "主要", "方法", "内容", "定义", "区别",
    "优势", "特点", "步骤", "过程", "举例", "这个", "那个", "它们", "相比",
    "逐步", "生成", "求解", "计算", "问题", "方面", "部分", "简介", "概述",
})

NL_SIGNAL_RE = re.compile(
    r"(?:[？?。！!，,；;、]|请问|什么|如何|怎么|总结|介绍|讲解|对比|比较|为什么|优缺点|适用场景|"
    r"\b(?:what|how|why|compare|explain|summarize|describe|does|between)\b)",
    re.I,
)


def keyword_terms(text: str) -> str:
    """Prepare positional Han tokens while retaining whole Latin words."""
    normalized = unicodedata.normalize("NFC", text.translate(WIDTH_AND_HYPHENS))
    return SEPARATE.sub(r" \1 ", normalized)


def keyword_title_terms(title_context: str) -> str:
    # A reserved phrase boundary prevents matches from spanning different titles.
    return " \ue000 ".join(keyword_terms(title) for title in json.loads(title_context))


def fts_phrase(terms: str) -> str:
    return '"' + terms.replace('"', '""') + '"'


def is_explicit_exact_query(query: str) -> bool:
    stripped = query.strip()
    if len(stripped) >= 3 and (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("“") and stripped.endswith("”"))
    ):
        return True
    if MATH_OPERATOR_RE.search(stripped):
        return True
    # Treat raw FTS5 / SQL syntax probes as exact literal phrases rather than tokenizing them.
    if any(marker in stripped for marker in ('"', "--", ";", "NEAR(", "*")):
        return True
    return False


def looks_like_natural_language(query: str) -> bool:
    stripped = query.strip()
    if NL_SIGNAL_RE.search(stripped):
        return True
    words = ENGLISH_TOKEN_RE.findall(stripped)
    if len(words) >= 4:
        return True
    han_runs = HAN_RUN_RE.findall(stripped)
    if any(len(run) >= 7 for run in han_runs) or len(han_runs) >= 2:
        return True
    return False


def extract_bm25_terms(query: str, *, max_terms: int = 12) -> tuple[str, ...]:
    """Extract meaningful multi-char Chinese terms, whole English words/phrases, and quoted spans.

    Never splits Chinese into single-character OR terms (prevents FTS5 noise explosion).
    """
    normalized = unicodedata.normalize("NFC", query.translate(WIDTH_AND_HYPHENS)).strip()
    candidates: list[str] = []

    # 1. Explicit quoted spans (excluding book titles ending in .pdf/.docx/.md)
    for m in QUOTED_SPAN_RE.finditer(normalized):
        span = (m.group(1) or m.group(2) or "").strip()
        if span and not re.search(r"\.(?:pdf|docx|md)$", span, re.I):
            candidates.append(span)

    # Remove book filenames like 《非结构网格生成技术调研.pdf》 before term extraction
    cleaned = re.sub(r"《[^》]*\.(?:pdf|docx|md)》", " ", normalized, flags=re.I)

    # 2. Multi-word English technical phrases (excluding stopword-only edges)
    for m in ENGLISH_PHRASE_RE.finditer(cleaned):
        words = [w for w in m.group(0).split() if w.lower() not in EN_STOPWORDS and len(w) >= 2]
        if len(words) >= 2:
            candidates.append(" ".join(words[:3]))

    # 3. Individual meaningful English tokens / acronyms (e.g., ADI, MLE, Delaunay, Bezier)
    for tok in ENGLISH_TOKEN_RE.findall(cleaned):
        if tok.lower() in EN_STOPWORDS:
            continue
        if len(tok) >= 2:
            candidates.append(tok)

    # 4. Chinese multi-character terms after removing conversational fillers
    for segment in ZH_SPLIT_RE.split(cleaned):
        for run in HAN_RUN_RE.findall(segment):
            if run in ZH_STOPWORDS:
                continue
            if 2 <= len(run) <= 8:
                candidates.append(run)
            if len(run) >= 5:
                # Add informative overlapping 2-4 char sub-compounds without single-char noise
                for width in (4, 3, 2):
                    if len(run) > width:
                        head = run[:width]
                        tail = run[-width:]
                        if head not in ZH_STOPWORDS:
                            candidates.append(head)
                        if tail not in ZH_STOPWORDS:
                            candidates.append(tail)
                # For 6+ char compounds also extract middle 2-char core (e.g. 非结构网格 -> 结构网格 / 网格)
                if len(run) >= 6:
                    mid = run[len(run) // 2 - 1 : len(run) // 2 + 1]
                    if len(mid) == 2 and mid not in ZH_STOPWORDS:
                        candidates.append(mid)

    unique: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        t = " ".join(term.split()).strip()
        if len(t) < 2:
            continue
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(t)
            if len(unique) >= max_terms:
                break
    return tuple(unique)


def build_bm25_or_query(query: str, *, extra_terms: tuple[str, ...] = ()) -> str | None:
    """Build a safe FTS5 OR expression from extracted terms and optional bilingual expansions."""
    terms = list(extract_bm25_terms(query))
    seen = {t.casefold() for t in terms}
    for extra in extra_terms:
        cleaned = " ".join(extra.strip().split())
        if len(cleaned) >= 2 and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            terms.append(cleaned)
    clauses: list[str] = []
    for term in terms[:16]:
        kt = keyword_terms(term).strip()
        if kt and any(c.isalnum() for c in kt):
            clauses.append(fts_phrase(kt))
    if not clauses:
        return None
    return " OR ".join(clauses)
