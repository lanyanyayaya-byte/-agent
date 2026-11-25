"""macOS Vision.framework OCR fallback for scanned PDF textbooks without an embedded text layer."""

import ctypes
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from typing import TYPE_CHECKING

from app.parsing.models import ContentBlock, ParseWarning, PdfSourceLocation, Section

if TYPE_CHECKING:
    from app.parsing.pdf_layout import Box, Line

OCR_HEADING_PATTERN = re.compile(
    r"^(?:第[零〇一二三四五六七八九十百0-9]+[章篇节部分]|§\s*[0-9]+(?:\.[0-9]+)*|[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}\s+\S|"
    r"Chapter\s+\d+|CHAPTER\s+\d+|目录|参考文献|附录|习题)",
)


class _CGRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("w", ctypes.c_double),
        ("h", ctypes.c_double),
    ]


class _DarwinVisionBindings:
    def __init__(self) -> None:
        self.objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        self.cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Vision.framework/Vision")

        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
        self.objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]

        self.cg.CGDataProviderCreateWithData.restype = ctypes.c_void_p
        self.cg.CGDataProviderCreateWithData.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p,
        ]
        self.cg.CGDataProviderRelease.argtypes = [ctypes.c_void_p]
        self.cg.CGPDFDocumentCreateWithProvider.restype = ctypes.c_void_p
        self.cg.CGPDFDocumentCreateWithProvider.argtypes = [ctypes.c_void_p]
        self.cg.CGPDFDocumentRelease.argtypes = [ctypes.c_void_p]
        self.cg.CGPDFDocumentGetNumberOfPages.restype = ctypes.c_size_t
        self.cg.CGPDFDocumentGetNumberOfPages.argtypes = [ctypes.c_void_p]
        self.cg.CGPDFDocumentGetPage.restype = ctypes.c_void_p
        self.cg.CGPDFDocumentGetPage.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.cg.CGPDFPageGetBoxRect.restype = _CGRect
        self.cg.CGPDFPageGetBoxRect.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.cg.CGColorSpaceCreateDeviceGray.restype = ctypes.c_void_p
        self.cg.CGColorSpaceRelease.argtypes = [ctypes.c_void_p]
        self.cg.CGBitmapContextCreate.restype = ctypes.c_void_p
        self.cg.CGBitmapContextCreate.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32,
        ]
        self.cg.CGContextSetGrayFillColor.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self.cg.CGContextFillRect.argtypes = [ctypes.c_void_p, _CGRect]
        self.cg.CGContextScaleCTM.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self.cg.CGContextDrawPDFPage.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.cg.CGBitmapContextCreateImage.restype = ctypes.c_void_p
        self.cg.CGBitmapContextCreateImage.argtypes = [ctypes.c_void_p]
        self.cg.CGContextRelease.argtypes = [ctypes.c_void_p]
        self.cg.CGImageRelease.argtypes = [ctypes.c_void_p]

        self.msg_ptr = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_ptr_ptr = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_ptr_ptr_ptr = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_ptr_ulong = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
        )(("objc_msgSend", self.objc))
        self.msg_ulong = ctypes.CFUNCTYPE(
            ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_void_long = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
        )(("objc_msgSend", self.objc))
        self.msg_void_bool = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool,
        )(("objc_msgSend", self.objc))
        self.msg_void_ptr = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_bool_ptr_ptr = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_cstr = ctypes.CFUNCTYPE(
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_rect = ctypes.CFUNCTYPE(
            _CGRect, ctypes.c_void_p, ctypes.c_void_p,
        )(("objc_msgSend", self.objc))
        self.msg_ptr_ptr_ulong = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong,
        )(("objc_msgSend", self.objc))

        self.S_ALLOC = self.objc.sel_registerName(b"alloc")
        self.S_INIT = self.objc.sel_registerName(b"init")
        self.S_UTF8 = self.objc.sel_registerName(b"stringWithUTF8String:")
        self.S_ARR_OBJ = self.objc.sel_registerName(b"arrayWithObject:")
        self.S_ARR_OBJS = self.objc.sel_registerName(b"arrayWithObjects:count:")
        self.S_DICT = self.objc.sel_registerName(b"dictionary")
        self.S_INIT_IMG = self.objc.sel_registerName(b"initWithCGImage:options:")
        self.S_SET_LEVEL = self.objc.sel_registerName(b"setRecognitionLevel:")
        self.S_SET_LANGS = self.objc.sel_registerName(b"setRecognitionLanguages:")
        self.S_SET_CORR = self.objc.sel_registerName(b"setUsesLanguageCorrection:")
        self.S_PERFORM = self.objc.sel_registerName(b"performRequests:error:")
        self.S_RESULTS = self.objc.sel_registerName(b"results")
        self.S_COUNT = self.objc.sel_registerName(b"count")
        self.S_OBJ_AT = self.objc.sel_registerName(b"objectAtIndex:")
        self.S_TOP = self.objc.sel_registerName(b"topCandidates:")
        self.S_STR = self.objc.sel_registerName(b"string")
        self.S_TO_UTF8 = self.objc.sel_registerName(b"UTF8String")
        self.S_BBOX = self.objc.sel_registerName(b"boundingBox")

        self.C_NSSTRING = self.objc.objc_getClass(b"NSString")
        self.C_NSARRAY = self.objc.objc_getClass(b"NSArray")
        self.C_NSDICT = self.objc.objc_getClass(b"NSDictionary")
        self.C_VNREQ = self.objc.objc_getClass(b"VNRecognizeTextRequest")
        self.C_VNHANDLER = self.objc.objc_getClass(b"VNImageRequestHandler")


_BINDINGS: _DarwinVisionBindings | None = None


def _get_bindings() -> _DarwinVisionBindings | None:
    global _BINDINGS
    if sys.platform != "darwin":
        return None
    if _BINDINGS is None:
        try:
            _BINDINGS = _DarwinVisionBindings()
        except Exception:
            return None
    return _BINDINGS


def _ocr_page_chunk(
    b: _DarwinVisionBindings,
    provider: int,
    page_numbers: list[int],
    scale: float = 1.35,
) -> dict[int, tuple[float, float, list[tuple[str, tuple[float, float, float, float]]]]]:
    """Run Vision OCR for a batch of 1-based page numbers using a thread-local CGPDFDocumentRef."""
    out: dict[int, tuple[float, float, list[tuple[str, tuple[float, float, float, float]]]]] = {}
    doc = b.cg.CGPDFDocumentCreateWithProvider(provider)
    if not doc:
        return out
    try:
        for page_no in page_numbers:
            pool = b.objc.objc_autoreleasePoolPush()
            try:
                page = b.cg.CGPDFDocumentGetPage(doc, page_no)
                if not page:
                    continue
                rect = b.cg.CGPDFPageGetBoxRect(page, 0)
                page_w = float(rect.w) if rect.w > 0 else 595.0
                page_h = float(rect.h) if rect.h > 0 else 842.0
                w = max(64, int(page_w * scale))
                h = max(64, int(page_h * scale))
                cs = b.cg.CGColorSpaceCreateDeviceGray()
                ctx = b.cg.CGBitmapContextCreate(None, w, h, 8, w, cs, 0)
                b.cg.CGColorSpaceRelease(cs)
                if not ctx:
                    continue
                b.cg.CGContextSetGrayFillColor(ctx, 1.0, 1.0)
                b.cg.CGContextFillRect(ctx, _CGRect(0, 0, w, h))
                b.cg.CGContextScaleCTM(ctx, scale, scale)
                b.cg.CGContextDrawPDFPage(ctx, page)
                img = b.cg.CGBitmapContextCreateImage(ctx)
                b.cg.CGContextRelease(ctx)
                if not img:
                    continue

                req = b.msg_ptr(b.msg_ptr(b.C_VNREQ, b.S_ALLOC), b.S_INIT)
                b.msg_void_long(req, b.S_SET_LEVEL, 0)  # Accurate
                zh = b.msg_ptr_ptr(b.C_NSSTRING, b.S_UTF8, b"zh-Hans")
                en = b.msg_ptr_ptr(b.C_NSSTRING, b.S_UTF8, b"en-US")
                langs_arr = (ctypes.c_void_p * 2)(zh, en)
                langs = b.msg_ptr_ptr_ulong(b.C_NSARRAY, b.S_ARR_OBJS, langs_arr, 2)
                b.msg_void_ptr(req, b.S_SET_LANGS, langs)
                b.msg_void_bool(req, b.S_SET_CORR, False)

                opts = b.msg_ptr(b.C_NSDICT, b.S_DICT)
                handler = b.msg_ptr_ptr_ptr(b.msg_ptr(b.C_VNHANDLER, b.S_ALLOC), b.S_INIT_IMG, img, opts)
                reqs = b.msg_ptr_ptr(b.C_NSARRAY, b.S_ARR_OBJ, req)
                ok = b.msg_bool_ptr_ptr(handler, b.S_PERFORM, reqs, None)
                b.cg.CGImageRelease(img)
                if not ok:
                    continue

                results = b.msg_ptr(req, b.S_RESULTS)
                count = b.msg_ulong(results, b.S_COUNT) if results else 0
                raw_lines: list[tuple[str, tuple[float, float, float, float]]] = []
                for idx in range(count):
                    obs = b.msg_ptr_ulong(results, b.S_OBJ_AT, idx)
                    cands = b.msg_ptr_ulong(obs, b.S_TOP, 1)
                    if not cands or b.msg_ulong(cands, b.S_COUNT) == 0:
                        continue
                    cand = b.msg_ptr_ulong(cands, b.S_OBJ_AT, 0)
                    ns_str = b.msg_ptr(cand, b.S_STR)
                    raw_cstr = b.msg_cstr(ns_str, b.S_TO_UTF8) if ns_str else None
                    if not raw_cstr:
                        continue
                    text = " ".join(raw_cstr.decode("utf-8", errors="ignore").split())
                    if not text:
                        continue
                    bbox = b.msg_rect(obs, b.S_BBOX)
                    x0 = max(0.0, min(page_w - 1.0, bbox.x * page_w))
                    x1 = max(x0 + 1.0, min(page_w, (bbox.x + bbox.w) * page_w))
                    top = max(0.0, min(page_h - 1.0, (1.0 - (bbox.y + bbox.h)) * page_h))
                    bottom = max(top + 1.0, min(page_h, (1.0 - bbox.y) * page_h))
                    raw_lines.append((text, (round(x0, 3), round(top, 3), round(x1, 3), round(bottom, 3))))
                out[page_no] = (page_w, page_h, raw_lines)
            finally:
                b.objc.objc_autoreleasePoolPop(pool)
    finally:
        b.cg.CGPDFDocumentRelease(doc)
    return out


def ocr_selected_pages(
    data: bytes,
    page_numbers: list[int],
    *,
    scale: float = 1.35,
) -> dict[int, tuple[float, float, list[tuple[str, tuple[float, float, float, float]]]]]:
    """Run macOS Vision OCR ONLY on specific 1-based scanned insert pages of a digital PDF."""
    b = _get_bindings()
    if b is None or not page_numbers:
        return {}
    provider = b.cg.CGDataProviderCreateWithData(None, data, len(data), None)
    if not provider:
        return {}
    try:
        workers = min(4, max(1, len(page_numbers)))
        shards = [page_numbers[i::workers] for i in range(workers)]
        page_map: dict[int, tuple[float, float, list[tuple[str, tuple[float, float, float, float]]]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_ocr_page_chunk, b, provider, shard, scale) for shard in shards if shard]
            for fut in futures:
                page_map.update(fut.result())
        return page_map
    finally:
        b.cg.CGDataProviderRelease(provider)


def _norm_ocr_edge(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.split())).strip().casefold()


def _detect_ocr_running_headers(
    page_map: dict[int, tuple[float, float, list[tuple[str, tuple[float, float, float, float]]]]],
) -> set[str]:
    """Detect repeated running headers/footers across scanned pages before heading classification."""
    if len(page_map) < 4:
        return set()
    from collections import Counter

    edge_counts: Counter[str] = Counter()
    for _pno, (_w, page_h, raw_lines) in page_map.items():
        seen_on_page: set[str] = set()
        for text, box in raw_lines:
            if box[1] <= page_h * 0.11 or box[3] >= page_h * 0.89:
                norm = _norm_ocr_edge(text)
                if norm and len(norm) <= 64 and norm not in seen_on_page:
                    seen_on_page.add(norm)
                    edge_counts[norm] += 1
    threshold = max(3, min(5, int(len(page_map) * 0.16)))
    return {norm for norm, cnt in edge_counts.items() if cnt >= threshold}


def _is_standalone_page_number(text: str) -> bool:
    s = text.strip()
    return bool(re.fullmatch(r"(?:[-—·•]\s*)?\d{1,4}(?:\s*[-—·•])?|(?:第\s*\d{1,4}\s*页)|[ivxlcdmIVXLCDM]{1,6}", s))


def _heading_level_ocr(text: str, size: float, body_size: float) -> int | None:
    if len(text) > 85 or "..." in text or "⋯" in text or "…" in text:
        return None
    if re.match(r"^第[零〇一二三四五六七八九十百0-9]+[章篇部分]", text):
        return 1
    if re.match(r"^(?:第[零〇一二三四五六七八九十百0-9]+节|§\s*[0-9]+)", text):
        return 2
    if re.match(r"^[0-9]{1,2}\.[0-9]{1,2}(?:\.[0-9]{1,2})?\s+[^\d\W]", text) and (
        size >= body_size * 1.08 or len(text) <= 42
    ):
        return 2
    if text in {"目录", "前言", "绪论", "参考文献", "附录", "索引"}:
        return 1
    return None


def ocr_scanned_pdf(
    data: bytes,
    version_id: str,
    page_count: int,
    labels: list[str],
    existing_warnings: list[ParseWarning],
) -> tuple[list[Section], list[ContentBlock], list[ParseWarning]] | None:
    """Extract structured blocks and sections from a scanned PDF using macOS Vision OCR."""
    b = _get_bindings()
    if b is None or page_count < 1:
        return None

    provider = b.cg.CGDataProviderCreateWithData(None, data, len(data), None)
    if not provider:
        return None

    try:
        workers = min(6, max(1, page_count))
        all_pages = list(range(1, page_count + 1))
        shards = [all_pages[i::workers] for i in range(workers)]
        page_map: dict[int, tuple[float, float, list[tuple[str, tuple[float, float, float, float]]]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_ocr_page_chunk, b, provider, shard) for shard in shards if shard]
            for fut in futures:
                page_map.update(fut.result())
    finally:
        b.cg.CGDataProviderRelease(provider)

    running_edges = _detect_ocr_running_headers(page_map)

    sections: list[Section] = []
    active_sections: list[Section] = []
    blocks: list[ContentBlock] = []
    warnings: list[ParseWarning] = [
        w for w in existing_warnings if w.code not in ("PDF_NO_TEXT_LAYER", "PDF_SUSPECTED_SCAN")
    ]

    for physical_page in range(1, page_count + 1):
        label = labels[physical_page - 1] if labels and physical_page - 1 < len(labels) else None
        if label and (len(label) > 64 or not label.isprintable()):
            label = None
        entry = page_map.get(physical_page)
        if not entry or not entry[2]:
            warnings.append(ParseWarning("PDF_NO_TEXT_LAYER", "No usable text was extracted from this page.", physical_page))
            continue

        page_w, page_h, raw_lines = entry
        heights = [box[3] - box[1] for _, box in raw_lines]
        body_size = median(heights) if heights else 12.0
        # Sort top-to-bottom with vertical tolerance for lines on the same row
        row_tol = max(4.0, body_size * 0.45)
        sorted_lines = sorted(raw_lines, key=lambda item: (round(item[1][1] / row_tol), item[1][0]))

        # Merge horizontal fragments on the same row and filter running headers/footers BEFORE heading detection!
        merged_rows: list[tuple[str, tuple[float, float, float, float]]] = []
        for text, box in sorted_lines:
            if merged_rows:
                prev_text, prev_box = merged_rows[-1]
                same_row = abs(box[1] - prev_box[1]) <= row_tol and abs(box[3] - prev_box[3]) <= row_tol * 1.2
                if same_row and box[0] >= prev_box[0]:
                    combined_box = (
                        min(prev_box[0], box[0]),
                        min(prev_box[1], box[1]),
                        max(prev_box[2], box[2]),
                        max(prev_box[3], box[3]),
                    )
                    merged_rows[-1] = (f"{prev_text} {text}", combined_box)
                    continue
            merged_rows.append((text, box))

        filtered_rows: list[tuple[str, tuple[float, float, float, float]]] = []
        for text, box in merged_rows:
            in_top_band = box[1] <= page_h * 0.11
            in_bottom_band = box[3] >= page_h * 0.89
            if in_top_band or in_bottom_band:
                if _is_standalone_page_number(text) or _norm_ocr_edge(text) in running_edges:
                    continue
            filtered_rows.append((text, box))

        page_groups: list[tuple[list[tuple[str, tuple[float, float, float, float]]], int | None]] = []
        current_group: list[tuple[str, tuple[float, float, float, float]]] = []

        for text, box in filtered_rows:
            line_h = box[3] - box[1]
            # Never promote lines inside the very top running-header strip unless clearly larger than body font
            if box[1] <= page_h * 0.085 and line_h < body_size * 1.15:
                level = None
            else:
                level = _heading_level_ocr(text, line_h, body_size)
            if level is not None:
                if current_group:
                    page_groups.append((current_group, None))
                    current_group = []
                page_groups.append(([(text, box)], level))
                continue

            if current_group:
                prev_text, prev_box = current_group[-1]
                vertical_gap = box[1] - prev_box[3]
                prev_ends_paragraph = prev_text.endswith(("。", "！", "？", ".", ";", "；")) and box[0] > prev_box[0] + body_size * 1.2
                if vertical_gap > max(18.0, body_size * 1.8) or prev_ends_paragraph or len(current_group) >= 14:
                    page_groups.append((current_group, None))
                    current_group = []
            current_group.append((text, box))

        if current_group:
            page_groups.append((current_group, None))

        for sequence, (group, level) in enumerate(page_groups, 1):
            block_id = f"{version_id}:pdf:{physical_page}:{sequence}"
            block_text = "\n".join(item[0] for item in group)
            locations = tuple(PdfSourceLocation(physical_page, label, item[1]) for item in group)
            if level is not None:
                while active_sections and active_sections[-1].level >= level:
                    active_sections.pop()
                parent = active_sections[-1].heading_block_id if active_sections else None
                section = Section(block_text, level, block_id, parent)
                sections.append(section)
                active_sections.append(section)
            blocks.append(ContentBlock(
                block_id,
                version_id,
                "heading" if level is not None else "paragraph",
                block_text,
                locations,
                active_sections[-1].heading_block_id if active_sections else None,
                tuple(section.title for section in active_sections),
            ))

    return sections, blocks, warnings
