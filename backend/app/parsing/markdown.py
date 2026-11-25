from markdown_it import MarkdownIt

from app.parsing.models import (
    BlockKind, ContentBlock, DocumentParseError, MarkdownSourceLocation, ParsedDocument, Section,
)

BLOCK_KINDS: dict[str, BlockKind] = {
    "heading_open": "heading",
    "paragraph_open": "paragraph",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "fence": "code",
    "code_block": "code",
    "table_open": "table",
    "blockquote_open": "blockquote",
    "hr": "thematic_break",
    "html_block": "html",
}


def parse_markdown(text: str, *, document_id: str, document_version_id: str) -> ParsedDocument:
    """Extract top-level CommonMark blocks with pipe tables and original line ranges.

    Paragraphs and headings retain inline Markdown. Lists, tables, and quotes retain
    their source syntax and nested content. Code keeps the parser's literal body.
    """
    # Match CommonMark's line boundaries. splitlines() also splits on Unicode
    # separators which CommonMark treats as characters within a physical line.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    parser = MarkdownIt("commonmark", {"maxNesting": 64}).enable("table").disable("inline")
    try:
        tokens = parser.parse(text)
    except Exception:
        raise DocumentParseError("MARKDOWN_PARSE_FAILED", "Unable to parse Markdown syntax.") from None

    literal_ends = {token.map[1] for token in tokens if token.type in ("fence", "html_block") and token.map}
    blocks: list[ContentBlock] = []
    sections: list[Section] = []
    active_sections: list[Section] = []
    for index, token in enumerate(tokens):
        # At its nesting limit the tokenizer can consume the remainder of the
        # surrounding block without parsing it, including later headings.
        if token.type in ("list_item_open", "blockquote_open") and token.level + 1 >= parser.options.maxNesting:
            raise DocumentParseError("MARKDOWN_NESTING_LIMIT", "Markdown lists or quotes are nested too deeply.")
        kind = BLOCK_KINDS.get(token.type)
        if token.level != 0 or kind is None:
            continue
        if token.map is None or not 0 <= token.map[0] < token.map[1] <= len(lines):
            raise DocumentParseError("MARKDOWN_SOURCE_MAP_INVALID", "A block has no reliable source line range.")
        start, end = token.map
        if kind in ("list", "blockquote"):
            # The parser can include a separating blank line in a container's map.
            while end > start + 1 and end not in literal_ends and not lines[end - 1].strip(" \t"):
                end -= 1

        start_line, end_line = start + 1, end
        block_id = f"{document_version_id}:md:{start_line}:{end_line}"
        heading_level = None
        code_language = None
        if kind in ("heading", "paragraph"):
            block_text = tokens[index + 1].content
        elif kind == "code":
            block_text = token.content
            info = token.info.split(maxsplit=1)
            code_language = info[0] if info else None
        else:
            block_text = "\n".join(lines[start:end])

        if kind == "heading":
            heading_level = int(token.tag[1:])
            while active_sections and active_sections[-1].level >= heading_level:
                active_sections.pop()
            parent_id = active_sections[-1].heading_block_id if active_sections else None
            section = Section(block_text, heading_level, block_id, parent_id)
            sections.append(section)
            active_sections.append(section)

        blocks.append(ContentBlock(
            block_id=block_id,
            document_version_id=document_version_id,
            kind=kind,
            text=block_text,
            source_locations=(MarkdownSourceLocation(start_line, end_line, heading_level),),
            section_heading_block_id=active_sections[-1].heading_block_id if active_sections else None,
            section_path=tuple(section.title for section in active_sections),
            code_language=code_language,
        ))

    return ParsedDocument(document_id, document_version_id, "md", tuple(sections), tuple(blocks))
