import katex from 'katex';
import 'katex/dist/katex.min.css';
import type { ReactNode } from 'react';

const HAS_RICH_SYNTAX =
  /(?:^#{1,4}\s|^```|^>\s|^[-*]\s|^\d+\.\s|^(?:---+|\*\*\*+|___+)\s*$|^\|.+\|\s*$|\*\*[^*]+\*\*|`[^`]+`|\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+\$|\[引用\s*\d+\])/m;

function renderKatexHtml(latex: string, displayMode: boolean): string {
  const cleaned = latex.trim();
  if (!cleaned) return '';
  try {
    return katex.renderToString(cleaned, {
      displayMode,
      throwOnError: false,
      strict: 'ignore',
      output: 'html',
      trust: false,
    });
  } catch {
    return cleaned;
  }
}

function MathSpan({ latex, displayMode = false }: { latex: string; displayMode?: boolean }) {
  const html = renderKatexHtml(latex, displayMode);
  if (displayMode) {
    return <div className="math-block" dangerouslySetInnerHTML={{ __html: html }} />;
  }
  return <span className="math-inline" dangerouslySetInnerHTML={{ __html: html }} />;
}

function renderInline(text: string): ReactNode[] {
  // Match display math inline ($$...$$), \[...\], \(...\), inline $...$, **bold**, `code`, [引用 N]
  const tokenPattern =
    /(\$\$[^$\n]+\$\$|\\\[[^\]\n]+\\\]|\\\([^)\n]+\\\)|\$[^$\n]+\$|\*\*[^*]+\*\*|`[^`]+`|\[引用\s*\d+\])/g;
  const parts = text.split(tokenPattern);
  return parts.map((part, idx) => {
    if (!part) return null;
    if (part.startsWith('$$') && part.endsWith('$$') && part.length > 4) {
      return <MathSpan key={idx} latex={part.slice(2, -2)} displayMode />;
    }
    if (part.startsWith('\\[') && part.endsWith('\\]') && part.length > 4) {
      return <MathSpan key={idx} latex={part.slice(2, -2)} displayMode />;
    }
    if (part.startsWith('\\(') && part.endsWith('\\)') && part.length > 4) {
      return <MathSpan key={idx} latex={part.slice(2, -2)} />;
    }
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
      return <MathSpan key={idx} latex={part.slice(1, -1)} />;
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={idx}>{renderInline(part.slice(2, -2))}</strong>;
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code key={idx} className="inline-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (/^\[引用\s*\d+\]$/.test(part)) {
      return (
        <span key={idx} className="citation-badge">
          {part}
        </span>
      );
    }
    return part;
  });
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function isTableSeparator(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes('|')) return false;
  const cells = splitTableRow(trimmed);
  return cells.length > 0 && cells.every((cell) => /^:?-{2,}:?$/.test(cell));
}

export default function MarkdownAnswer({
  content,
  ariaLabel,
  messageId,
}: {
  content: string;
  ariaLabel: string;
  messageId?: string;
}) {
  if (!HAS_RICH_SYNTAX.test(content)) {
    return (
      <p className="answer" aria-label={ariaLabel} data-message-id={messageId}>
        {content}
      </p>
    );
  }

  const lines = content.split('\n');
  const blocks: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Fenced code block
    if (line.trimStart().startsWith('```')) {
      const lang = line.trimStart().slice(3).trim();
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      blocks.push(
        <pre key={`code-${i}`} className="code-block" data-lang={lang || undefined}>
          <code>{codeLines.join('\n')}</code>
        </pre>,
      );
      continue;
    }

    // Display math block: $$...$$ or \[...\]
    if (trimmed.startsWith('$$') || trimmed.startsWith('\\[')) {
      const openDelim = trimmed.startsWith('$$') ? '$$' : '\\[';
      const closeDelim = openDelim === '$$' ? '$$' : '\\]';
      if (trimmed.endsWith(closeDelim) && trimmed.length > openDelim.length + closeDelim.length) {
        const latex = trimmed.slice(openDelim.length, -closeDelim.length).trim();
        blocks.push(<MathSpan key={`math-${i}`} latex={latex} displayMode />);
        i += 1;
        continue;
      }
      const mathLines: string[] = [trimmed.slice(openDelim.length)];
      i += 1;
      while (i < lines.length && !lines[i].trim().endsWith(closeDelim)) {
        mathLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) {
        mathLines.push(lines[i].trim().slice(0, -closeDelim.length));
        i += 1;
      }
      blocks.push(<MathSpan key={`math-${i}`} latex={mathLines.join('\n').trim()} displayMode />);
      continue;
    }

    // Horizontal rule
    if (/^(?:---+|\*\*\*+|___+)$/.test(trimmed)) {
      blocks.push(<hr key={`hr-${i}`} className="md-hr" />);
      i += 1;
      continue;
    }

    // Markdown table
    if (trimmed.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = splitTableRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().includes('|') && lines[i].trim().length > 0) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      blocks.push(
        <div key={`tbl-${i}`} className="md-table-wrap">
          <table className="md-table">
            <thead>
              <tr>
                {headers.map((h, hIdx) => (
                  <th key={hIdx}>{renderInline(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rIdx) => (
                <tr key={rIdx}>
                  {headers.map((_, cIdx) => (
                    <td key={cIdx}>{renderInline(row[cIdx] ?? '')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      const Tag = `h${Math.min(6, level + 2)}` as 'h3' | 'h4' | 'h5' | 'h6';
      blocks.push(
        <Tag key={`h-${i}`} className="md-heading">
          {renderInline(text)}
        </Tag>,
      );
      i += 1;
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(<li key={`ul-li-${i}`}>{renderInline(lines[i].replace(/^\s*[-*]\s+/, ''))}</li>);
        i += 1;
      }
      blocks.push(
        <ul key={`ul-${i}`} className="md-list">
          {items}
        </ul>,
      );
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(<li key={`ol-li-${i}`}>{renderInline(lines[i].replace(/^\s*\d+\.\s+/, ''))}</li>);
        i += 1;
      }
      blocks.push(
        <ol key={`ol-${i}`} className="md-list">
          {items}
        </ol>,
      );
      continue;
    }

    // Blockquote
    if (line.trimStart().startsWith('> ')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trimStart().startsWith('> ')) {
        quoteLines.push(lines[i].trimStart().slice(2));
        i += 1;
      }
      blocks.push(
        <blockquote key={`q-${i}`} className="md-quote">
          {renderInline(quoteLines.join('\n'))}
        </blockquote>,
      );
      continue;
    }

    if (!trimmed) {
      i += 1;
      continue;
    }

    blocks.push(
      <p key={`p-${i}`} className="md-paragraph">
        {renderInline(line)}
      </p>,
    );
    i += 1;
  }

  return (
    <div className="answer markdown-body" aria-label={ariaLabel} data-message-id={messageId}>
      {blocks}
    </div>
  );
}
