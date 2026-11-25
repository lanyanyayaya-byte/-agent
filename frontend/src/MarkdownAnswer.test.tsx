// @vitest-environment jsdom

import { cleanup, render } from '@testing-library/react';
import { afterEach, expect, test } from 'vitest';
import MarkdownAnswer from './MarkdownAnswer';

afterEach(() => cleanup());

test('renders inline and display LaTeX with KaTeX, horizontal rules, and tables', () => {
  const md = [
    '### 二、$\\chi^2$ 检验法的构造思路',
    '',
    '检验假设为：',
    '$$H_0: F(x)=F_0(x) \\qquad \\text{vs} \\qquad H_1: F(x)\\neq F_0(x)$$',
    '',
    '分组区间为 $(-\\infty, t_1]$，频数为 \\(\\nu_i\\)，参见 [引用 3]。',
    '',
    '---',
    '',
    '| 符号 | 含义 |',
    '| --- | --- |',
    '| $\\nu_i$ | 第 $i$ 组频数 |',
  ].join('\n');

  const { container } = render(<MarkdownAnswer content={md} ariaLabel="已保存回答" messageId="m1" />);
  // KaTeX elements rendered (not raw \chi^2 or \qquad source code)
  expect(container.querySelectorAll('.katex').length).toBeGreaterThanOrEqual(5);
  expect(container.querySelector('.math-block .katex-display')).not.toBeNull();
  expect(container.textContent).not.toContain('\\qquad');
  expect(container.textContent).not.toContain('\\chi^2');
  expect(container.textContent).not.toContain('\\nu_i');
  expect(container.querySelector('.md-hr')).not.toBeNull();
  expect(container.querySelector('.md-table')).not.toBeNull();
  expect(container.querySelector('.citation-badge')?.textContent).toBe('[引用 3]');
});
