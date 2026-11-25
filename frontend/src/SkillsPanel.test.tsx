// @vitest-environment jsdom

import { invoke } from '@tauri-apps/api/core';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import { connectApiClient, type RuntimeStatus, type SkillImportPreview, type SkillView } from './api/apiClient';
import SkillsPanel from './SkillsPanel';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(), isTauri: vi.fn(() => true) }));

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

const mockRuntime: RuntimeStatus = {
  runtime_dir: '/tmp/runtime',
  venv_path: '/tmp/runtime/venv',
  venv_ready: true,
  venv_python: '/tmp/runtime/venv/bin/python',
  installed_python_packages: { 'python-docx': '1.2.0' },
  puppeteer_config_path: '/tmp/runtime/cache/puppeteer-config.json',
  uv: { available: true, path: '/opt/homebrew/bin/uv', version: 'uv 0.13.7', source: 'system' },
  python: { available: true, path: '/tmp/runtime/venv/bin/python', version: 'Python 3.12.16', source: 'venv' },
  node: { available: true, path: '/opt/homebrew/bin/node', version: 'v22.23.4', source: 'system' },
  npm: { available: true, path: '/opt/homebrew/bin/npm', version: '10.9.0', source: 'system' },
  npx: { available: true, path: '/opt/homebrew/bin/npx', version: '10.9.0', source: 'system' },
  git: { available: true, path: '/usr/bin/git', version: 'git 2.54.0', source: 'system' },
  chrome: { available: true, path: '/Applications/Google Chrome.app', version: null, source: 'system' },
  mermaid_cli: { available: true, path: '/tmp/runtime/npm-tools/node_modules/.bin/mmdc', version: '11.4.2', source: 'runtime' },
  shell: { available: true, path: '/bin/zsh', version: null, source: 'system' },
  warnings: [],
};

const builtinSkill: SkillView = {
  skill_id: 'mind-map',
  name: 'mind-map',
  description: '生成高清中文思维导图并输出 .mmd / .svg / .png 产物。',
  source_type: 'builtin',
  source_url: null,
  revision: 'builtin',
  enabled: true,
  ready: true,
  dependency_status: {
    ready: true,
    missing_python_packages: [],
    missing_node_tools: [],
    missing_system_commands: [],
    can_auto_prepare: false,
    summary: '所有依赖已就绪',
  },
  warnings: [],
  created_at: '2026-11-25T08:00:00Z',
  updated_at: '2026-11-25T08:00:00Z',
};

test('lists installed skills, previews GitHub repo, selectively installs, and toggles skill', async () => {
  const skills: SkillView[] = [builtinSkill];
  vi.mocked(invoke).mockImplementation(async (cmd, args: any) => {
    if (cmd === 'get_backend_connection') {
      return { base_url: 'http://127.0.0.1:8766', ui_token: 'ui-test' };
    }
    if (cmd === 'skills_preview_github') {
      expect(args.request.url).toBe('https://github.com/anthropics/skills');
      const preview: SkillImportPreview = {
        preview_id: 'prev-1',
        source_kind: 'github',
        source_url: 'https://github.com/anthropics/skills',
        resolved_ref: 'main',
        commit_sha: 'abcdef1234567890',
        subpath: null,
        candidates: [
          {
            relative_path: 'skills/xlsx',
            name: 'xlsx',
            description: 'Create and edit Excel spreadsheets with openpyxl.',
            license: 'Apache-2.0',
            compatibility: 'Python 3',
            existing_skill_id: null,
            dependency_status: {
              ready: false,
              missing_python_packages: ['openpyxl'],
              missing_node_tools: [],
              missing_system_commands: [],
              can_auto_prepare: true,
              summary: '缺少 Python 依赖：openpyxl',
            },
            required_python_packages: ['openpyxl'],
            required_node_tools: [],
            required_system_commands: [],
            warnings: [],
            compatibility_notes: [],
            resource_count: 2,
          },
          {
            relative_path: 'skills/pdf',
            name: 'pdf',
            description: 'PDF processing toolkit.',
            license: 'Apache-2.0',
            compatibility: null,
            existing_skill_id: null,
            dependency_status: {
              ready: true,
              missing_python_packages: [],
              missing_node_tools: [],
              missing_system_commands: [],
              can_auto_prepare: false,
              summary: '所有依赖已就绪',
            },
            required_python_packages: [],
            required_node_tools: [],
            required_system_commands: [],
            warnings: [],
            compatibility_notes: [],
            resource_count: 1,
          },
        ],
      };
      return preview;
    }
    if (cmd === 'skills_install_from_preview') {
      expect(args.request.selected_relative_paths).toEqual(['skills/xlsx']);
      const installed: SkillView = {
        ...builtinSkill,
        skill_id: 'xlsx',
        name: 'xlsx',
        description: 'Create and edit Excel spreadsheets with openpyxl.',
        source_type: 'github',
        source_url: 'https://github.com/anthropics/skills',
        revision: 'abcdef12',
        ready: false,
      };
      skills.push(installed);
      return { skills: [installed] };
    }
    if (cmd === 'skills_set_enabled') {
      const target = skills.find((s) => s.skill_id === args.request.skill_id)!;
      target.enabled = args.request.enabled;
      return { ...target };
    }
    throw new Error(`Unexpected IPC ${String(cmd)}`);
  });

  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url).pathname.replace('/api/v1', '');
    if (path === '/skills') return Response.json(skills);
    if (path === '/runtime/status') return Response.json(mockRuntime);
    throw new Error(`Unexpected fetch ${path}`);
  }));

  const client = await connectApiClient();
  const onChange = vi.fn();
  render(<SkillsPanel client={client} active onSkillsChange={onChange} />);

  expect(await screen.findByText('mind-map')).toBeTruthy();
  expect(screen.getByText(/Python venv：已就绪/)).toBeTruthy();

  // Preview GitHub repo
  const urlInput = screen.getByPlaceholderText(/粘贴公开 GitHub 仓库/);
  fireEvent.change(urlInput, { target: { value: 'https://github.com/anthropics/skills' } });
  fireEvent.click(screen.getByRole('button', { name: '获取技能列表' }));

  expect(await screen.findByText('发现 2 个可安装技能')).toBeTruthy();
  expect(screen.getByText('skills/xlsx')).toBeTruthy();
  expect(screen.getByText('skills/pdf')).toBeTruthy();

  // Select only xlsx and install
  const checkboxes = screen.getAllByRole<HTMLInputElement>('checkbox');
  const xlsxCheckbox = checkboxes.find((cb) => cb.closest('li')?.textContent?.includes('skills/xlsx'))!;
  fireEvent.click(xlsxCheckbox);
  fireEvent.click(screen.getByRole('button', { name: '安装并启用（1）' }));

  await waitFor(() => expect(screen.getByText('xlsx')).toBeTruthy());
  expect(screen.getByText(/已成功安装并启用 1 个技能：xlsx/)).toBeTruthy();
});
