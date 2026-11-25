import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  mcpErrorMessage,
  runtimePrepare,
  skillsDelete,
  skillsInstallFromPreview,
  skillsPickAndPreviewLocal,
  skillsPrepare,
  skillsPreviewGithub,
  skillsSetEnabled,
  skillsUpdate,
  type ApiClient,
  type RuntimeStatus,
  type SkillDetailView,
  type SkillImportPreview,
  type SkillView,
} from './api/apiClient';

function sourceLabel(sourceType: SkillView['source_type']): string {
  if (sourceType === 'builtin') return '内置技能';
  if (sourceType === 'github') return 'GitHub 技能';
  return '本地技能';
}

export default function SkillsPanel({
  client,
  active,
  onSkillsChange,
}: {
  client?: ApiClient;
  active: boolean;
  onSkillsChange?: (skills: SkillView[]) => void;
}) {
  const [skills, setSkills] = useState<SkillView[]>([]);
  const [runtime, setRuntime] = useState<RuntimeStatus>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  // GitHub & Local import state
  const [githubUrl, setGithubUrl] = useState('');
  const [showAdvancedSource, setShowAdvancedSource] = useState(false);
  const [githubRef, setGithubRef] = useState('');
  const [githubSubpath, setGithubSubpath] = useState('');
  const [previewBusy, setPreviewBusy] = useState(false);
  const [preview, setPreview] = useState<SkillImportPreview | null>(null);
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [installBusy, setInstallBusy] = useState(false);

  // Per-skill actions & detail viewer
  const [busySkillId, setBusySkillId] = useState<string>();
  const [runtimePreparing, setRuntimePreparing] = useState(false);
  const [expandedId, setExpandedId] = useState<string>();
  const [detailsById, setDetailsById] = useState<Record<string, SkillDetailView>>({});
  const [detailLoadingId, setDetailLoadingId] = useState<string>();

  const refresh = useCallback(async (signal?: AbortSignal) => {
    if (!client) return;
    const effective = signal ?? new AbortController().signal;
    setLoading(true);
    try {
      const [list, rt] = await Promise.all([
        client.listSkills(effective),
        client.getRuntimeStatus(effective),
      ]);
      if (!effective.aborted) {
        setSkills(list);
        setRuntime(rt);
        onSkillsChange?.(list);
        setError('');
      }
    } catch (err) {
      if (!effective.aborted) {
        setError(mcpErrorMessage(err, '无法加载 Skills 列表或运行环境状态。'));
      }
    } finally {
      if (!effective.aborted) setLoading(false);
    }
  }, [client, onSkillsChange]);

  useEffect(() => {
    if (!client || !active) return;
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [client, active, refresh]);

  async function handlePreviewGithub(event: FormEvent) {
    event.preventDefault();
    const url = githubUrl.trim();
    if (!url || previewBusy) return;
    setPreviewBusy(true);
    setError('');
    setNotice('正在从 GitHub 获取技能目录...');
    try {
      const result = await skillsPreviewGithub(url, githubRef || undefined, githubSubpath || undefined);
      setPreview(result);
      setSelectedPaths(result.candidates.length === 1 ? [result.candidates[0].relative_path] : []);
      setNotice(`发现 ${result.candidates.length} 个可安装技能，请勾选后点击安装。`);
    } catch (err) {
      setNotice('');
      setError(mcpErrorMessage(err, '获取 GitHub 技能列表失败，请确认仓库公开且包含 SKILL.md。'));
    } finally {
      setPreviewBusy(false);
    }
  }

  async function handlePickLocal(mode: 'directory' | 'zip') {
    if (previewBusy) return;
    setPreviewBusy(true);
    setError('');
    setNotice('');
    try {
      const result = await skillsPickAndPreviewLocal(mode);
      if (!result) return;
      setPreview(result);
      setSelectedPaths(result.candidates.map((c) => c.relative_path));
      setNotice(`从本地${mode === 'zip' ? '压缩包' : '文件夹'}发现 ${result.candidates.length} 个技能。`);
    } catch (err) {
      setError(mcpErrorMessage(err, '读取本地技能来源失败。'));
    } finally {
      setPreviewBusy(false);
    }
  }

  async function handleInstallSelected() {
    if (!preview || selectedPaths.length === 0 || installBusy) return;
    setInstallBusy(true);
    setError('');
    setNotice(`正在安装选中的 ${selectedPaths.length} 个技能...`);
    try {
      const res = await skillsInstallFromPreview(preview.preview_id, selectedPaths, true);
      setPreview(null);
      setSelectedPaths([]);
      setGithubUrl('');
      setGithubRef('');
      setGithubSubpath('');
      setNotice(`已成功安装并启用 ${res.skills.length} 个技能：${res.skills.map((s) => s.name).join('、')}`);
      await refresh();
    } catch (err) {
      setNotice('');
      setError(mcpErrorMessage(err, '安装技能失败。'));
    } finally {
      setInstallBusy(false);
    }
  }

  async function handleToggleEnabled(skill: SkillView) {
    if (busySkillId) return;
    setBusySkillId(skill.skill_id);
    setError('');
    try {
      const updated = await skillsSetEnabled(skill.skill_id, !skill.enabled);
      setSkills((prev) => {
        const next = prev.map((item) => (item.skill_id === updated.skill_id ? updated : item));
        onSkillsChange?.(next);
        return next;
      });
    } catch (err) {
      setError(mcpErrorMessage(err, '切换技能启用状态失败。'));
    } finally {
      setBusySkillId(undefined);
    }
  }

  async function handlePrepareSkill(skill: SkillView) {
    if (busySkillId) return;
    setBusySkillId(skill.skill_id);
    setError('');
    setNotice(`正在为「${skill.name}」准备独立运行环境依赖，请稍候...`);
    try {
      const res = await skillsPrepare(skill.skill_id);
      setNotice(res.ok ? `「${skill.name}」依赖已就绪。` : `依赖准备未完全就绪：${res.dependency_status.summary}`);
      await refresh();
    } catch (err) {
      setNotice('');
      setError(mcpErrorMessage(err, '准备技能依赖失败。'));
    } finally {
      setBusySkillId(undefined);
    }
  }

  async function handlePrepareBaseRuntime() {
    if (runtimePreparing) return;
    setRuntimePreparing(true);
    setError('');
    setNotice('正在准备独立 Python 环境（python-docx）与 Mermaid CLI，请稍候...');
    try {
      const res = await runtimePrepare({
        python_packages: ['python-docx'],
        install_mermaid_cli: true,
        ensure_python_venv: true,
      });
      setRuntime(res.status);
      setNotice(res.ok ? '基础运行环境已全部就绪。' : '部分依赖未能自动安装，请查看运行环境状态。');
      await refresh();
    } catch (err) {
      setNotice('');
      setError(mcpErrorMessage(err, '准备基础运行环境失败。'));
    } finally {
      setRuntimePreparing(false);
    }
  }

  async function handleUpdateSkill(skill: SkillView) {
    if (busySkillId) return;
    setBusySkillId(skill.skill_id);
    setError('');
    setNotice(`正在从来源更新「${skill.name}」...`);
    try {
      const updated = await skillsUpdate(skill.skill_id);
      setNotice(`「${updated.name}」已更新至 ${updated.revision}。`);
      setDetailsById((prev) => {
        const next = { ...prev };
        delete next[skill.skill_id];
        return next;
      });
      await refresh();
    } catch (err) {
      setNotice('');
      setError(mcpErrorMessage(err, '更新技能失败。'));
    } finally {
      setBusySkillId(undefined);
    }
  }

  async function handleDeleteSkill(skill: SkillView) {
    if (busySkillId) return;
    setBusySkillId(skill.skill_id);
    setError('');
    try {
      await skillsDelete(skill.skill_id);
      if (expandedId === skill.skill_id) setExpandedId(undefined);
      setNotice(`已删除技能「${skill.name}」（历史产物已保留）。`);
      await refresh();
    } catch (err) {
      setError(mcpErrorMessage(err, '删除技能失败。'));
    } finally {
      setBusySkillId(undefined);
    }
  }

  async function handleToggleDetail(skillId: string) {
    if (expandedId === skillId) {
      setExpandedId(undefined);
      return;
    }
    setExpandedId(skillId);
    if (!client || detailsById[skillId]) return;
    setDetailLoadingId(skillId);
    try {
      const detail = await client.getSkillDetail(skillId, new AbortController().signal);
      setDetailsById((prev) => ({ ...prev, [skillId]: detail }));
    } catch (err) {
      setError(mcpErrorMessage(err, '无法加载技能详情。'));
    } finally {
      setDetailLoadingId(undefined);
    }
  }

  function toggleCandidate(path: string) {
    setSelectedPaths((prev) => (prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path]));
  }

  const enabledCount = skills.filter((s) => s.enabled).length;

  return (
    <section className="skills-panel" aria-label="Agent Skills 管理">
      <header className="skills-header">
        <div>
          <h2>Agent Skills</h2>
          <p className="skills-subtitle">
            管理内置与第三方技能（标准 SKILL.md），在独立 Runtime 中处理文档、运行脚本并交付真实产物。
          </p>
        </div>
        <div className="skills-header-stats">
          <span className="skills-stat-pill">已安装 {skills.length}</span>
          <span className="skills-stat-pill active">已启用 {enabledCount}</span>
          <button type="button" className="secondary-btn" disabled={loading} onClick={() => void refresh()}>
            {loading ? '刷新中…' : '刷新'}
          </button>
        </div>
      </header>

      {error && <p role="alert" className="skills-banner error">{error}</p>}
      {notice && <p role="status" className="skills-banner info">{notice}</p>}

      {/* Runtime readiness bar */}
      <section className="skills-card runtime-card" aria-label="独立运行环境状态">
        <div className="skills-card-head">
          <div>
            <h3>本地独立运行环境（Runtime）</h3>
            <p className="skills-muted">
              脚本在应用数据目录的独立环境中运行（继承当前本机用户权限，不影响后端服务依赖）。
            </p>
          </div>
          <button
            type="button"
            className="primary-btn"
            disabled={runtimePreparing || !client}
            onClick={() => void handlePrepareBaseRuntime()}>
            {runtimePreparing ? '正在准备环境…' : '准备基础依赖'}
          </button>
        </div>
        {runtime && (
          <div className="runtime-badges">
            <span className={`rt-badge ${runtime.venv_ready ? 'ready' : 'missing'}`}>
              Python venv：{runtime.venv_ready ? '已就绪' : '未初始化'}
            </span>
            <span className={`rt-badge ${runtime.uv.available ? 'ready' : 'missing'}`}>
              uv：{runtime.uv.available ? (runtime.uv.version ?? '可用') : '未检测到'}
            </span>
            <span className={`rt-badge ${runtime.node.available ? 'ready' : 'missing'}`}>
              Node.js：{runtime.node.available ? (runtime.node.version ?? '可用') : '未检测到'}
            </span>
            <span className={`rt-badge ${runtime.mermaid_cli.available ? 'ready' : 'missing'}`}>
              Mermaid CLI：{runtime.mermaid_cli.available ? '已安装' : '未安装'}
            </span>
            <span className={`rt-badge ${runtime.chrome.available ? 'ready' : 'missing'}`}>
              Chrome：{runtime.chrome.available ? '已复用本机 Chrome' : '未检测到'}
            </span>
            <span className={`rt-badge ${runtime.git.available ? 'ready' : 'missing'}`}>
              Git：{runtime.git.available ? '可用' : '将使用 ZIP 归档'}
            </span>
          </div>
        )}
      </section>

      {/* Import from GitHub / Local Folder / ZIP */}
      <section className="skills-card import-card" aria-label="导入技能">
        <div className="skills-card-head">
          <div>
            <h3>导入第三方技能</h3>
            <p className="skills-muted">
              支持公开 GitHub 仓库或子目录链接（如 <code>https://github.com/anthropics/skills</code>），以及本地文件夹或 ZIP 压缩包。
            </p>
          </div>
          <div className="import-local-actions">
            <button type="button" className="secondary-btn" disabled={previewBusy} onClick={() => void handlePickLocal('directory')}>
              📁 导入本地文件夹
            </button>
            <button type="button" className="secondary-btn" disabled={previewBusy} onClick={() => void handlePickLocal('zip')}>
              🗜️ 导入 ZIP
            </button>
          </div>
        </div>

        <form className="github-import-form" onSubmit={(e) => void handlePreviewGithub(e)}>
          <input
            type="url"
            className="github-url-input"
            placeholder="粘贴公开 GitHub 仓库或技能目录 URL（例如 https://github.com/anthropics/skills）"
            value={githubUrl}
            onChange={(e) => setGithubUrl(e.target.value)}
            disabled={previewBusy}
          />
          <button
            type="button"
            className="text-toggle-btn"
            onClick={() => setShowAdvancedSource((prev) => !prev)}>
            {showAdvancedSource ? '收起分支/路径' : '指定分支/子目录'}
          </button>
          <button type="submit" className="primary-btn" disabled={previewBusy || !githubUrl.trim()}>
            {previewBusy ? '获取中…' : '获取技能列表'}
          </button>
        </form>

        {showAdvancedSource && (
          <div className="github-advanced-row">
            <input
              type="text"
              placeholder="可选分支 / Tag / Commit（如 main）"
              value={githubRef}
              onChange={(e) => setGithubRef(e.target.value)}
              disabled={previewBusy}
            />
            <input
              type="text"
              placeholder="可选子目录路径（如 skills/xlsx）"
              value={githubSubpath}
              onChange={(e) => setGithubSubpath(e.target.value)}
              disabled={previewBusy}
            />
          </div>
        )}

        {preview && (
          <div className="import-preview-box">
            <div className="import-preview-header">
              <div>
                <strong>发现 {preview.candidates.length} 个可安装技能</strong>
                <span className="skills-muted">
                  {' · '}来源：{preview.source_url}
                  {preview.commit_sha ? ` @ ${preview.commit_sha.slice(0, 8)}` : ''}
                </span>
              </div>
              <div className="import-preview-actions">
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => setSelectedPaths(
                    selectedPaths.length === preview.candidates.length
                      ? []
                      : preview.candidates.map((c) => c.relative_path),
                  )}>
                  {selectedPaths.length === preview.candidates.length ? '取消全选' : '全选'}
                </button>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => { setPreview(null); setSelectedPaths([]); }}>
                  取消
                </button>
                <button
                  type="button"
                  className="primary-btn"
                  disabled={installBusy || selectedPaths.length === 0}
                  onClick={() => void handleInstallSelected()}>
                  {installBusy ? '安装中…' : `安装并启用（${selectedPaths.length}）`}
                </button>
              </div>
            </div>
            <ul className="import-candidate-list">
              {preview.candidates.map((cand) => {
                const checked = selectedPaths.includes(cand.relative_path);
                return (
                  <li key={cand.relative_path} className={`import-candidate-item ${checked ? 'selected' : ''}`}>
                    <label className="candidate-label">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleCandidate(cand.relative_path)}
                      />
                      <div className="candidate-body">
                        <div className="candidate-title-row">
                          <strong>{cand.name}</strong>
                          <code>{cand.relative_path}</code>
                          <span className={`dep-pill ${cand.dependency_status.ready ? 'ready' : 'pending'}`}>
                            {cand.dependency_status.ready ? '依赖就绪' : '需准备依赖'}
                          </span>
                          {cand.existing_skill_id && <span className="dep-pill warn">将覆盖同名技能</span>}
                        </div>
                        <p className="candidate-desc">{cand.description}</p>
                        {!cand.dependency_status.ready && (
                          <p className="candidate-note">{cand.dependency_status.summary}</p>
                        )}
                        {cand.compatibility_notes.map((note, idx) => (
                          <p key={idx} className="candidate-note">{note}</p>
                        ))}
                      </div>
                    </label>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </section>

      {/* Installed skills list */}
      <section className="skills-list-section" aria-label="已安装技能列表">
        {skills.length === 0 && !loading && (
          <p className="skills-empty">尚未安装任何技能。</p>
        )}
        <div className="skills-grid">
          {skills.map((skill) => {
            const isBusy = busySkillId === skill.skill_id;
            const isExpanded = expandedId === skill.skill_id;
            const detail = detailsById[skill.skill_id];
            return (
              <article key={skill.skill_id} className={`skill-item-card ${skill.enabled ? 'is-enabled' : ''}`}>
                <div className="skill-item-top">
                  <div className="skill-item-info">
                    <div className="skill-title-row">
                      <h3>{skill.name}</h3>
                      <span className={`skill-source-badge src-${skill.source_type}`}>
                        {sourceLabel(skill.source_type)}
                      </span>
                      <span className={`dep-pill ${skill.ready ? 'ready' : 'pending'}`}>
                        {skill.ready ? '依赖就绪' : '缺少依赖'}
                      </span>
                      <code className="skill-rev">{skill.revision}</code>
                    </div>
                    <p className="skill-desc">{skill.description}</p>
                    {skill.source_url && (
                      <p className="skill-source-url">来源：{skill.source_url}</p>
                    )}
                    {!skill.ready && (
                      <p className="skill-dep-hint">{skill.dependency_status.summary}</p>
                    )}
                  </div>

                  <div className="skill-item-controls">
                    <label className="skill-switch" title={skill.enabled ? '停用技能' : '启用技能'}>
                      <input
                        type="checkbox"
                        checked={skill.enabled}
                        disabled={isBusy}
                        onChange={() => void handleToggleEnabled(skill)}
                      />
                      <span>{skill.enabled ? '已启用' : '已停用'}</span>
                    </label>
                    {!skill.ready && skill.dependency_status.can_auto_prepare && (
                      <button
                        type="button"
                        className="primary-btn small"
                        disabled={isBusy}
                        onClick={() => void handlePrepareSkill(skill)}>
                        {isBusy ? '准备中…' : '安装依赖'}
                      </button>
                    )}
                    <button
                      type="button"
                      className="secondary-btn small"
                      onClick={() => void handleToggleDetail(skill.skill_id)}>
                      {isExpanded ? '收起详情' : '查看详情'}
                    </button>
                    {skill.source_type !== 'builtin' && (
                      <>
                        <button
                          type="button"
                          className="secondary-btn small"
                          disabled={isBusy}
                          onClick={() => void handleUpdateSkill(skill)}>
                          更新
                        </button>
                        <button
                          type="button"
                          className="danger-btn small"
                          disabled={isBusy}
                          onClick={() => void handleDeleteSkill(skill)}>
                          删除
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <div className="skill-detail-drawer">
                    {detailLoadingId === skill.skill_id && <p className="skills-muted">正在加载 SKILL.md...</p>}
                    {detail && (
                      <>
                        {detail.compatibility && (
                          <p className="skill-meta-line"><strong>环境说明：</strong>{detail.compatibility}</p>
                        )}
                        {detail.license && (
                          <p className="skill-meta-line"><strong>许可证：</strong>{detail.license}</p>
                        )}
                        {detail.warnings.length > 0 && (
                          <div className="skill-warnings">
                            {detail.warnings.map((w, idx) => <p key={idx}>⚠️ {w}</p>)}
                          </div>
                        )}
                        {detail.resources.length > 0 && (
                          <div className="skill-resources">
                            <strong>附带资源（{detail.resources.length}）：</strong>
                            <div className="skill-resource-pills">
                              {detail.resources.map((r) => (
                                <code key={r.relative_path}>{r.relative_path}</code>
                              ))}
                            </div>
                          </div>
                        )}
                        <div className="skill-instructions-wrap">
                          <strong>SKILL.md 指令内容：</strong>
                          <pre className="skill-instructions-pre">{detail.instructions}</pre>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </section>
    </section>
  );
}
