# KnowledgeAgent

Tauri 自动启动、监督和关闭 Python FastAPI Sidecar。设计基线见[架构](架构.md)、[接口契约](接口.md)、[开发路线](开发路线与验收.md)和[RAG 架构方案](KnowledgeAgent_RAG_Architecture_v0.2.md)。当前进度与开发总结见[项目进度与缺陷分析及后续开发计划](项目进度与缺陷分析及后续开发计划.md)。

## 启动开发环境

在项目根目录安装锁定依赖，然后启动桌面应用。

```sh
pnpm --dir frontend install --frozen-lockfile
uv sync --project backend --python 3.12 --locked
(cd desktop && ../frontend/node_modules/.bin/tauri dev)
```

Tauri CLI 启动 Vite，Rust 宿主直接使用 `backend/.venv` 的 Python 解释器启动 Sidecar。窗口显示 `ready` 时，Tauri 已通过 stdin/stdout 握手和宿主认证就绪检查，React 已使用 `get_backend_connection` 返回的动态地址与 UI Token 读取后端状态。

关闭窗口会请求 Sidecar 优雅退出。若 Sidecar 崩溃，窗口会在下一次状态轮询时显示 `connection_error`。点击「重试」会通过原有连接 IPC 启动新实例并取得新的连接信息。若强制结束 Tauri，Sidecar 会检测宿主 stdin 管道的 EOF 并退出。

## 测试与构建

```sh
(cd backend && uv run --locked pytest && uv run --locked python -m compileall -q app tests)
pnpm --dir frontend test
pnpm --dir frontend build
(cd desktop && cargo fmt --check && cargo test --locked)
(cd desktop && cargo check --locked --release --features custom-protocol)
```

后端测试用真实子进程验证动态 socket 交接、双 Token、HTTP 停机和 stdin EOF。Rust 测试验证无效就绪信息、进程崩溃后的令牌轮换和宿主重新配置钩子的时序。

## 构建 macOS `.app` 与测试安装包

在 Apple Silicon 上安装前端锁定依赖，然后构建桌面应用包。PyInstaller 使用独立的构建虚拟环境，Python 运行时随应用打包。

```sh
pnpm --dir frontend install --frozen-lockfile
desktop/packaging/build-macos.sh onedir
desktop/packaging/build-macos.sh onefile
open -n desktop/target/packaging/apps/onedir/KnowledgeAgent.app
```

推荐用 `onedir` 继续开发与分发。

## 配置 DeepSeek 与 SiliconFlow Embedding

- **DeepSeek 对话模型**：在顶部密码输入框中提交 DeepSeek API Key。看到「已配置」表示凭据已进入当前 Sidecar。Rust 宿主将凭据保存在 macOS 钥匙串，应用或 Sidecar 重启时会重新注入 Sidecar 内存。模型默认使用 `deepseek-flash`，可通过宿主继承的 `KNOWLEDGEAGENT_DEEPSEEK_MODEL` 修改模型 ID。
- **SiliconFlow 向量模型**：输入 SiliconFlow API Key 并点击「保存 Key」，点击「测试连接」验证 `BAAI/bge-m3`（1024 维）。SiliconFlow 和 DeepSeek 使用独立的 macOS 钥匙串账户。

## 教材知识库（RAG）与引用溯源

1. **导入与索引**：在「教材知识库」标签页导入 PDF、DOCX 或 Markdown 教材，点击「详情与索引 → 开始建立索引」构建本地 FTS5 关键词索引与 `BAAI/bge-m3` 向量索引（未配置 SiliconFlow Key 时自动降级为本地关键词索引）。
2. **本地教材检索**：无需配置 DeepSeek，即可在「教材知识库」底部的检索面板按混合/关键词/向量模式及全局/跨教材对比策略搜索教材，查看精确到 PDF 物理页（及页标签）、Markdown 行号或 DOCX 段落/表格的已核验原文证据与数学公式字形提示。
3. **教材问答与引用卡片**：在「聊天」标签页勾选「结合教材知识库回答」，可限定教材范围并选择「全局相关性」或「跨教材对比」策略。系统自动处理多轮指代消解、注入公式核验护栏，并在回答下方持久化展示可展开核对原文的结构化引用卡片。
