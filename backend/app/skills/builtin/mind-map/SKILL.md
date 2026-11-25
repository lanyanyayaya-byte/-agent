---
name: mind-map
description: 根据用户提供的文本、知识点、Markdown 文档、教材检索结果或历史对话提炼层级结构，生成可编辑的 Mermaid 思维导图源文件（.mmd）并渲染为真实可打开的中文 SVG 矢量图和 PNG 高清图，支持基于历史 .mmd 产物继续修改。
license: MIT
compatibility: Requires Node.js and @mermaid-js/mermaid-cli (mmdc) in the local KnowledgeAgent Runtime.
metadata:
  author: KnowledgeAgent
  version: "1.0"
  runtime:
    node_tools:
      - mermaid-cli
    system_commands:
      - node
---

# Mind Map（中文思维导图生成与修改）

## 适用场景
- 将一段文本、某个技术/学术主题、Markdown 笔记、教材 RAG 证据或对话内容提炼为结构清晰的中文思维导图。
- 在同一会话的后续轮次中，基于上一轮已发布的 `.mmd` 产物增删或调整节点并生成新版本。

## 工作流

1. **整理输入与层级结构**
   - 如果用户提供了工作区附件或要求修改上一轮思维导图，先用 `list_workspace` 和 `read_file` 读取原文件或历史 `.mmd` 产物内容。
   - 提炼 1 个中心主题（`root((中心主题))`）、3–6 个核心一级分支，以及清晰的子节点（建议 3–4 层，单节点文字精炼，避免超长句子导致节点重叠或裁切）。

2. **编写 Mermaid 思维导图源文件**
   - 使用 `write_file` 将 Mermaid 代码写入 `output/<文件名>.mmd`（例如 `output/mindmap.mmd`）。
   - 语法注意事项：
     - 第一行必须是 `mindmap`。
     - 严格使用 2 空格缩进表达父子层级，不要混用 Tab。
     - 节点文字中避免裸露圆括号 `()`、方括号 `[]`、花括号 `{}` 或冒号引起的解析冲突；如需标点优先使用中文全角标点或简短词语。

3. **渲染为真实 SVG 与 PNG**
   - 使用 `execute_command` 调用本技能目录中的渲染脚本（`skill_dir` 由 `load_skill` 返回）：
     ```text
     command: "python"
     args: [
       "<skill_dir>/scripts/render_mindmap.py",
       "--input", "output/<文件名>.mmd",
       "--output-svg", "output/<文件名>.svg",
       "--output-png", "output/<文件名>.png",
       "--config", "<skill_dir>/references/mermaid-config.json"
     ]
     ```
   - 如果脚本返回非零退出码（例如 Mermaid 语法错误），阅读 `stderr` 中的报错行号，修改 `.mmd` 后重新执行渲染，直到成功。

4. **发布产物并汇报**
   - 渲染并校验通过后，依次调用 `publish_artifact` 发布以下文件：
     - `output/<文件名>.mmd`（可编辑源文件）
     - `output/<文件名>.svg`（矢量图）
     - `output/<文件名>.png`（高清预览图）
   - 仅在 `publish_artifact` 成功后向用户说明已生成的思维导图结构与产物列表。
