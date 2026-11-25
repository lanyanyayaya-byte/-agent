---
name: word-formatting
description: 根据用户要求对 DOCX Word 文档进行真实格式排版调整（标题样式、中西文字体、字号、行距、段间距、页边距、首行缩进、对齐方式及基础表格格式），完整保留原有正文并生成新的可编辑 DOCX 产物，同时执行回读验证。
license: MIT
compatibility: Requires Python with python-docx in the local KnowledgeAgent Runtime.
metadata:
  author: KnowledgeAgent
  version: "1.0"
  runtime:
    python_packages:
      - python-docx
---

# Word Formatting（DOCX 文档格式调整与校验）

## 适用场景
- 用户上传/导入了 `.docx` 附件，或需要继续调整上一轮已生成的 `.docx` 产物。
- 根据用户提出的具体排版要求调整标题、正文、字体（中文 `w:eastAsia` + 西文 `w:ascii`/`w:hAnsi`）、字号、行距、段前段后间距、首行缩进、对齐方式、页边距和基础表格样式。

## 重要原则
- **严格以用户实际要求为准**，不要对用户未要求修改的样式强行套用固定模板。
- **保留原始文件不变**，始终在 `work/` 操作副本，将排版结果输出到 `output/<原名>_排版后.docx`（或用户指定文件名）。
- **不擅自改写正文措辞**。
- 如果检查发现原文档包含复杂数学公式（OMML）、域代码、批注/修订或复杂浮动图形，必须在回复中如实说明兼容范围，不伪造无损编辑承诺。

## 工作流

1. **定位并复制待处理文档**
   - 用 `list_workspace` 查看当前会话可用的附件（`inputs/...`）或历史产物（`artifacts/...`）。
   - 用 `copy_file` 将目标 `.docx` 复制到 `work/source.docx`。

2. **检查文档结构与限制提示**
   - 运行检查脚本了解段落、标题层级、表格、图片及特殊元素：
     ```text
     command: "python"
     args: ["<skill_dir>/scripts/inspect_docx.py", "work/source.docx"]
     ```

3. **按用户要求编写格式规格并直接调用内置排版脚本（强烈优先复用！）**
   - 无需重复读取 `scripts/format_docx.py` 或重写自定义排版脚本！直接使用 `<skill_dir>/scripts/format_docx.py` 即可处理绝大多数排版需求（包括所有段落样式都是 `Normal` 的真实中文文档）。
   - 若 `inspect_docx.py` 显示文档未使用 Word 内置 `Heading 1 / Heading 2` 样式（例如全部为 `Normal`），只需在 `work/format_spec.json` 中通过段落索引列表指定角色即可：
     - `heading1_indices`: 一级标题段落索引列表（如 `[0]`）
     - `heading2_indices`: 二级标题段落索引列表（如 `[11, 12, 15]`）
     - `heading3_indices`: 三级标题段落索引列表（可选）
     - `no_indent_indices`: 需要应用正文样式但不应首行缩进的段落索引（如签字栏、日期行、列表项等）
     - `skip_indices`: 需要保持原样不改动的段落索引（可选）
   - 常用中文字号参考：三号=16pt、四号=14pt、小四=12pt、五号=10.5pt（也可直接写 `"三号"` / `"四号"` / `"小四"`，脚本自动换算，一般无需再读取 `references/chinese-typography.md`）。
   - 用 `write_file` 将 JSON 格式配置写入 `work/format_spec.json`，然后直接运行：
     ```text
     command: "python"
     args: [
       "<skill_dir>/scripts/format_docx.py",
       "--input", "work/source.docx",
       "--output", "output/<输出文件名>.docx",
       "--spec", "work/format_spec.json"
     ]
     ```
   - `format_spec.json` 支持的样式字段示例（均为可选）：
     - `heading1` / `heading2` / `heading3` / `body`：
       - `east_asia_font`: 如 `"宋体"`、`"黑体"`、`"楷体"`、`"微软雅黑"`
       - `western_font`: 如 `"Times New Roman"`、`"Arial"`
       - `size`: 如 `"三号"`、`"小四"`、`"四号"`、`"五号"` 或数字磅值（如 `12`、`16`）
       - `bold`: `true` / `false`
       - `italic`: `true` / `false`
       - `line_spacing`: 倍数（如 `1.5`）
       - `space_before_pt` / `space_after_pt`: 段前/段后磅值
       - `first_line_indent_chars`: 首行缩进字符数（如 `2`，按该段字号自动换算磅值；设 `0` 可清除缩进）
       - `first_line_indent_pt`: 首行缩进磅值
       - `alignment`: `"left"` / `"center"` / `"right"` / `"justify"`
     - `margins_cm`: `{"top": 2.54, "bottom": 2.54, "left": 3.18, "right": 3.18}`
     - `table`: `{"east_asia_font": "宋体", "western_font": "Times New Roman", "size": "五号", "header_bold": true, "line_spacing": 1.15, "alignment": "left"}`
   - 仅当内置 `format_docx.py` 确实无法满足特殊需求（如复杂自定义页眉页脚、特定文字着色）时，才在 `work/` 编写简短补充脚本。

4. **回读验证格式与正文完整性**
   - 直接运行内置校验脚本（它同样支持 `heading1_indices` / `heading2_indices` / `skip_indices`）：
     ```text
     command: "python"
     args: [
       "<skill_dir>/scripts/validate_docx.py",
       "--original", "work/source.docx",
       "--formatted", "output/<输出文件名>.docx",
       "--spec", "work/format_spec.json"
     ]
     ```
   - 若校验失败或脚本报错，根据 `stderr` 修正 `format_spec.json` 后重新运行。

5. **发布产物并汇报**
   - 校验通过后立即调用 `publish_artifact` 注册 `output/<输出文件名>.docx`。
   - 正常排版任务应在 **6～8 步工具调用**内干净完成（`load_skill` → `copy_file` → `inspect_docx.py` → `write_file(format_spec.json)` → `format_docx.py` → `validate_docx.py` → `publish_artifact`）。
