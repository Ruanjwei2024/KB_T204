"""
账枢 LedgerMind — 国际化 (i18n)
中英文文案字典 + 语言切换工具
"""

TRANSLATIONS = {
    "zh": {
        # --- 应用信息 ---
        "app_title": "📒 账枢 LedgerMind — 托管资产会计 AI 智库",
        "app_page_title": "账枢 LedgerMind",
        "app_caption": "LLM: **{llm}** · Embed: **{embed}** · 向量库: **{vs}** · 数据源: FASB ASC / SEC / Fed / AICPA",
        "app_subtitle": "托管资产会计 AI 智库",
        "demo_data_source": "数据源: FASB ASC / SEC / Fed / AICPA",

        # --- 侧边栏 ---
        "lang_label": "🌐 语言 / Language",
        "menu_settings": "⚙️ 设置",
        "menu_settings_en": "⚙️ Settings",
        "menu_lang_section": "🌐 界面语言 / UI Language",
        "menu_api_section": "🔌 API 连通性 / API Connectivity",
        "sidebar_apis": "⚙️ API 连通性",
        "btn_test_embed": "🔌 测试 Embedding API",
        "btn_test_llm": "🤖 测试 LLM",
        "loading_model": "正在加载模型(首次约 30-120 秒,会从 HuggingFace 下载 ~409MB)...",
        "calling_llm": "正在调用 LLM(首次可能需要加载模型)...",
        "embed_api_ok": "✅ Embedding API 正常",
        "llm_api_ok": "✅ LLM 正常",
        "embed_api_fail": "❌ Embedding API 失败",
        "llm_api_fail": "❌ LLM 失败",
        "hf_blocked_warning": "💡 国内访问 huggingface.co 不稳定,请设环境变量后重启:\n  `$env:HF_ENDPOINT='https://hf-mirror.com'`\n  `streamlit run rag_skeleton.py`",

        "sidebar_docs": "📄 上传文档",
        "upload_label": "选择文档 (PDF / HTML / XBRL / MD / TXT)",
        "upload_hint": "支持 PDF、HTML、XBRL、Markdown、TXT",
        "btn_build_kb": "🔨 构建知识库",
        "processing_file": "处理 {name}...",
        "index_success": "✅ 已索引 {n} 份 / {total} 个片段",
        "index_failed": "❌ 索引失败",
        "btn_clear_kb": "🗑️ 清空知识库",
        "clear_help": "清空 FAISS 知识库",
        "clear_success": "知识库已清空,刷新页面生效",
        "kb_size": "📊 知识库规模: **{n}** 个片段",
        "kb_empty": "⚠️ 知识库为空,请先上传文档",

        # --- Tab ---
        "qa_header": "🎯 智能问答",
        "tab_qa": "💬 普通问答",
        "tab_compare": "🔄 多准则对比",
        "tab_graph": "🔗 知识图谱",

        # --- Tab1: QA ---
        "qa_input_label": "问题输入",
        "qa_placeholder": "如:ASC 860 下客户资产是否进资产负债表?",
        "qa_answer_header": "### 回答",
        "qa_retrieving": "检索 + 生成中...",
        "qa_sources": "📚 {n} 条引用片段",
        "qa_source_relevance": "相关度 {score:.2f}",
        "qa_error": "❌ 出错了",
        "qa_empty_answer": "⚠️ LLM 返回了空内容。可能原因:\n"
                           "- 本地模型 `qwen2.5:14b` 太大,在 CPU 上不稳定\n"
                           "- Prompt 太长被截断\n"
                           "- 超时但已经返回部分\n\n"
                           "**建议**:\n"
                           "1. 切到 `qwen2.5:7b` 或 `qwen2.5:7b-instruct`\n"
                           "2. 查终端日志,看原始 LLM 响应\n"
                           "3. 换 LLM provider(MiniMax 或 Bedrock)",

        # --- Tab2: Compare ---
        "compare_intro": "> **多准则对比**:自动从多个角度检索资料,生成结构化的多准则对比表(US GAAP / IFRS / PRC GAAP 等)。\n"
                          "> 这是通用 RAG 框架没有的差异化功能。",
        "compare_intro_en": "> **Multi-Standard Comparison**: automatically retrieves from multiple angles and generates a structured comparison table\n"
                            "> (US GAAP / IFRS / PRC GAAP, etc.) — a differentiated feature generic RAG frameworks lack.",
        "compare_topic_label": "对比主题",
        "compare_topic_placeholder": "如:客户资金隔离 / 收入确认 / 衍生品会计处理",
        "compare_standards_label": "对比准则",
        "btn_generate": "🔄 生成对比",
        "compare_result_header": "### 📊 对比结果",
        "compare_spinner": "多次检索 + 生成对比中(检索 4 次 + 1 次 LLM 调用)...",
        "compare_sources": "📚 参考了 {n} 条资料片段",
        "compare_empty_topic": "⚠️ 请输入对比主题",
        "compare_empty_standards": "⚠️ 请至少选择一个准则",
        "compare_empty_answer": "⚠️ LLM 返回了空内容。\n\n"
                               "**最可能的原因**:本地模型 `qwen2.5:14b` 在 CPU 上返回空。\n\n"
                               "**最快的修法**:\n"
                               "1. 切到 7B 模型:`ollama pull qwen2.5:7b-instruct`\n"
                               "2. 或切到 MiniMax API(改 `.env` 的 `LLM_PROVIDER=MiniMax`)\n"
                               "3. 查终端日志里的 `[DEBUG]` 行",

        # --- Tab3: Graph ---
        "graph_intro": "> **知识图谱视图**:把文档、准则、术语的关系可视化。\n"
                       "> - 🟦 蓝色方块 = 文档\n"
                       "> - 🟧 橙色椭圆 = 准则/术语\n"
                       "> - 灰色细线 = 文档提到的准则\n"
                       "> - 绿色粗线 = 文档间相似度高(共同关键词 ≥3)\n"
                       "> - **可拖动节点、缩放、悬停查看详情**",
        "graph_intro_en": "> **Knowledge Graph View**: visualize relationships between documents, standards and terms.\n"
                          "> - 🟦 Blue squares = documents\n"
                          "> - 🟧 Orange ellipses = standards/terms\n"
                          "> - Thin grey lines = standards mentioned by the document\n"
                          "> - Thick green lines = similar documents (≥3 shared keywords)\n"
                          "> - **Drag nodes, zoom, hover for details**",
        "btn_generate_graph": "🔗 生成知识图谱",
        "graph_building": "构建知识图谱中...",
        "graph_size": "📊 图谱规模: {n} 个节点, {e} 条边",

        # --- 通用 ---
        "common_ok": "✅",
        "common_error": "❌",
        "common_warning": "⚠️",
    },

    "en": {
        # --- App info ---
        "app_title": "📒 LedgerMind — Custody Asset Accounting AI Knowledge Hub",
        "app_page_title": "LedgerMind",
        "app_caption": "LLM: **{llm}** · Embed: **{embed}** · Vector Store: **{vs}** · Sources: FASB ASC / SEC / Fed / AICPA",
        "app_subtitle": "Custody Asset Accounting AI Knowledge Hub",
        "demo_data_source": "Sources: FASB ASC / SEC / Fed / AICPA",

        # --- Sidebar ---
        "lang_label": "🌐 Language / 语言",
        "menu_settings": "⚙️ Settings",
        "menu_settings_en": "⚙️ 设置",
        "menu_lang_section": "🌐 UI Language / 界面语言",
        "menu_api_section": "🔌 API Connectivity / API 连通性",
        "sidebar_apis": "⚙️ API Connectivity",
        "btn_test_embed": "🔌 Test Embedding API",
        "btn_test_llm": "🤖 Test LLM",
        "loading_model": "Loading model (first run ~30-120s, downloads ~409MB from HuggingFace)...",
        "calling_llm": "Calling LLM (first run may need to load the model)...",
        "embed_api_ok": "✅ Embedding API OK",
        "llm_api_ok": "✅ LLM OK",
        "embed_api_fail": "❌ Embedding API failed",
        "llm_api_fail": "❌ LLM failed",
        "hf_blocked_warning": "💡 huggingface.co is unstable in CN, set env var and restart:\n  `$env:HF_ENDPOINT='https://hf-mirror.com'`\n  `streamlit run rag_skeleton.py`",

        "sidebar_docs": "📄 Upload Documents",
        "upload_label": "Choose documents (PDF / HTML / XBRL / MD / TXT)",
        "upload_hint": "Supports PDF, HTML, XBRL, Markdown, TXT",
        "btn_build_kb": "🔨 Build Knowledge Base",
        "processing_file": "Processing {name}...",
        "index_success": "✅ Indexed {n} documents / {total} chunks",
        "index_failed": "❌ Indexing failed",
        "btn_clear_kb": "🗑️ Clear Knowledge Base",
        "clear_help": "Clear FAISS knowledge base",
        "clear_success": "Knowledge base cleared. Refresh to take effect.",
        "kb_size": "📊 Knowledge base size: **{n}** chunks",
        "kb_empty": "⚠️ Knowledge base is empty. Please upload documents first.",

        # --- Tabs ---
        "qa_header": "🎯 Intelligent Q&A",
        "tab_qa": "💬 Standard Q&A",
        "tab_compare": "🔄 Multi-Standard Compare",
        "tab_graph": "🔗 Knowledge Graph",

        # --- Tab1: QA ---
        "qa_input_label": "Your question",
        "qa_placeholder": "e.g. Under ASC 860, are client assets recorded on the balance sheet?",
        "qa_answer_header": "### Answer",
        "qa_retrieving": "Retrieving + generating...",
        "qa_sources": "📚 {n} cited chunks",
        "qa_source_relevance": "Relevance {score:.2f}",
        "qa_error": "❌ Error occurred",
        "qa_empty_answer": "⚠️ LLM returned empty content. Possible causes:\n"
                           "- Local model `qwen2.5:14b` is too large and unstable on CPU\n"
                           "- Prompt was truncated\n"
                           "- Timed out after partial generation\n\n"
                           "**Suggestions**:\n"
                           "1. Switch to `qwen2.5:7b` or `qwen2.5:7b-instruct`\n"
                           "2. Check terminal logs for the raw LLM response\n"
                           "3. Change LLM provider (MiniMax or Bedrock)",

        # --- Tab2: Compare ---
        "compare_intro": "> **Multi-Standard Comparison**: automatically retrieves from multiple angles and generates a structured comparison table\n"
                          "> (US GAAP / IFRS / PRC GAAP, etc.) — a differentiated feature generic RAG frameworks lack.",
        "compare_intro_en": "> **Multi-Standard Comparison**: automatically retrieves from multiple angles and generates a structured comparison table\n"
                            "> (US GAAP / IFRS / PRC GAAP, etc.) — a differentiated feature generic RAG frameworks lack.",
        "compare_topic_label": "Comparison topic",
        "compare_topic_placeholder": "e.g. Client fund segregation / Revenue recognition / Derivative accounting",
        "compare_standards_label": "Standards",
        "btn_generate": "🔄 Generate Comparison",
        "compare_result_header": "### 📊 Comparison Result",
        "compare_spinner": "Multi-query retrieval + comparison generation (4 searches + 1 LLM call)...",
        "compare_sources": "📚 Referenced {n} chunks",
        "compare_empty_topic": "⚠️ Please enter a comparison topic",
        "compare_empty_standards": "⚠️ Please select at least one standard",
        "compare_empty_answer": "⚠️ LLM returned empty content.\n\n"
                               "**Most likely cause**: local model `qwen2.5:14b` returns empty on CPU.\n\n"
                               "**Fastest fix**:\n"
                               "1. Switch to a 7B model: `ollama pull qwen2.5:7b-instruct`\n"
                               "2. Or switch to the MiniMax API (change `LLM_PROVIDER=MiniMax` in `.env`)\n"
                               "3. Check the terminal log for the `[DEBUG]` line",

        # --- Tab3: Graph ---
        "graph_intro": "> **Knowledge Graph View**: visualize relationships between documents, standards and terms.\n"
                       "> - 🟦 Blue squares = documents\n"
                       "> - 🟧 Orange ellipses = standards/terms\n"
                       "> - Thin grey lines = standards mentioned by the document\n"
                       "> - Thick green lines = similar documents (≥3 shared keywords)\n"
                       "> - **Drag nodes, zoom, hover for details**",
        "graph_intro_en": "> **Knowledge Graph View**: visualize relationships between documents, standards and terms.\n"
                          "> - 🟦 Blue squares = documents\n"
                          "> - 🟧 Orange ellipses = standards/terms\n"
                          "> - Thin grey lines = standards mentioned by the document\n"
                          "> - Thick green lines = similar documents (≥3 shared keywords)\n"
                          "> - **Drag nodes, zoom, hover for details**",
        "btn_generate_graph": "🔗 Generate Knowledge Graph",
        "graph_building": "Building knowledge graph...",
        "graph_size": "📊 Graph size: {n} nodes, {e} edges",

        # --- Common ---
        "common_ok": "✅",
        "common_error": "❌",
        "common_warning": "⚠️",
    },
}


def get_lang() -> str:
    return LANG  # 由主程序设置


def t(key: str, **kwargs) -> str:
    """按当前语言取值,缺 key 时回退到 key 本身"""
    v = TRANSLATIONS.get(LANG, TRANSLATIONS["zh"]).get(key)
    if v is None:
        return key
    return v.format(**kwargs) if kwargs else v


# 当前语言(主程序会在 UI 初始化前覆盖)
LANG = "zh"