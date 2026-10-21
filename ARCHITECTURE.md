# 账枢 LedgerMind — 技术架构与实现说明

> **读者**：接手维护本项目的技术人员。
> **用途**：理解模块边界、关键不变量、数据流与已知限制，用于排障和二次开发。
> **代码基线**：2026-10-21，`rag_skeleton.py` 1140 行，`i18n.py` 232 行。
> 本文中出现的函数名与常量名均可在代码中直接搜索定位；行号仅作参考，会随改动漂移。

---

## 1. 系统定位与能力边界

账枢 LedgerMind 是一个面向**托管资产会计**的垂直 RAG 知识库。它把公开监管文本
（SEC / FASB / IRS / Federal Reserve / Basel / PCAOB / AICPA）变成可对话、可引用、可比对的知识服务。

三条硬边界，任何改动都不得突破：

| 边界 | 说明 |
|---|---|
| **零内部数据** | 只接受公开权威资料，不接入公司内部文件、系统、客户数据 |
| **可追溯** | 每条回答都必须带 `[来源N]` 标注，答案与原文片段一一对应 |
| **不绑死单一云** | LLM / Embedding / 向量库三层各自可换 provider，本地与 AWS 都可跑 |

**明确不在范围内**：多轮对话记忆、页码级引用、扫描件 OCR、检索指标评估（Recall@5 / MRR）。

---

## 2. 技术栈

| 层 | 技术 | 版本约束来源 |
|---|---|---|
| 前端 | Streamlit | `>=1.32.0`，用到 `st.popover` / `st.query_params` / `use_container_width` |
| 后端 | Python | 3.11 / 3.12 |
| PDF 解析 | pypdf | `>=4.0.0` |
| HTML / XBRL 解析 | BeautifulSoup | `>=4.12.0`，用 `html.parser` 与 `xml` 两个解析器 |
| 向量库（本地） | faiss-cpu | `>=1.7.4,<1.9` |
| 向量库（AWS） | opensearch-py | 可选，比赛环境 |
| 本地 Embedding | sentence-transformers | `>=6.0`，**不要降级到 5.x**（见 §11） |
| 知识图谱 | pyvis | `>=0.3.0` |
| 配置 | python-dotenv | `>=1.0.0` |

### 2.1 numpy 版本地狱（最容易踩的坑）

`numpy` / `scipy` / `scikit-learn` / `faiss-cpu` / `sentence-transformers` 必须锁在同一套 ABI 上：

```
numpy 1.26.4  +  scipy 1.12.0  +  scikit-learn 1.3.2
             +  faiss-cpu 1.8.0.post1  +  sentence-transformers 6.1.0
             +  torch 2.14.1+cpu  +  torchvision 0.29.1+cpu   ← 已验证可跑通
```

- `numpy` 必须 `<2.0`：faiss-cpu 1.8 尚未支持 numpy 2.x 的 ABI，会报
  `numpy.core.multiarray failed to import`。
- `scipy` 必须 `<1.13`：1.13+ 依赖 numpy 2.x。
- PyTorch 必须装 CPU 版，否则会拉 CUDA 依赖：
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

---

## 3. 运行架构

```
┌───────────────────────────────────────────────────────────────┐
│  Streamlit UI                                                  │
│  ┌──────────────┐  ┌─────────────────────────────────────┐   │
│  │ 侧边栏        │  │ 右上角 ⚙ 设置                        │   │
│  │ 文档上传      │  │  · 语言切换 (zh / en)                │   │
│  │ 构建知识库    │  │  · Embedding API 连通性测试          │   │
│  │ 清空知识库    │  │  · LLM API 连通性测试                │   │
│  └──────┬───────┘  └─────────────────────────────────────┘   │
│         │                                                      │
│  ┌──────▼────────────────────────────────────────────────┐    │
│  │ Tab1 智能问答 │ Tab2 多准则对比 │ Tab3 知识图谱         │    │
│  └──────┬────────────────┬───────────────────┬──────────┘    │
└─────────┼────────────────┼───────────────────┼───────────────┘
          │                │                   │
   ┌──────▼──────┐  ┌──────▼──────┐   ┌────────▼────────┐
   │ 问答        │  │ 多准则对比   │   │ 知识图谱         │
   │ search(k=5)│  │ 4 路检索     │   │ 关键词匹配建图   │
   │ ask()      │  │ + 表格 prompt│  │ (仅 FAISS 模式)  │
   └──────┬──────┘  └──────┬──────┘   └────────┬────────┘
          │                │                   │
   ┌──────▼────────────────▼───────────────────▼────────┐
   │  向量层   search(query, k) / index_docs(chunks, src)│
   │  ┌────────────────┐      ┌────────────────────┐   │
   │  │ LocalFAISS      │      │ OpenSearch (AWS)   │   │
   │  │ IndexFlatIP     │      │ knn_vector + HNSW  │   │
   │  │ L2 归一化 → 余弦 │      │ BM25 + KNN 混合    │   │
   │  └────────────────┘      └────────────────────┘   │
   └──────────────────────┬────────────────────────────┘
                          │
   ┌──────────────────────▼────────────────────────────┐
   │  检索层   search(query, k=5)   type_="query"        │
   └──────────────────────┬────────────────────────────┘
                          │
   ┌──────────────────────▼────────────────────────────┐
   │  向量化层 embed(texts, type_)                      │
   │  local(sentence-transformers) │ MiniMax │ Bedrock  │
   └──────────────────────┬────────────────────────────┘
                          │
   ┌──────────────────────▼────────────────────────────┐
   │  生成层   ask(question, ctxs) / compare_standards()  │
   │  local_qwen(Ollama) │ MiniMax │ Bedrock Claude      │
   └────────────────────────────────────────────────────┘
```

### 3.1 一条问答请求的完整数据流

1. 用户在 Tab1 输入问题 → `search(q, k=5)`
2. `embed([q], type_="query")` 得到查询向量
   - `type_="query"` 与入库时的 `"db"` 区分，**只有 MiniMax 真正使用该参数**
3. 向量库返回 top-5 片段，每项是 `{"text", "source", "score"}`
4. `ask(q, ctxs)` 调 LLM，prompt 由 `_build_user_msg()` 拼装：
   - 片段格式化为 `[来源N: 文件名]\n片段内容`，用 `\n\n---\n\n` 连接
   - 强制四条约束：只用资料、关键信息标注来源、未覆盖要明说、跟随界面语言回答
5. 回答下方用 `st.expander` 展示引用片段（截断到 400 字符）

---

## 4. 模块详解

代码按分区注释组织，顺序即依赖顺序。

### 4.1 配置层（文件顶部）

所有配置走环境变量，`.env` 加载一次。三个 provider 选择器是**互不依赖**的：

```
LLM_PROVIDER    = bedrock | MiniMax | local_qwen
EMBED_PROVIDER  = bedrock | MiniMax | local
VECTOR_STORE    = faiss | opensearch
```

关键常量：

| 常量 | 默认 | 说明 |
|---|---|---|
| `INDEX_NAME` | `rag-custody` | OpenSearch 索引名，硬编码 |
| `SCRIPT_DIR` | `Path(__file__).resolve().parent` | **不要改成相对路径**，Windows 下会触发 `PermissionError` |
| `EMBED_DIM` | `768` | 必须与 embedding 模型实际输出维度一致 |
| `LLM_MODEL_BEDROCK` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | 硬编码，不走环境变量 |
| `EMBED_MODEL_BEDROCK` | `amazon.titan-embed-text-v2:0` | 硬编码 |
| `OS_HOST` / `OS_PASS` | 占位串 | **代码里有硬编码默认密码，接真实环境前必须用环境变量覆盖** |

启动时会做一次 `vector_store/` 目录的写入探测，失败则 `st.stop()` 并提示权限问题。
这是为了提前暴露 Windows 目录权限问题，而不是等到索引写入时才失败。

`boto3` 与 `opensearchpy` 都是**条件导入**：只有对应 provider 被选中时才 import，
未选中的路径不产生依赖。

### 4.2 文档解析 `load_pdf(file)`

函数名叫 `load_pdf`，实际是**五格式统一入口**，不要被名字误导。

格式判定分两级：

1. **文件对象**：读前 512 字节 magic number
   - `%PDF` → PDF
   - `<?xml` 或前 200 字节含 `<xbrl` → XML/XBRL
   - `<!DOCTYPE` / `<html` / `<HTML` → HTML
2. **文件名后缀**：`.xml/.xbrl`、`.html/.htm`、`.md/.markdown/.mdown/.mkd/.txt/.text`
3. **都不匹配**：兜底当 PDF 试

各分支处理：

- **Markdown / TXT**：按 `utf-8 → utf-8-sig → gbk → latin-1` 顺序尝试解码，全部失败则
  `errors="ignore"`。随后剥离：代码块围栏、图片 `![]()`、链接保留文字、
  `**粗体**`/`__粗体__`/`*斜体*`、表格分隔行 `|---|`。
- **PDF**：`PdfReader(file).pages` 逐页 `extract_text()`，以 `\n\n` 连接。
  **只支持文本型 PDF，扫描件会得到空字符串。**
- **XBRL/XML**：`BeautifulSoup(content, "xml")` 取文本，再丢弃长度 ≤ 10 的行
  （XBRL 里大量纯数字元数据会污染检索）。
- **HTML**：`BeautifulSoup(content, "html.parser")`，先 `decompose()` 掉
  `script / style / noscript / header / footer / nav`，再取文本。

**统一的失败出口**：解析后 `len(text) < 50` 抛 `ValueError`，并给出三种可能原因
（扫描件 / XBRL 只有数字元数据 / 文件加密）。这个阈值是经验值，调低会让噪声入库。

### 4.3 切块 `chunk(text, size=600, overlap=80)`

```python
range(0, len(text), size - overlap)   # 步长 520
text[i:i+size]                        # 窗口 600
```

即**固定 600 字符窗口、80 字符重叠**，纯字符切分，**不感知句子和段落边界**。

维护须知：
- 改 `size` / `overlap` 会改变已有索引的语义，**必须重建整个索引**，否则新旧块无法对齐。
- 块可能从句子中间截断，所以引用片段读起来可能不完整。这是当前已知限制。
- 只过滤 `strip()` 后为空的块，没有最小长度过滤。

### 4.4 向量化层

三层结构：统一入口 `embed()` → 三个 provider 实现 → 格式适配器。

```
embed(texts, type_="db")
  ├─ EMBED_PROVIDER == "bedrock"  → _embed_bedrock(texts)
  ├─ EMBED_PROVIDER == "local"    → _embed_local(texts)
  └─ else（MiniMax）              → _embed_MiniMax(texts, type_=type_)
```

**`_embed_bedrock`**：逐条调用（**没有批量接口**），每条截断到 8000 字符。

**`_embed_local`**：`SentenceTransformer.encode(normalize_embeddings=True)`，
返回值已 L2 归一化，numpy 转 list。模型**懒加载**并缓存在模块级 `_local_model`，
只有 `EMBED_PROVIDER=local` 时才下载。首次约 30–120 秒 / 409MB。

**`_embed_MiniMax`**：协议与 OpenAI 不同，**必须用 `texts` 而不是 `input`**：

```json
POST https://api.MiniMax.chat/v1/embeddings
{ "model": "embo-01", "texts": ["..."], "type": "db" }
```

响应要检查 `base_resp.status_code != 0` 抛业务错误——该 API 会在 HTTP 200 下返回
`{"status_code": 1008, "insufficient balance"}` 这类错误。

**`_parse_embedding_response(resp)`** 是统一格式适配器，识别三种形状：

| 形状 | 来源 |
|---|---|
| `{"vectors": [[...], ...]}` | MiniMax 批量 |
| `{"data": [{"embedding": [...]}]}` | OpenAI 标准 |
| `{"embedding": [...]}` | 单条直返 |

`_embed_MiniMax` 末尾还有一个兜底：若首个元素不是 list（单条返回），包成 `list[list]`。

**维度契约（重要）**：`embed()` 发现实际维度与 `EMBED_DIM` 不一致时**只警告不报错**，
并提示同步改 `.env`；随后 `LocalFAISS.add()` 会采用实际维度重建索引。
这是刻意的容错——但**跨会话会踩坑**：换模型后旧索引维度对不上，`faiss.read_index`
加载失败会退化成"重建索引"，数据静默丢失。**换 embedding 模型一定要先点「清空知识库」。**

### 4.5 向量存储层

#### LocalFAISS（本地）

| 方法 | 行为 |
|---|---|
| `__init__(dim)` | 尝试 `_load()`；失败则 `st.warning` 并留空，不抛异常 |
| `_load()` | 读 `faiss.index` + `metadata.pkl`；两者必须同时存在 |
| `_save()` | 仅当 `index is not None and index.ntotal > 0` 时才落盘 |
| `reset()` | 置空索引并**删除磁盘上的两个文件** |
| `add(chunks, source)` | 校验空列表 → embedding → 维度一致性 → float32 堆叠 → L2 归一化 → 写入 |
| `search(query_vec, k=5)` | `k = min(k, ntotal)`，返回 `{"text","source","score"}` |
| `count()` | 索引为空返回 0 |

**索引类型 `faiss.IndexFlatIP`（内积）**，配合写入前后的 `faiss.normalize_L2`，
等价于**精确余弦相似度**。没有近似检索，10 万块以内性能可接受。

`add()` 里有一道容易被忽略的校验：`dims = {len(v) for v in vecs_raw}`，
若集合大小 > 1 说明同批向量维度不一致，直接抛错——防止污染索引。

`metadata.pkl` 与 `faiss.index` **必须成对存在**，元数据靠 list 下标与向量 ID 对齐。
单独删掉任何一个会导致下次加载失败并重建。

#### OpenSearch（比赛环境）

索引 `rag-custody`，mapping：

- `text` → `text`（BM25 可检索）
- `embedding` → `knn_vector`，HNSW，`space_type: cosinesimil`
- `source` → `keyword`

检索用 `bool.should` 混合打分：BM25 `boost: 0.4` + KNN kNN。

**局限**：`get_vs()` 在 OpenSearch 模式下**返回 `None`**，因此
① 知识图谱 Tab 完全不可用；② 侧边栏知识库规模永远显示 0；
③ OpenSearch 模式没有「清空知识库」按钮（按钮条件是 `VECTOR_STORE == "faiss"`）。

### 4.6 生成层

```
ask(question, ctxs)
  ├─ bedrock      → _ask_bedrock      max_tokens=1024, temperature=0.2, timeout 默认
  ├─ local_qwen   → _ask_local_qwen   max_tokens=512,  temperature=0.2, timeout=300
  └─ else         → _ask_MiniMax      max_tokens=1024, temperature=0.2, timeout=60
```

注意三家的 `max_tokens` 不一致：本地 14B 模型刻意压到 512 以提速，
Bedrock / MiniMax 给到 1024。

**`_ask_local_qwen` 的超时处理**：`requests.exceptions.ReadTimeout` 被捕获后
转成带诊断信息的 `TimeoutError`，提示四条排查方向（检查 `ollama ps` 是否走 GPU /
重试一次 / 换 7B / 减小检索 k）。timeout 从 120s 提到 300s 就是因为 14B 首次
加载要 30–60 秒。

**`_extract_llm_text(data, source)`** 统一 LLM 响应格式：

1. `{"error": {...}}` → 拼成 `❌ {source} API 错误: ...`
2. `{"choices":[{"message":{"content":...}}]}` → OpenAI 兼容（MiniMax / Ollama 的 `/v1` 端点）
3. `{"choices":[{"text":...}]}` → 旧式 completions
4. `{"response": "..."}` → Ollama 原生 `/api/chat`
5. 都不匹配 → 打印完整响应到终端，页面返回可读错误

content 为空时会额外 `print` 原始响应到终端——**调试空回答时先看 Streamlit 终端**。

#### Prompt 构造与语言联动

```
_UI_LANG / SYSTEM_PROMPT / COMPARISON_PROMPT
  ↑ 在 UI 段每次 rerun 时按 st.session_state["lang"] 重新赋值
```

`_build_user_msg()` 读 `_lang_code()`（即 `globals()["_UI_LANG"]`）决定中英文，
并输出**语言跟随界面**的约束。

**维护红线**：不要在模块顶层写 `global _UI_LANG` 或 `global SYSTEM_PROMPT`。
Streamlit 每次交互都会重跑整个脚本，模块级赋值本身就是全局写入，
加 `global` 语句会触发 `SyntaxError: assigned before global declaration`。

### 4.7 多准则对比

差异化功能，入口 `compare_standards(topic, standards)`。

```
multi_query_search(topic, k_per_query=4)
  ├─ "{topic} US GAAP ASC"
  ├─ "{topic} IFRS IAS"
  ├─ "{topic} SEC regulation"
  └─ "{topic}"                        ← 通用兜底
  ↓ 按 (source, text[:80]) 去重合并
compare_standards(...)
  ↓ COMPARISON_PROMPT_ZH / _EN
  ↓ max_tokens=1500, temperature=0.1
  ↓ 输出 Markdown 表格 + 关键差异摘要 + 资料覆盖度(高/中/低)
```

四个 query 是硬编码的。**去重键是 `(source, text[:80])`**，即同一文件前 80 字符相同
的片段会被合并——这在多准则场景下可能误杀来自同一文件的真实差异片段，改动时需权衡。

对比 prompt 相比普通问答多了两项硬要求：**必须输出 Markdown 表格**，
以及**显式标注资料覆盖完整度**。后者是刻意设计——让模型承认资料不足，
避免编造对比。

多准则对比**不使用 system prompt**（直接走 user message），与 `ask()` 不同。

### 4.8 知识图谱

`build_knowledge_graph()` 返回 pyvis `Network`，页面上通过
`st.components.v1.html(net.generate_html(), height=620)` 渲染。

算法（**纯关键词，不走 embedding**）：

1. 按 `source` 聚合片段 → 蓝色方块文档节点
2. 逐文档全文小写匹配 `KEYWORDS` → 命中的标准/术语
3. 橙色椭圆标准节点，`title` 显示被几份文档提到
4. 文档 → 标准 灰线
5. **文档间相似度 = 两文档共同命中的关键词数 ≥ 3** → 绿色粗线
6. forceAtlas2 物理引擎，节点可拖拽

`KEYWORDS` 是硬编码列表，覆盖 US GAAP（ASC 860/805/606/980/326）、IFRS
（IFRS 10/IAS 32/IFRS 7/9/IAS 39）、SEC Rules（15c3-3/15c3-1）、Federal Reserve
（Reg T/D/W）、PCAOB/AICPA/Basel，以及托管业务中英术语。**扩充领域就改这个列表。**

两两比较是 O(n²)，文档数很多时会变慢。

### 4.9 国际化与 UI

`i18n.py` 提供 `TRANSLATIONS = {"zh": {...}, "en": {...}}` 与
`t(key, **kwargs) -> str`。缺 key 时**回退返回 key 本身**，不会崩。

当前规模：中英各 **64 个 key，两边完全一致**；主程序有 **50 处真实 `t()` 调用**，
全部有双语覆盖。另有 14 个 key 定义了但未被使用（早期 UI 重构后的遗留），保留无害。

> 校验脚本思路：用 `ast` 遍历出所有 `Call(func=Name('t'))` 节点，
> 与 `TRANSLATIONS['zh'].keys()` 求差集。用正则匹配 `t("...")` 会误判
> （会命中 `st.caption(t(...))` 等嵌套写法），不要用正则做这个检查。

主程序在 UI 段做三件事：

```python
_lang_from_url = st.query_params.get("lang", "")   # URL 参数优先
if _lang_from_url in ("zh", "en"):
    st.session_state["lang"] = _lang_from_url     # 写入 session
if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"               # 默认中文
i18n.LANG = st.session_state["lang"]              # 同步给 t()
_UI_LANG = st.session_state["lang"]               # 同步给 prompt 层
```

语言状态优先级：**URL `?lang=en` > session_state > 默认 zh**。
切换语言时会同时写 session 和 query_params 再 `st.rerun()`，
所以语言可分享、刷新不丢。

界面布局：

- **侧边栏**：只有文档上传 / 构建 / 清空 / 规模显示
- **右上角 `⚙️ 设置` popover**：语言切换 + Embedding API 测试 + LLM API 测试
  （用空左列 `[6, 1]` 把按钮推到最右）
- **主区域**：三个 Tab（智能问答 / 多准则对比 / 知识图谱）

**Streamlit 陷阱**：`st.button()` **不支持** `label_visibility` 参数，
只有 `st.text_input` / `st.file_uploader` / `st.multiselect` 支持。给 button 传这个
参数会抛 `TypeError`。

**「清空知识库」按钮为什么不能直接判断模块级 `_vs`**：Streamlit 每次交互都会
**从头重跑整个脚本**，模块级 `_vs = None` 必然先被执行一次。原实现里清空逻辑
写在 `get_vs()` 首次实例化之前，因此判断时 `_vs` 永远是 `None`，
`reset()` 从不执行，**但仍然弹出「知识库已清空」的成功提示**——一个静默的 no-op。
正确写法是先 `get_vs()` 拿到实例再 `reset()`。

**错误处理模式**在三个 Tab 里一致：捕获异常 → `st.error()` → `st.exception()`。
调试时优先看终端 traceback。

---

## 5. 关键接口契约

| 函数 | 签名 | 返回 | 调用方 |
|---|---|---|---|
| `load_pdf` | `(file)` | `str` | 侧边栏建库 |
| `chunk` | `(text, size=600, overlap=80)` | `list[str]` | 侧边栏建库 |
| `embed` | `(texts, type_="db")` | `list[list[float]]` | 向量层、测试按钮 |
| `index_docs` | `(chunks, source)` | `None` | 侧边栏建库 |
| `search` | `(query, k=5)` | `list[{text,source,score}]` | Tab1、Tab2 |
| `ask` | `(question, ctxs)` | `str` | Tab1 |
| `compare_standards` | `(topic, standards)` | `(str, ctxs)` | Tab2 |
| `build_knowledge_graph` | `()` | `Network \| None` | Tab3 |
| `test_embedding_api` | `()` | `(bool, str)` | 设置菜单 |
| `test_llm_api` | `()` | `(bool, str)` | 设置菜单 |
| `t` | `(key, **kwargs)` | `str` | 全 UI |

### 5.1 返回结构约定

检索结果统一是：

```python
{"text": "<片段原文>", "source": "<文件名>", "score": <float>}
```

UI 层直接依赖 `c["source"]` / `c["text"]` / `c["score"]` 三个键。
**改动返回结构必须同步改三处 UI 渲染代码**（Tab1 引用区、Tab2 引用区）。

### 5.2 索引文件契约

| 文件 | 内容 | 注意 |
|---|---|---|
| `vector_store/faiss.index` | FAISS 索引 | 与 metadata.pkl 必须成对 |
| `vector_store/metadata.pkl` | `[{"text","source"}, ...]` | 靠下标与向量 ID 对齐 |

路径基于 `Path(__file__).resolve().parent`，**不随 cwd 变化**。

---

## 6. Provider 切换矩阵

| 场景 | LLM | Embed | Vector | 成本 |
|---|---|---|---|---|
| 本地调试（推荐起步） | `local_qwen` | `local` | `faiss` | 0 |
| 本地 LLM + 云端向量 | `local_qwen` | `MiniMax` | `faiss` | 低 |
| 本地 LLM + Bedrock 向量 | `local_qwen` | `bedrock` | `faiss` | 低 |
| 纯 MiniMax | `MiniMax` | `MiniMax` | `faiss` | 中 |
| 比赛 AWS 环境 | `bedrock` | `bedrock` | `opensearch` | AWS 内网 |

三层互不依赖，可自由组合。

**默认组合是 `local_qwen` + `local` + `faiss`**，理由：零成本、零外网依赖、
可在无 AWS 环境下完整演示。

---

## 7. 部署形态

### 7.1 本地开发

```powershell
cd D:\working\T204
pip install -r requirements.txt
copy .env.example .env       # 填入实际配置
# 国内网络:本地 embedding 首次下载走镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
streamlit run rag_skeleton.py
```

默认监听 `http://localhost:8501`。语言可通过 `?lang=en` 直接进入英文界面。

### 7.2 比赛 / AWS 环境

```powershell
# .env 切换
LLM_PROVIDER=bedrock
EMBED_PROVIDER=bedrock
VECTOR_STORE=opensearch
AWS_REGION=us-east-1
OS_HOST=<真实 endpoint>
OS_USER=<...>
OS_PASS=<...>
```

`boto3` / `opensearch-py` 需要额外安装（`requirements.txt` 中已注释）。

内网闭环要点：**模型与向量检索全部走 VPC 内网 endpoint，不出公网**，
满足零出网合规要求。

---

## 8. 扩展指南

### 8.1 新增一个 LLM provider

1. 在配置段加 `LLM_PROVIDER_XXX` 与对应环境变量
2. 写 `_ask_xxx(question, ctxs) -> str`，返回纯文本
3. 在 `ask()` 加一个 `if` 分支
4. 在 `test_llm_api()` 加对应分支（**必须加**，否则设置菜单里的连通性测试会测错 provider）
5. 更新 `.env.example` 与本文档 §6

新 provider 只要返回 OpenAI 兼容的 `{"choices":[{"message":{"content":...}}]}`，
`_extract_llm_text()` 可以直接复用，无需改动。

### 8.2 新增一个 Embedding provider

1. 写 `_embed_xxx(texts) -> list[list[float]]`
2. 在 `embed()` 加分支
3. 在 `test_embedding_api()` 加分支
4. 确认返回维度并更新 `EMBED_DIM` 默认值
5. **清空现有索引**（换了 embedding 模型，旧索引的向量空间完全不同）

### 8.3 新增一种文档格式

改 `load_pdf()`：加一条 magic number / 后缀判定分支 + 一个解析分支。
注意保留末尾 `len(text) < 50` 的统一失败出口。

### 8.4 新增一个语种

1. 在 `i18n.py` 的 `TRANSLATIONS` 加一个 key 字典
2. 补齐所有 key——缺 key 会回退显示 key 名本身，**页面上会出现裸 `tab_qa` 这种字符串**
3. 加 `SYSTEM_PROMPT_XXX` / `COMPARISON_PROMPT_XXX` 并在 UI 段按语言选择
4. 语言按钮的选项元组 `"zh", "en"` 需要扩展

### 8.5 扩充知识图谱领域

直接编辑 `KEYWORDS` 列表。加词条不需要改任何其他代码。

---

## 9. 故障排查（已验证）

| 现象 | 根因 | 处理 |
|---|---|---|
| `numpy.core.multiarray failed to import` | numpy 2.x 与 faiss-cpu 1.8 ABI 冲突 | 降到 `numpy==1.26.4` |
| scipy 报 `np.long` 缺失 | scipy 1.13+ 需要 numpy 2.x | 降到 `scipy==1.12.0` |
| `PermissionError: vector_store` | 用了相对路径 | 保持 `Path(__file__).resolve().parent` |
| MiniMax embedding `IndexError` | 用了 OpenAI 风格的 `input` 字段 | 改用 `texts` + `type` |
| MiniMax HTTP 200 但报错 | 业务错误藏在 `base_resp` 里 | 检查 `base_resp.status_code != 0` |
| `insufficient balance` | 账户余额不足 | 改用 `EMBED_PROVIDER=local` |
| HF 下载 `WinError 10060` | 直连 huggingface.co 超时 | `$env:HF_ENDPOINT="https://hf-mirror.com"` |
| Ollama 长回答超时 | 14B CPU 推理慢 | timeout 已提到 300s；再不行换 7B 或减小 k |
| `TypeError` on `label_visibility` | `st.button` 不支持该参数 | 只在 input/uploader/multiselect 上用 |
| `SyntaxError: assigned before global declaration` | 顶层写了 `global _UI_LANG` | 删掉 `global`，顶层赋值本身就是全局 |
| `NameError: LANG_FULL` | 变量名拼错 | 正确是 `LANG_FULL` |
| 回答为空但无报错 | LLM 返回空 content | 看终端 `[DEBUG]` 打印的原始响应 |
| Windows 终端打印 emoji 崩 | 控制台默认 GBK | 脚本里包 `io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")` |
| `uv` 命令不存在 | Windows 环境未装 | 用 `python` 直接调 |

---

## 10. 已知限制与待修项

### 10.1 功能限制

1. **仅文本型 PDF** — 扫描件需要 OCR，当前会直接抛"内容过短"
2. **块级引用** — 切分不感知句子边界，引用片段可能从中间截断；没有页码级定位
3. **单轮问答** — 无上下文记忆，同一会话里不能追问
4. **BM25 混合检索仅 OpenSearch 模式** — FAISS 只走向量检索
5. **知识图谱仅 FAISS 模式可用** — OpenSearch 模式下 `get_vs()` 返回 `None`，图谱 Tab 直接报知识库为空
6. **OpenSearch 模式无清空按钮** — 侧边栏规模永远显示 0
7. **知识图谱 O(n²)** — 文档数量大时构图变慢
8. **`multi_query_search` 的四个 query 硬编码** — 无法按准则动态生成
9. **多准则对比去重可能误杀** — 去重键 `(source, text[:80])` 对同文件相似片段不区分

### 10.2 代码层面的待改进项

- `_vs` 缓存只在单次脚本执行内有效。Streamlit 每次交互重跑脚本，
  意味着 `LocalFAISS._load()` 会**反复从磁盘加载索引**。文档量大后应改用
  `@st.cache_resource` 缓存实例。
- `OS_PASS` 在代码里有硬编码默认密码（`YourStrongPassword_123`）。
  虽然会被环境变量覆盖，但默认值不应出现在源码里。
- `build_knowledge_graph()` 每两文档拼一次全文做关键词匹配
  （`" ".join(docs[s]["texts"])`），应提到循环外缓存。
- `compare_standards()` 里 LLM 调用逻辑与 `ask()` 重复了三份，可抽公共函数。

---

## 11. 合规约束（写代码时必须遵守）

1. **零内部数据**：不接受、不接入任何公司内部文件、系统、客户数据。
   `data/` 目录下的演示语料全部是 IRS / SEC / Basel 等公开资料。
2. **可审计**：每条结论必须有 `[来源N]`，UI 必须展示原始片段。
   任何"优化"掉引用展示的改动都要慎重。
3. **不出公网**（AWS 环境）：Bedrock 和 OpenSearch 都走 VPC 内网 endpoint。
4. **数据不用于训练**：AWS Bedrock 明确承诺输入输出不用于基础模型训练。
5. **不承诺做不到的事**：不要在文档或演示中承诺页码级引用、
   扫描件支持、多轮记忆——这些当前都做不到。

---

## 12. 附：文件与数据清单

```
D:\working\T204\
├── rag_skeleton.py       主程序，单文件，1140 行 / 27 个顶层函数 / 1 个类
├── i18n.py               中英文案字典，中英各 64 个 key
├── requirements.txt      已验证的依赖组合
├── .env.example          环境变量模板
├── download_demo_data.py 演示语料下载脚本
├── data/                 演示语料（公开资料）
│   ├── IRS_P550_Investment_Income.pdf     投资收入
│   ├── IRS_P552_Recordkeeping.pdf         记录保存
│   ├── IRS_P559_Survivors.pdf             遗产受益人
│   ├── IRS_P946_How_To_Depreciate.pdf     折旧
│   ├── BNY_Mellon_10K_latest.xml          SEC XBRL 实例
│   ├── Basel_III_Framework.html           巴塞尔协议 III
│   ├── Basel_Operational_Risk.html        操作风险
│   └── ASC860_KeyPoints.md                ASC 860 要点
├── vector_store/         FAISS 索引（自动生成，不入库）
├── slides/               中文版 PPT 源码
└── slides_en/            英文版 PPT 源码
```

`data/` 里的文件**需要通过界面上传**，应用不会自动扫描目录。

---

## 13. 快速自检清单

改完代码后按顺序过一遍：

- [ ] `python -c "import ast; ast.parse(open('rag_skeleton.py',encoding='utf-8').read())"` 通过
- [ ] 新增的 i18n key 在 `zh` 和 `en` **两边都补齐**（缺 key 会显示裸 key 名）
- [ ] 改了 `chunk()` 参数 → 重建索引
- [ ] 换了 embedding 模型 / 维度 → **先清空知识库**
- [ ] 新增 provider → 同步更新 `test_*_api()`
- [ ] 改了 `search()` 返回结构 → 同步改 Tab1 / Tab2 的引用渲染
- [ ] 新增文档格式 → 保留 `len(text) < 50` 的失败出口
- [ ] 设置菜单里的两个连通性测试按钮能测到新 provider