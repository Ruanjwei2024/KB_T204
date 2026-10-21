# 📒 账枢 LedgerMind

> 托管资产会计 AI 智库 —— 基于 RAG 的银行业公开法规 / 准则智能问答系统

一个 **完全基于公开数据** 的垂直 AI 知识库，专注于托管资产会计（custody accounting）领域，可用于学习、培训、自助查询。**零内部数据、零公司系统依赖**。

---

## 📌 项目背景

- **赛事**：Global Hackathon（银行 IT 部门）
- **方向**：AI 知识库
- **时间**：2-3 天
- **约束**：不允许使用公司数据 / 文件 / 系统；公司电脑可连接 AWS（且仅 AWS）
- **公司**：美资外企，部门为托管资产会计

## 🎯 核心亮点

| 亮点 | 说明 |
|---|---|
| **合规友好** | 全部数据来自 FASB / SEC / IRS / Fed / Basel 等公开源 |
| **多 Provider 切换** | 一套代码同时支持 AWS Bedrock / MiniMax API / 本地 Ollama |
| **本地可调试** | 无需 AWS 也无需付费 API，本地 + Ollama + BGE 即可跑通 |
| **AWS 闭环** | 比赛环境跑在 AWS 内网，零出网，符合金融合规 |
| **可引用回答** | 每个回答强制标注原文片段来源，避免幻觉 |
| **多准则对比** | 同一议题自动分别检索 US GAAP / IFRS / SEC Rules，生成并排对比表与差异摘要 |
| **知识图谱视图** | 基于文档、准则与术语关键词构建可交互图谱，快速识别知识盲区 |
| **多格式解析** | PDF / HTML / XBRL / Markdown / TXT 统一入库 |
| **中英双语** | 界面与 AI 回答语言跟随选择，一键切换 |

---

## 🏗️ 架构

```
┌────────────────────────────────────────────────────────┐
│  Streamlit UI  (本地)                                   │
│  ├─ 侧边栏: 上传文档 (PDF/HTML/XBRL/MD/TXT)            │
│  ├─ 右上角 ⚙ 设置: 语言切换 + API 连通性测试           │
│  ├─ 提问 + 引用展示                                     │
│  ├─ 多准则对比 (US GAAP / IFRS / SEC Rules)             │
│  └─ 知识图谱 (pyvis 交互视图)                           │
└──────────────────┬─────────────────────────────────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
┌─────▼──────┐         ┌────────▼────────┐
│   LLM      │         │   Embedding     │
│  (推理)    │         │   (向量化)      │
├────────────┤         ├─────────────────┤
│ • Bedrock  │         │ • Bedrock Titan │
│ • MiniMax  │         │ • MiniMax API   │
│ • local    │         │ • local (BGE)   │
│   (Qwen)   │         │                 │
└────────────┘         └────────┬────────┘
                                │
                       ┌────────▼─────────┐
                       │   向量库          │
                       ├──────────────────┤
                       │ • FAISS (本地)   │
                       │ • OpenSearch(AWS)│
                       └──────────────────┘
```

---

## 🧰 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Streamlit |
| 后端 | Python 3.12 |
| LLM | AWS Bedrock (Claude 3.5 Sonnet) / MiniMax API / Ollama (Qwen 2.5) |
| Embedding | Amazon Titan v2 / MiniMax / BAAI BGE (本地) |
| 向量库 | FAISS (本地) / OpenSearch (AWS) |
| 文档解析 | pypdf (PDF) / BeautifulSoup (HTML) / xml.etree (XBRL) / Markdown / TXT |
| 图谱 | pyvis |
| 国际化 | 自建 `i18n.py` (中英各 64 个 key) |
| 环境管理 | python-dotenv |

---

## 🎤 项目介绍 PPT

按公司模板视觉风格生成的 11 页中文项目介绍 PPT，产物：

```
D:\working\T204\slides\output\LedgerMind-托管资产会计AI智库-项目介绍.pptx
D:\working\T204\slides_en\output\LedgerMind-Custody-Accounting-AI-KB-English.pptx   # 英文版
```

两版页面结构、配色、版式完全一致，英文版字体改用 Arial，其余逐页重新配平（英文比中文长，
字号和文本框宽度都做了调整）。中文版原稿不受英文版改动影响。

重新生成：

```powershell
cd D:\working\T204\slides
node lint.js        # 预编译检查: 拒绝中文全角标点 / 智能引号
node compile.js     # 编译 11 页到 output/
python check_deck.py  # 静态校验: 画布越界 / 页码 / 颜色 / 占位文本
powershell -ExecutionPolicy Bypass -File .\render_png.ps1  # 逐页渲染 PNG 复核

cd D:\working\T204\slides_en
node lint.js        # 英文版检查: 源码不得出现任何 CJK 字符
node compile.js     # 编译英文版
python ..\slides\check_deck.py .\output\LedgerMind-Custody-Accounting-AI-KB-English.pptx Arial
powershell -ExecutionPolicy Bypass -File ..\slides\render_png.ps1 -BaseDir D:\working\T204\slides_en
```

页面结构：封面 / 目录 / 章节页 01 / 六大痛点 / 知识孤岛四类风险 / 八大挑战 /
章节页 02 / 三大核心能力 / 技术架构与合规 / 演示价值与差异化 / THANKS。11 页均带演讲者备注。

模板仿制要点：蓝底(304EFC) + 白色圆角卡片 + 红黄绿窗口圆点 + 超大章节数字压边，
中文版字体沿用模板实际的微软雅黑。源码为每页一个 `slide-NN.js`，同步导出
`createSlide(pres, theme)`，`theme.js` 只保留 primary/secondary/accent/light/bg 五个键。

---

## 📐 技术文档

面向维护者的技术说明：[ARCHITECTURE.md](./ARCHITECTURE.md)

涵盖运行架构与数据流、五个核心模块的实现细节、函数接口契约、embedding 维度等
关键不变量、provider 扩展指南、已验证的故障排查表、已知限制与合规红线。

本 README 面向使用者与评委；架构、接口契约和排障细节看 `ARCHITECTURE.md`。

---

## 📂 目录结构

```
D:\working\T204\
├── ARCHITECTURE.md        # 技术架构与实现说明(维护者向)
├── rag_skeleton.py          # 主程序 (1140 行 / 27 个顶层函数 / 1 个类)
├── i18n.py                  # 中英文案 (各 64 个 key)
├── requirements.txt         # Python 依赖(已验证兼容组合)
├── .env.example             # 环境变量模板
├── .env                     # 实际配置(含真实 key,禁止提交 / 泄露)
├── README.md                # 本文件
├── data/                    # 演示语料 (IRS PDF / SEC XML / Basel HTML / MD)
├── slides/                  # 中文版 PPT 源码
│   ├── theme.js             # 5 键主题
│   ├── lib.js               # 共享设计令牌与模板组件
│   ├── slide-01..11.js      # 每页一个模块
│   ├── lint.js              # 预编译检查(禁止中文全角标点)
│   ├── compile.js           # 编译入口
│   ├── check_deck.py        # 静态校验
│   ├── dump_text.py         # 导出全文供内容 QA
│   ├── render_png.ps1       # PowerPoint COM 渲染(支持 -BaseDir)
│   ├── previews/            # 逐页渲染图
│   ├── template_previews/   # 模板原页渲染图(比对用)
│   └── output/              # 产出的 pptx
├── slides_en/               # 英文版 PPT 源码(结构同上, 字体 Arial)
└── vector_store/            # FAISS 索引和元数据(自动生成)
    ├── faiss.index
    └── metadata.pkl
```

---

## 🚀 快速开始

### 1. 安装依赖

```powershell
# 推荐 Python 3.11 / 3.12
pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
# 复制模板
copy .env.example .env
# 编辑 .env 填入真实 key / 路径
notepad .env
```

### 3. 启动

```powershell
# 国内用户,加镜像加速 embedding 模型下载
$env:HF_ENDPOINT = "https://hf-mirror.com"

streamlit run rag_skeleton.py
```

浏览器自动打开 `http://localhost:8501`

---

## ⚙️ 配置说明

`rag_skeleton.py` 顶部所有可调参数都通过环境变量控制：

| 变量 | 可选值 | 默认 | 说明 |
|---|---|---|---|
| `LLM_PROVIDER` | `bedrock` / `MiniMax` / `local_qwen` | `local_qwen` | LLM 后端 |
| `EMBED_PROVIDER` | `bedrock` / `MiniMax` / `local` | `local` | Embedding 后端 |
| `VECTOR_STORE` | `faiss` / `opensearch` | `faiss` | 向量库 |
| `LOCAL_LLM_BASE` | URL | `http://localhost:11434/v1` | Ollama / vLLM / LM Studio |
| `LOCAL_LLM_MODEL` | 模型名 | `qwen2.5` | 本地 LLM 模型 |
| `LOCAL_EMBED_MODEL` | HF 模型名 | `BAAI/bge-base-zh-v1.5` | 本地 Embedding 模型 |
| `EMBED_DIM` | 整数 | `768` | Embedding 维度(必须与模型匹配) |
| `MiniMax_API_KEY` | 字符串 | - | MiniMax 订阅 Key |
| `MiniMax_BASE` | URL | `https://api.MiniMax.chat/v1` | MiniMax API 地址 |
| `LLM_MODEL_MINIMAX` | 模型名 | `MiniMax-Text-01` | MiniMax LLM 模型 |
| `AWS_REGION` | Region | `us-east-1` | Bedrock 区域 |
| `OS_HOST` / `OS_USER` / `OS_PASS` | 字符串 | 占位 | OpenSearch 终端 |

### 常用 Provider 组合

| 场景 | LLM | Embed | Vector |
|---|---|---|---|
| **本地调试(零成本)** | `local_qwen` | `local` | `faiss` |
| **本地 + 云端 LLM** | `MiniMax` | `local` | `faiss` |
| **比赛 AWS 环境** | `bedrock` | `bedrock` | `opensearch` |
| **纯云端(都花钱)** | `MiniMax` | `MiniMax` | `opensearch` |

---

## 📚 数据源

本项目**完全不接触公司内部数据**，所有数据来自公开权威源：

| 数据源 | 类型 | URL |
|---|---|---|
| FASB ASC (860, 805, 606, 980) | 会计准则 | https://asc.fasb.org |
| SEC EDGAR (BNY Mellon / State Street / Citi 10-K) | 上市公司财报 | https://www.sec.gov/edgar |
| SEC Rules (15c3-3, 15c3-1) | 监管规则 | https://www.sec.gov |
| Federal Reserve Regulations | 美联储规则 | https://www.federalreserve.gov |
| AICPA Audit Guides | 审计指引 | https://www.aicpa.org |
| PCAOB AS 1200 | 审计准则 | https://pcaobus.org |
| Basel Committee | 国际标准 | https://www.bis.org |
| IRS Publications | 税务规则 | https://www.irs.gov |

---

## 🛡️ 合规说明（讲给评委听）

1. **零内部数据风险** —— 全部数据来自公开权威源，不接触任何客户信息 / 内部系统
2. **AWS 闭环** —— 比赛环境跑在 AWS 内网，**不出网**
3. **可审计** —— 每次回答强制标注原文片段来源 ([来源N])
4. **数据不用于训练** —— AWS Bedrock 明确承诺输入输出不用于训练基础模型
5. **可私有化** —— FAISS 本地版可直接部署在客户内网

---

## ⚠️ 已知问题 / 限制

1. **PDF 解析仅支持文本型** —— 扫描件需要 OCR
2. **本地大模型推理慢** —— 14B 在 CPU 上首次需 30-120 秒
3. **BM25 混合检索** —— 仅 OpenSearch 模式支持,FAISS 模式只走向量检索
4. **多轮对话** —— 当前为单轮,未实现上下文记忆
5. **章节定位** —— 引用只到 chunk 级别,未到页码级

---

## 🛠️ 调试经验记录

| 问题 | 解决 |
|---|---|
| `IndexError: tuple index out of range` | MiniMax API 期望 `texts` 字段,不是 `input` |
| `insufficient balance` | MiniMax 账户余额不足,改用本地 embedding |
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| `numpy.core.multiarray failed to import` | numpy 2.x 与 faiss-cpu 冲突,降到 1.26 |
| `scipy ... np.long` 报错 | scipy 1.13+ 要 numpy 2.x,降到 1.12 |
| HuggingFace 连接超时 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| `PermissionError: vector_store` | 用绝对路径 `Path(__file__).parent` |
| Ollama ReadTimeout | 14B 模型首次慢,timeout 提到 300s |
| `label_visibility` 报错 | Streamlit 旧版不支持,只在 `text_input` 用 |

---

## 🗺️ 后续计划（P1 / P2）

### Day 2：差异化功能
- [x] **多准则对比**：US GAAP / IFRS / SEC Rules 并排对比表 + 差异摘要
- [x] **知识图谱视图**：基于文档、准则与术语关键词的交互式图谱
- [x] **多格式解析**：PDF / HTML / XBRL / Markdown / TXT 统一入库
- [x] **中英双语**：界面与 AI 回答语言跟随设置菜单切换
- [ ] **合规清单生成**：输入业务场景 → 输出涉及的所有监管报告
- [ ] **多轮对话记忆**：让 demo 更像产品

### Day 3：演示打磨
- [x] 准备 4 份核心演示文档 (IRS P550 / P552 / P559 / P946)
- [x] 项目介绍 PPT (11 页，按公司模板风格)
- [ ] 写 3 分钟演示脚本
- [ ] 录制 demo 视频

### 锦上添花
- [ ] 时间线视图
- [ ] 术语词典
- [ ] 章节级引用(到页码)
- [ ] 多模态(表格 / 图片)
- [ ] 检索评估 (Recall@5, MRR)

---

## 📖 参考资料

- [Dify](https://github.com/langgenius/dify) - LLM 应用平台
- [FastGPT](https://github.com/labring/FastGPT) - 国内开源知识库
- [RAGFlow](https://github.com/infiniflow/ragflow) - 深度文档理解
- [BAAI BGE](https://github.com/FlagOpen/FlagEmbedding) - 中文 Embedding
- [Ollama](https://ollama.com/) - 本地 LLM 运行
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索
- [sentence-transformers](https://www.sbert.net/) - Embedding 库

---

## 📄 许可

MIT License (待定)

---

> **TL;DR**: 一个用 RAG 技术把公开金融法规变成可对话 AI 的项目,2-3 天能跑通,合规友好,AWS 闭环。
