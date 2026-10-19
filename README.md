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
| **合规友好** | 全部数据来自 FASB / SEC / Fed / AICPA 等公开源 |
| **多 Provider 切换** | 一套代码同时支持 AWS Bedrock / MiniMax API / 本地 Ollama |
| **本地可调试** | 无需 AWS 也无需付费 API，本地 + Ollama + BGE 即可跑通 |
| **AWS 闭环** | 比赛环境跑在 AWS 内网，零出网，符合金融合规 |
| **可引用回答** | 每个回答强制标注原文片段来源，避免幻觉 |

---

## 🏗️ 架构

```
┌────────────────────────────────────────────────────────┐
│  Streamlit UI  (本地)                                   │
│  ├─ 上传 PDF                                            │
│  ├─ 测试 LLM / Embedding                                │
│  └─ 提问 + 引用展示                                      │
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
| PDF 解析 | pypdf |
| 环境管理 | python-dotenv |

---

## 📂 目录结构

```
D:\working\
├── rag_skeleton.py          # 主程序 (~600 行,单文件)
├── requirements.txt         # Python 依赖(已验证兼容组合)
├── .env.example             # 环境变量模板
├── .env                     # 实际配置(本地调试用,不在 git)
├── README.md                # 本文件
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
- [ ] **多准则对比**：US GAAP vs IFRS 在托管业务上的差异表
- [ ] **合规清单生成**：输入业务场景 → 输出涉及的所有监管报告
- [ ] **多轮对话记忆**：让 demo 更像产品

### Day 3：演示打磨
- [ ] 准备 3 份核心演示文档
- [ ] 写 3 分钟演示脚本
- [ ] 录制 demo 视频
- [ ] 写立项 PPT(架构图 + 合规说明 + 成本预估)

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
