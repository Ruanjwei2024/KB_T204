"""
账枢 LedgerMind — 托管资产会计 AI 智库
最小 RAG 骨架 v4 — 健壮版
  • 兼容 MiniMax embedding 的多种返回格式
  • 测试 API 按钮(在调试出问题前先验证连通性)
  • 友好的错误提示(直接显示在页面)
  • 向量库: faiss (本地) | opensearch (AWS)
  • LLM: MiniMax | bedrock
"""
import os
import json
import pickle
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import requests
import numpy as np
import faiss
from pypdf import PdfReader

# ============================================================
# 配置
# ============================================================
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "local_qwen")  # ★ 改默认: 本地 Qwen
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
VECTOR_STORE   = os.getenv("VECTOR_STORE", "faiss")

# 本地 LLM (Ollama / vLLM / LM Studio 等 OpenAI 兼容服务)
LOCAL_LLM_BASE     = os.getenv("LOCAL_LLM_BASE", "http://localhost:11434/v1")
LOCAL_LLM_MODEL    = os.getenv("LOCAL_LLM_MODEL", "qwen2.5")
LOCAL_LLM_API_KEY  = os.getenv("LOCAL_LLM_API_KEY", "ollama")  # Ollama 不需要 key,占位即可

MiniMax_API_KEY = os.getenv("MiniMax_API_KEY", "")
MiniMax_BASE    = os.getenv("MiniMax_BASE", "https://api.MiniMax.chat/v1")
LLM_MODEL_MINIMAX   = os.getenv("LLM_MODEL_MINIMAX", "MiniMax-Text-01")
EMBED_MODEL_MINIMAX = os.getenv("EMBED_MODEL_MINIMAX", "embo-01")

# 本地 embedding 模型(免账号,免费用,首次使用自动下载)
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-zh-v1.5")
# bge-base-zh-v1.5 = 768 维,适合中文
# bge-small-zh-v1.5 = 512 维,更小更快
# bge-large-zh-v1.5 = 1024 维,质量最高
# bge-m3          = 1024 维,多语言最强
EMBED_DIM           = int(os.getenv("EMBED_DIM", "768"))

AWS_REGION          = os.getenv("AWS_REGION", "us-east-1")
LLM_MODEL_BEDROCK   = "anthropic.claude-3-5-sonnet-20240620-v1:0"
EMBED_MODEL_BEDROCK = "amazon.titan-embed-text-v2:0"

OS_HOST    = os.getenv("OS_HOST", "your-search-domain.us-east-1.es.amazonaws.com")
OS_USER    = os.getenv("OS_USER", "master")
OS_PASS    = os.getenv("OS_PASS", "YourStrongPassword_123")
INDEX_NAME = "rag-custody"

# 用脚本所在目录,避免相对路径的权限问题
SCRIPT_DIR = Path(__file__).resolve().parent
FAISS_DIR  = SCRIPT_DIR / "vector_store"
try:
    FAISS_DIR.mkdir(exist_ok=True)
    # 测试可写
    test_file = FAISS_DIR / ".write_test"
    test_file.write_text("ok")
    test_file.unlink()
except Exception as e:
    st.error(f"❌ 无法创建/写入 {FAISS_DIR}: {e}\n请用管理员权限运行,或改 FAISS_DIR 到可写目录")
    st.stop()
FAISS_FILE = FAISS_DIR / "faiss.index"
META_FILE  = FAISS_DIR / "metadata.pkl"

bedrock = None
if LLM_PROVIDER == "bedrock" or EMBED_PROVIDER == "bedrock":
    import boto3
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# 懒加载本地 embedding 模型(用 local 时才下载)
_local_model = None
def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_model

os_client = None
if VECTOR_STORE == "opensearch":
    from opensearchpy import OpenSearch, helpers
    os_client = OpenSearch(
        hosts=[{"host": OS_HOST, "port": 443}],
        http_auth=(OS_USER, OS_PASS),
        use_ssl=True, verify_certs=True, timeout=30,
    )

SYSTEM_PROMPT = "你是托管资产会计 AI 助手。仅根据以下公开资料回答,并标注引用。"

# ============================================================
# ★ 新增:API 连通性测试 + 响应格式自动适配
# ============================================================
def _parse_embedding_response(resp):
    """兼容 MiniMax / Bedrock / OpenAI 多种返回格式"""
    if not isinstance(resp, dict):
        return None
    # 格式0: {"vectors": [[...], [...]]}  MiniMax 风格(批量)
    if "vectors" in resp and isinstance(resp["vectors"], list) and resp["vectors"]:
        return resp["vectors"]  # 返回 list of list
    # 格式1: {"data": [{"embedding": [...]}]}        OpenAI 标准
    if "data" in resp:
        d = resp["data"]
        if isinstance(d, list) and d:
            if isinstance(d[0], dict) and "embedding" in d[0]:
                return d[0]["embedding"]
            if isinstance(d[0], (int, float)):
                return d
        if isinstance(d, dict) and "embedding" in d:
            return d["embedding"]
    # 格式3: {"embedding": [...]}                      直接返回
    if "embedding" in resp:
        return resp["embedding"]
    return None

def test_llm_api() -> tuple[bool, str]:
    """测试 LLM 是否能正常返回"""
    if LLM_PROVIDER == "bedrock":
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "回复 OK 两个字"}],
            })
            r = bedrock.invoke_model(
                modelId=LLM_MODEL_BEDROCK, body=body,
                contentType="application/json", accept="application/json",
            )
            text = json.loads(r["body"].read())["content"][0]["text"]
            return True, f"✅ Bedrock OK | 模型: {LLM_MODEL_BEDROCK} | 返回: {text[:50]}"
        except Exception as e:
            return False, f"❌ Bedrock 失败: {type(e).__name__}: {str(e)[:300]}"
    if LLM_PROVIDER == "local_qwen":
        try:
            r = requests.post(
                f"{LOCAL_LLM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"},
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": [{"role": "user", "content": "回复 OK 两个字"}],
                    "temperature": 0.1, "max_tokens": 50,
                },
                timeout=120,
            )
            if r.status_code != 200:
                return False, f"❌ HTTP {r.status_code}: {r.text[:300]}"
            text = r.json()["choices"][0]["message"]["content"]
            return True, f"✅ 本地 LLM OK | 模型: {LOCAL_LLM_MODEL} | BASE: {LOCAL_LLM_BASE} | 返回: {text[:50]}"
        except Exception as e:
            return False, f"❌ 本地 LLM 失败: {type(e).__name__}: {str(e)[:300]}"
    # MiniMax
    try:
        r = requests.post(
            f"{MiniMax_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MiniMax_API_KEY}"},
            json={
                "model": LLM_MODEL_MINIMAX,
                "messages": [{"role": "user", "content": "回复 OK 两个字"}],
                "temperature": 0.1, "max_tokens": 50,
            },
            timeout=60,
        )
        if r.status_code != 200:
            return False, f"❌ HTTP {r.status_code}: {r.text[:300]}"
        text = r.json()["choices"][0]["message"]["content"]
        return True, f"✅ MiniMax OK | 模型: {LLM_MODEL_MINIMAX} | 返回: {text[:50]}"
    except Exception as e:
        return False, f"❌ MiniMax 失败: {type(e).__name__}: {str(e)[:300]}"


def test_embedding_api() -> tuple[bool, str]:
    """测试 embedding API 是否能正常返回向量"""
    if EMBED_PROVIDER == "bedrock":
        try:
            r = bedrock.invoke_model(
                modelId=EMBED_MODEL_BEDROCK,
                body=json.dumps({"inputText": "test"}),
                contentType="application/json", accept="application/json",
            )
            emb = json.loads(r["body"].read())["embedding"]
            return True, f"✅ Bedrock OK | 模型: {EMBED_MODEL_BEDROCK} | 维度: {len(emb)}"
        except Exception as e:
            return False, f"❌ Bedrock 失败: {type(e).__name__}: {str(e)[:300]}"
    if EMBED_PROVIDER == "local":
        try:
            vecs = _embed_local(["test"])
            return True, f"✅ 本地模型 OK | 模型: {LOCAL_EMBED_MODEL} | 维度: {len(vecs[0])}"
        except Exception as e:
            return False, f"❌ 本地模型失败: {type(e).__name__}: {str(e)[:300]}"
    # MiniMax — 使用 texts + type 参数(批量,db 类型)
    try:
        r = requests.post(
            f"{MiniMax_BASE}/embeddings",
            headers={"Authorization": f"Bearer {MiniMax_API_KEY}"},
            json={"model": EMBED_MODEL_MINIMAX, "texts": ["test"], "type": "db"},
            timeout=15,
        )
        if r.status_code != 200:
            return False, f"❌ HTTP {r.status_code}: {r.text[:300]}"
        resp = r.json()
        # 检查 base_resp 错误
        if isinstance(resp, dict) and "base_resp" in resp:
            br = resp["base_resp"]
            if isinstance(br, dict) and br.get("status_code", 0) != 0:
                return False, f"❌ API 业务错误: {br}"
        vecs = _parse_embedding_response(resp)
        if vecs is None or not vecs:
            return False, f"❌ 响应格式无法识别。原始响应:\n{json.dumps(resp, ensure_ascii=False)[:500]}"
        return True, f"✅ MiniMax OK | 模型: {EMBED_MODEL_MINIMAX} | 维度: {len(vecs[0])} | BASE: {MiniMax_BASE}"
    except Exception as e:
        return False, f"❌ 请求失败: {type(e).__name__}: {str(e)[:300]}"

# ============================================================
# 1) PDF 解析(带空内容检测)
# ============================================================
def load_pdf(file) -> str:
    pages = [p.extract_text() or "" for p in PdfReader(file).pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "PDF 解析后为空。可能原因:\n"
            "1) PDF 是扫描件(图片),需要 OCR\n"
            "2) PDF 受密码保护\n"
            "3) PDF 格式异常"
        )
    return text

# ============================================================
# 2) 文本切块
# ============================================================
def chunk(text, size=600, overlap=80):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)
            if text[i:i+size].strip()]

# ============================================================
# 3) Embedding(带格式适配 + 维度校验)
# ============================================================
def _embed_bedrock(texts):
    out = []
    for t in texts:
        body = json.dumps({"inputText": t[:8000]})
        r = bedrock.invoke_model(
            modelId=EMBED_MODEL_BEDROCK, body=body,
            contentType="application/json", accept="application/json",
        )
        out.append(json.loads(r["body"].read())["embedding"])
    return out

def _embed_local(texts):
    """本地 sentence-transformers embedding(免费,无需 API key)"""
    if not texts:
        return []
    model = _get_local_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()  # numpy → list of list

def _embed_MiniMax(texts, type_: str = "db"):
    """MiniMax embedding API:
       请求参数: {model, texts: [...], type: "db"|"query"}    ← 必须带 type
       响应: {vectors: [[...], [...]], base_resp: {status_code: 0}}
    """
    if not texts:
        return []
    r = requests.post(
        f"{MiniMax_BASE}/embeddings",
        headers={"Authorization": f"Bearer {MiniMax_API_KEY}"},
        json={"model": EMBED_MODEL_MINIMAX, "texts": texts, "type": type_},
        timeout=60,
    )
    r.raise_for_status()
    resp = r.json()
    # 业务错误检查
    if isinstance(resp, dict) and "base_resp" in resp:
        br = resp["base_resp"]
        if isinstance(br, dict) and br.get("status_code", 0) != 0:
            raise ValueError(f"MiniMax API 错误: {br}")
    vecs = _parse_embedding_response(resp)
    if vecs is None:
        raise ValueError(
            f"MiniMax embedding 响应无法解析。\n"
            f"原始响应: {json.dumps(resp, ensure_ascii=False)[:500]}"
        )
    # 兜底:如果是单条返回(只有一个 vector),转成 list
    if vecs and not isinstance(vecs[0], list):
        vecs = [vecs]
    return vecs

def embed(texts, type_: str = "db"):
    """统一入口。type_: "db" 用于入库,"query" 用于检索(仅 MiniMax 用)"""
    if not texts:
        return []
    if EMBED_PROVIDER == "bedrock":
        vecs = _embed_bedrock(texts)
    elif EMBED_PROVIDER == "local":
        vecs = _embed_local(texts)
    else:
        vecs = _embed_MiniMax(texts, type_=type_)
    # 维度校验
    if vecs and vecs[0]:
        actual = len(vecs[0])
        if actual != EMBED_DIM:
            st.warning(
                f"⚠️ 配置 EMBED_DIM={EMBED_DIM},但 API 实际返回 {actual} 维。"
                f"已自动以实际维度为准,请同步修改 .env 里的 EMBED_DIM={actual}"
            )
    return vecs

# ============================================================
# 4) FAISS 本地存储(健壮版)
# ============================================================
class LocalFAISS:
    def __init__(self, dim):
        self.dim = dim
        self.index = None
        self.meta = []
        self._load()

    def _load(self):
        if FAISS_FILE.exists() and META_FILE.exists():
            try:
                self.index = faiss.read_index(str(FAISS_FILE))
                with open(META_FILE, "rb") as f:
                    self.meta = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ 加载已有索引失败,将重建: {e}")
                self.index = None
                self.meta = []

    def _save(self):
        if self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, str(FAISS_FILE))
            with open(META_FILE, "wb") as f:
                pickle.dump(self.meta, f)

    def reset(self):
        """清空知识库"""
        self.index = None
        self.meta = []
        for p in [FAISS_FILE, META_FILE]:
            if p.exists():
                p.unlink()

    def add(self, chunks, source):
        if not chunks:
            raise ValueError("没有可索引的文本块(PDF 解析后可能为空)")
        vecs_raw = embed(chunks)
        if not vecs_raw or not vecs_raw[0]:
            raise ValueError("Embedding API 返回为空")
        # 维度一致性校验
        dims = {len(v) for v in vecs_raw}
        if len(dims) > 1:
            raise ValueError(f"Embedding 维度不一致: {dims}")
        actual_dim = dims.pop()
        if self.dim != actual_dim:
            self.dim = actual_dim
        # 显式构造 2D float32 数组
        vecs_np = np.vstack([np.asarray(v, dtype="float32") for v in vecs_raw])
        if vecs_np.shape[0] == 0 or vecs_np.shape[1] == 0:
            raise ValueError(f"向量数组维度异常: shape={vecs_np.shape}")
        faiss.normalize_L2(vecs_np)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs_np)
        for c in chunks:
            self.meta.append({"text": c, "source": source})
        self._save()

    def search(self, query_vec, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        kk = min(k, self.index.ntotal)
        scores, ids = self.index.search(q, kk)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if 0 <= int(i) < len(self.meta):
                out.append({"text": self.meta[int(i)]["text"],
                            "source": self.meta[int(i)]["source"],
                            "score": float(s)})
        return out

    def count(self):
        return 0 if self.index is None else int(self.index.ntotal)

# ============================================================
# 5) OpenSearch 实现(比赛环境)
# ============================================================
def _os_ensure_index():
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "text":      {"type": "text"},
            "embedding": {"type": "knn_vector", "dimension": EMBED_DIM,
                          "method": {"name": "hnsw", "space_type": "cosinesimil"}},
            "source":    {"type": "keyword"},
        }},
    }
    if not os_client.indices.exists(index=INDEX_NAME):
        os_client.indices.create(index=INDEX_NAME, body=body)

def _os_add(chunks, source):
    from opensearchpy import helpers
    vecs = embed(chunks)
    actions = [{
        "_op_type": "index", "_index": INDEX_NAME,
        "_source": {"text": c, "embedding": v, "source": source},
    } for c, v in zip(chunks, vecs)]
    helpers.bulk(os_client, actions)
    os_client.indices.refresh(index=INDEX_NAME)

def _os_search(query, qvec, k=5):
    body = {"size": k, "query": {"bool": {"should": [
        {"match": {"text":    {"query": query, "boost": 0.4}}},
        {"knn":   {"embedding": {"vector": qvec, "k": k}}}
    ]}}}
    hits = os_client.search(index=INDEX_NAME, body=body)["hits"]["hits"]
    return [{"text": h["_source"]["text"],
             "source": h["_source"]["source"],
             "score": h["_score"]} for h in hits]

# 工厂
_vs = None
def get_vs():
    global _vs
    if VECTOR_STORE == "opensearch":
        return None
    if _vs is None:
        _vs = LocalFAISS(EMBED_DIM)
    return _vs

def index_docs(chunks, source):
    if VECTOR_STORE == "opensearch":
        _os_ensure_index()
        _os_add(chunks, source)
    else:
        get_vs().add(chunks, source)

def search(query, k=5):
    # 查询时用 type="query",与入库的 "db" 区分
    qvec = embed([query], type_="query")[0]
    if VECTOR_STORE == "opensearch":
        return _os_search(query, qvec, k)
    return get_vs().search(qvec, k)

# ============================================================
# 6) LLM 回答
# ============================================================
def _build_user_msg(question, ctxs):
    ctx = "\n\n---\n\n".join(
        f"[来源{i+1}: {c['source']}]\n{c['text']}" for i, c in enumerate(ctxs))
    return (f"【资料】\n{ctx}\n\n【问题】{question}\n\n"
            "要求:1) 只用资料中的内容 2) 关键信息后用 [来源N] 标注 "
            "3) 资料未覆盖则明确说'未在公开资料中找到'")

def _ask_bedrock(question, ctxs):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024, "temperature": 0.2,
        "messages": [{"role": "user", "content": _build_user_msg(question, ctxs)}],
    })
    r = bedrock.invoke_model(
        modelId=LLM_MODEL_BEDROCK, body=body,
        contentType="application/json", accept="application/json",
    )
    return json.loads(r["body"].read())["content"][0]["text"]

def _ask_MiniMax(question, ctxs):
    r = requests.post(
        f"{MiniMax_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {MiniMax_API_KEY}"},
        json={
            "model": LLM_MODEL_MINIMAX,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_msg(question, ctxs)},
            ],
            "temperature": 0.2, "max_tokens": 1024,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def _ask_local_qwen(question, ctxs):
    """本地 LLM (Ollama / vLLM / LM Studio, OpenAI 兼容 API)"""
    try:
        r = requests.post(
            f"{LOCAL_LLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"},
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_msg(question, ctxs)},
                ],
                "temperature": 0.2, "max_tokens": 512,  # 14B 模型,512 够用,更快
            },
            timeout=300,  # 本地推理慢,首次加载要 30-60 秒
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ReadTimeout:
        raise TimeoutError(
            f"本地模型 {LOCAL_LLM_MODEL} 推理超过 5 分钟未返回。\n"
            f"可能原因:\n"
            f"  1) 14B 模型太大,CPU 推理慢。检查 Ollama 是否用了 GPU:\n"
            f"     `ollama ps`  看 PROCESSOR 列是 GPU 还是 CPU\n"
            f"  2) 首次加载慢,再试一次(模型已加载到内存,会快很多)\n"
            f"  3) 换更小的模型: `LOCAL_LLM_MODEL=qwen2.5:7b`\n"
            f"  4) 减少检索片段: 改代码里 search() 的 k=5 为 k=3"
        )

def ask(question, ctxs):
    if LLM_PROVIDER == "bedrock":
        return _ask_bedrock(question, ctxs)
    if LLM_PROVIDER == "local_qwen":
        return _ask_local_qwen(question, ctxs)
    return _ask_MiniMax(question, ctxs)

# ============================================================
# 7) Streamlit UI
# ============================================================
st.set_page_config("账枢 LedgerMind", "📒", layout="wide")
st.title("📒 账枢 LedgerMind — 托管资产会计 AI 智库")
st.caption(
    f"LLM: **{LLM_PROVIDER}** · Embed: **{EMBED_PROVIDER}** · "
    f"向量库: **{VECTOR_STORE}** · 数据源: FASB ASC / SEC / Fed / AICPA"
)

with st.sidebar:
    st.header("⚙️ API 连通性")
    if st.button("🔌 测试 Embedding API"):
        with st.spinner("正在加载模型(首次约 30-120 秒,会从 HuggingFace 下载 ~409MB)..."):
            ok, msg = test_embedding_api()
        if ok:
            st.success(msg)
        else:
            st.error(msg)
            if "10060" in msg or "connection" in msg.lower() or "huggingface" in msg.lower():
                st.warning(
                    "💡 国内访问 huggingface.co 不稳定,请设环境变量后重启:\n"
                    "  `$env:HF_ENDPOINT='https://hf-mirror.com'`\n"
                    "  `streamlit run D:\\working\\rag_skeleton.py`"
                )
    if st.button("🤖 测试 LLM"):
        with st.spinner("正在调用 LLM(首次可能需要加载模型)..."):
            ok, msg = test_llm_api()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.divider()
    st.header("📄 上传文档")
    files = st.file_uploader("选择 PDF 文件", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("🔨 构建知识库") and files:
        try:
            total = 0
            for f in files:
                with st.spinner(f"处理 {f.name}..."):
                    text  = load_pdf(f)
                    chunks = chunk(text)
                    index_docs(chunks, f.name)
                    total += len(chunks)
            st.success(f"✅ 已索引 {len(files)} 份 / {total} 个片段")
        except Exception as e:
            st.error(f"❌ 索引失败: {type(e).__name__}: {e}")
            st.exception(e)

    st.divider()
    if VECTOR_STORE == "faiss" and st.button("🗑️ 清空知识库", help="清空 FAISS 知识库"):
        if _vs:
            _vs.reset()
        st.success("知识库已清空,刷新页面生效")
        st.rerun()

    st.caption(f"📊 知识库规模: **{get_vs().count() if get_vs() else 0}** 个片段")

st.header("💬 提问")
q = st.text_input("问题输入", placeholder="如:ASC 860 下客户资产是否进资产负债表?", label_visibility="collapsed")
if q:
    try:
        with st.spinner("检索 + 生成中..."):
            ctxs = search(q, k=5)
            ans  = ask(q, ctxs) if ctxs else "⚠️ 知识库为空,请先上传文档"
        st.markdown("### 回答")
        st.write(ans)
        if ctxs:
            with st.expander(f"📚 {len(ctxs)} 条引用片段"):
                for i, c in enumerate(ctxs):
                    st.markdown(f"**[{i+1}] {c['source']}**  ·  相关度 {c['score']:.2f}")
                    st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
    except Exception as e:
        st.error(f"❌ 出错了: {type(e).__name__}: {e}")
        st.exception(e)
