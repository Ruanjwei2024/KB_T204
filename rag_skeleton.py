"""
账枢 LedgerMind — 托管资产会计 AI 智库
最小 RAG 骨架 v4 — 健壮版
  • 兼容 MiniMax embedding 的多种返回格式
  • 测试 API 按钮(在调试出问题前先验证连通性)
  • 友好的错误提示(直接显示在页面)
  • 向量库: faiss (本地) | opensearch (AWS)
  • LLM: MiniMax | bedrock | local_qwen (Ollama)
  • 文档格式: PDF / HTML / XBRL / Markdown / TXT
  • 功能 Tab: 普通问答 | 多准则对比 | 知识图谱
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

SYSTEM_PROMPT_ZH = "你是托管资产会计 AI 助手。仅根据以下公开资料回答,并标注引用。用中文回答。"
SYSTEM_PROMPT_EN = ("You are a custody asset accounting AI assistant. Answer strictly based on the "
                    "provided public sources and include citations. Answer in English.")
SYSTEM_PROMPT = SYSTEM_PROMPT_ZH  # 运行时按语言覆盖

# UI 语言(由 UI 段在渲染时设置,给 prompt 函数用)
_UI_LANG = "zh"

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
# 1) 多格式文档解析 (PDF / HTML / XML / Markdown / TXT)
# ============================================================
def load_pdf(file) -> str:
    """智能加载:根据文件类型自动选择 PDF / HTML / XML / Markdown 解析器"""
    # 读取前几个字节
    if hasattr(file, "read"):
        head = file.read(512)
        file.seek(0)
        is_pdf = head.startswith(b"%PDF")
        is_xml = head.lstrip().startswith(b"<?xml") or b"<xbrl" in head[:200].lower()
        is_html = (head.lstrip().startswith(b"<!DOCTYPE") or
                   head.lstrip().startswith(b"<html") or
                   head.lstrip().startswith(b"<HTML"))
        name_hint = str(getattr(file, "name", "")).lower()
    else:
        is_pdf = str(file).lower().endswith(".pdf")
        is_xml = str(file).lower().endswith((".xml", ".xbrl"))
        is_html = str(file).lower().endswith((".html", ".htm"))
        name_hint = str(file).lower()

    is_md = name_hint.endswith((".md", ".markdown", ".mdown", ".mkd", ".txt", ".text"))

    if is_md:
        # Markdown / 纯文本 —— 直接读,清理标记
        import re as _re
        try:
            content = file.read() if hasattr(file, "read") else Path(file).read_bytes()
            if isinstance(content, bytes):
                for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
                    try:
                        content = content.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    content = content.decode("utf-8", errors="ignore")
            # 移除代码块围栏标记(保留内容)
            content = _re.sub(r"^```.*$", "", content, flags=_re.MULTILINE)
            # 移除行内代码反引号和图片
            content = _re.sub(r"!\[.*?\]\(.*?\)", "", content)
            # 移除链接,只保留文字
            content = _re.sub(r"\[(.*?)\]\(.*?\)", r"\1", content)
            # 移除多余强调标记(只处理成对的 __ 和 **,不动单个 _ *)
            content = _re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
            content = _re.sub(r"__([^_]+)__", r"\1", content)
            content = _re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", content)
            # 表格分隔行 |---|---|
            content = _re.sub(r"^\s*\|[\s\-:|]+\|\s*$", "", content, flags=_re.MULTILINE)
            text = content.strip()
        except Exception as e:
            raise ValueError(f"Markdown 解析失败: {e}")
    elif is_pdf:
        # 用 pypdf
        try:
            pages = [p.extract_text() or "" for p in PdfReader(file).pages]
        except Exception as e:
            raise ValueError(f"PDF 解析失败: {e}")
        text = "\n\n".join(pages).strip()
    elif is_xml:
        # XBRL/XML - 用 BeautifulSoup 的 xml 解析器
        from bs4 import BeautifulSoup
        try:
            content = file.read() if hasattr(file, "read") else Path(file).read_bytes()
            soup = BeautifulSoup(content, "xml")
            text = soup.get_text(separator="\n", strip=True)
            # 去掉过短的纯数字行(XBRL 元数据)
            lines = [l for l in text.split("\n") if len(l.strip()) > 10]
            text = "\n".join(lines)
        except Exception as e:
            raise ValueError(f"XML/XBRL 解析失败: {e}")
    elif is_html:
        # HTML - 用 BeautifulSoup
        from bs4 import BeautifulSoup
        try:
            content = file.read() if hasattr(file, "read") else Path(file).read_bytes()
            soup = BeautifulSoup(content, "html.parser")
            # 移除无关元素
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except Exception as e:
            raise ValueError(f"HTML 解析失败: {e}")
    else:
        # 兜底当 PDF 试
        try:
            pages = [p.extract_text() or "" for p in PdfReader(file).pages]
            text = "\n\n".join(pages).strip()
        except Exception as e:
            raise ValueError(f"无法识别文件格式,尝试 PDF 解析也失败: {e}")

    if not text or len(text) < 50:
        raise ValueError(
            "文档解析后内容过短(可能原因):\n"
            "1) PDF 是扫描件(图片),需要 OCR\n"
            "2) XBRL 文件只含数字元数据\n"
            "3) 文件受密码保护"
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
def _lang_code() -> str:
    """当前 UI 语言,给 prompt 用"""
    return globals().get("_UI_LANG", "zh")


def _build_user_msg(question, ctxs):
    """构造用户 prompt,语言跟随 UI 选择"""
    lang = _lang_code()
    ctx = "\n\n---\n\n".join(
        f"[来源{i+1}: {c['source']}]\n{c['text']}" for i, c in enumerate(ctxs))
    if lang == "en":
        return (f"[CONTEXT]\n{ctx}\n\n[QUESTION] {question}\n\n"
                "Requirements:\n"
                "1) Use ONLY the content from the provided context\n"
                "2) Mark key facts with [Source N] citations\n"
                "3) If the context does not cover it, say "
                "'Not found in the public sources'\n"
                "4) Answer in English.")
    return (f"【资料】\n{ctx}\n\n【问题】{question}\n\n"
            "要求:1) 只用资料中的内容 2) 关键信息后用 [来源N] 标注 "
            "3) 资料未覆盖则明确说'未在公开资料中找到'"
            "4) 用中文回答")

def _ask_bedrock(question, ctxs):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024, "temperature": 0.2,
        "system": SYSTEM_PROMPT,
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
    return _extract_llm_text(r.json(), source="MiniMax")

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
        return _extract_llm_text(r.json(), source="Ollama")
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


def _extract_llm_text(data, source="LLM") -> str:
    """从各种 LLM 响应格式中提取文本 —— 通用适配"""
    if not isinstance(data, dict):
        return f"❌ {source} 返回非 dict: {str(data)[:300]}"
    # 错误格式
    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        return f"❌ {source} API 错误: {msg[:500]}"
    # OpenAI 兼容: {"choices": [{"message": {"content": "..."}}]}
    if "choices" in data and data["choices"]:
        c = data["choices"][0]
        if isinstance(c, dict):
            if "message" in c and isinstance(c["message"], dict):
                content = c["message"].get("content", "")
                if not content:
                    # 调试:把整个响应打到终端
                    print(f"[DEBUG] {source} content 为空,原始响应: {json.dumps(data, ensure_ascii=False)[:800]}")
                return content
            if "text" in c:
                return c["text"]
    # Ollama 原生: {"response": "..."}
    if "response" in data:
        return data["response"]
    # 兜底
    print(f"[DEBUG] {source} 响应格式无法识别: {json.dumps(data, ensure_ascii=False)[:500]}")
    return f"❌ {source} 响应格式无法识别: {json.dumps(data, ensure_ascii=False)[:500]}"

# ============================================================
# 6b) 多准则对比 —— 杀手锏功能
# ============================================================
COMPARISON_PROMPT_ZH = """你是托管资产会计专家。请基于以下公开资料,对比多个会计准则/监管标准在"{topic}"这个话题上的处理差异。

【资料】
{ctx}

【输出要求】
1) 必须用 Markdown 表格输出,格式如下:

| 维度 | {standards_str} |
|---|---|---|
| 主要规定 | ... | ... |
| 适用范围 | ... | ... |
| 会计处理 | ... | ... |
| 关键差异 | ... |
| 实务影响 | ... |

2) 每个关键结论后面用 [来源N] 标注引用
3) 资料中未找到的标准(明确写"未在公开资料中找到")
4) 在表格下方列出"关键差异摘要"(3-5 条要点)
5) 标注资料的覆盖完整度(高/中/低)
6) **用中文输出**

只输出 Markdown 表格 + 摘要 + 覆盖度,不要其他解释。"""

COMPARISON_PROMPT_EN = """You are a custody asset accounting expert. Based on the public sources below,
compare how different accounting standards / regulations handle "{topic}".

[CONTEXT]
{ctx}

[OUTPUT REQUIREMENTS]
1) MUST output a Markdown table:

| Dimension | {standards_str} |
|---|---|---|
| Main provisions | ... | ... |
| Scope | ... | ... |
| Accounting treatment | ... | ... |
| Key differences | ... |
| Practical impact | ... |

2) Mark key conclusions with [Source N] citations
3) For standards not covered by the sources, write "Not found in the public sources"
4) Below the table, add a "Key Differences Summary" (3-5 bullet points)
5) Indicate coverage completeness (High / Medium / Low)
6) **Answer entirely in English**

Output ONLY the Markdown table + summary + coverage. No other explanation."""

COMPARISON_PROMPT = COMPARISON_PROMPT_ZH  # 默认,运行时按语言覆盖


def multi_query_search(topic: str, k_per_query: int = 4):
    """多次检索覆盖不同角度,然后合并去重"""
    queries = [
        f"{topic} US GAAP ASC",
        f"{topic} IFRS IAS",
        f"{topic} SEC regulation",
        topic,  # 通用检索兜底
    ]
    seen_ids = set()
    all_ctxs = []
    for q in queries:
        try:
            results = search(q, k=k_per_query)
        except Exception:
            continue
        for i, c in enumerate(results):
            # 用 (source, text[:80]) 去重
            cid = (c["source"], c["text"][:80])
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_ctxs.append(c)
    return all_ctxs


def compare_standards(topic: str, standards: list[str]):
    """多准则对比主函数
    standards: 用户选择的准则列表,如 ["US GAAP", "IFRS", "PRC GAAP"]
    """
    # 1) 多次检索
    ctxs = multi_query_search(topic, k_per_query=4)
    if not ctxs:
        return "⚠️ 知识库为空,请先上传文档", []

    # 3) 构造 prompt —— 语言跟随 UI
    standards_str = " | ".join(standards)
    if _lang_code() == "en":
        ctx_str = "\n\n---\n\n".join(
            f"[Source {i+1}: {c['source']}]\n{c['text']}"
            for i, c in enumerate(ctxs))
        prompt_tpl = COMPARISON_PROMPT_EN
    else:
        ctx_str = "\n\n---\n\n".join(
            f"[来源{i+1}: {c['source']}]\n{c['text']}"
            for i, c in enumerate(ctxs))
        prompt_tpl = COMPARISON_PROMPT_ZH
    user_msg = prompt_tpl.format(
        topic=topic, ctx=ctx_str, standards_str=standards_str)

    # 4) 调用 LLM —— 兼容多种返回格式
    if LLM_PROVIDER == "bedrock":
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500, "temperature": 0.1,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        })
        r = bedrock.invoke_model(
            modelId=LLM_MODEL_BEDROCK, body=body,
            contentType="application/json", accept="application/json",
        )
        answer = json.loads(r["body"].read())["content"][0]["text"]
    elif LLM_PROVIDER == "local_qwen":
        try:
            r = requests.post(
                f"{LOCAL_LLM_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"},
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": [{"role": "user", "content": user_msg}],
                    "temperature": 0.1, "max_tokens": 1500,
                },
                timeout=300,
            )
            r.raise_for_status()
            answer = _extract_llm_text(r.json(), source="Ollama")
        except requests.exceptions.ReadTimeout:
            answer = "❌ 本地模型超时,请尝试减小输入或换更小模型"
    else:
        r = requests.post(
            f"{MiniMax_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MiniMax_API_KEY}"},
            json={
                "model": LLM_MODEL_MINIMAX,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": 0.1, "max_tokens": 1500,
            },
            timeout=60,
        )
        r.raise_for_status()
        answer = _extract_llm_text(r.json(), source="MiniMax")

    return answer, ctxs


# ============================================================
# 6c) 知识图谱 —— 视觉冲击
# ============================================================
# 托管业务相关关键词库(可继续扩充)
KEYWORDS = [
    # US GAAP
    "ASC 860", "ASC 805", "ASC 606", "ASC 980", "ASC 326",
    # IFRS
    "IFRS 10", "IAS 32", "IFRS 7", "IFRS 9", "IAS 39",
    # SEC Rules
    "Rule 15c3-3", "Rule 15c3-1", "15c3-3", "15c3-1",
    # Federal Reserve
    "Reg T", "Reg D", "Reg W", "Federal Reserve",
    # 其他标准
    "GAAP", "IFRS", "Basel", "PCAOB", "AICPA",
    # 业务术语
    "DVP", "Settlement", "Custody", "Safekeeping",
    "客户资金隔离", "券款对付", "托管", "结算",
]


def build_knowledge_graph():
    """构建知识图谱
    返回 pyvis Network 对象,可直接渲染为 HTML
    """
    from pyvis.network import Network
    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="#333", notebook=False, directed=False)

    vs = get_vs()
    if vs is None or not vs.meta:
        return None

    # 1) 按文档聚合
    docs = {}  # source -> {"count": int, "texts": [str]}
    for m in vs.meta:
        src = m["source"]
        if src not in docs:
            docs[src] = {"count": 0, "texts": []}
        docs[src]["count"] += 1
        docs[src]["texts"].append(m["text"])

    # 2) 添加文档节点 + 检测提到的标准
    mentioned_keywords = {}  # keyword -> set(sources)
    for src, info in docs.items():
        combined = " ".join(info["texts"])
        net.add_node(
            src, label=src[:40] + ("..." if len(src) > 40 else ""),
            title=f"<b>{src}</b><br>片段数: {info['count']}",
            color="#4A90E2", size=30, shape="box",
        )
        for kw in KEYWORDS:
            if kw.lower() in combined.lower():
                mentioned_keywords.setdefault(kw, set()).add(src)

    # 3) 添加标准/术语节点
    for kw in mentioned_keywords:
        net.add_node(
            kw, label=kw,
            title=f"<b>{kw}</b><br>被 {len(mentioned_keywords[kw])} 份文档提到",
            color="#F5A623", size=15, shape="ellipse",
        )

    # 4) 添加边:文档 → 提到的标准
    for kw, sources in mentioned_keywords.items():
        for src in sources:
            net.add_edge(src, kw, color="#aaa", width=1)

    # 5) 文档间相似度(基于内容重叠)
    sources = list(docs.keys())
    for i in range(len(sources)):
        for j in range(i+1, len(sources)):
            s1 = " ".join(docs[sources[i]]["texts"]).lower()
            s2 = " ".join(docs[sources[j]]["texts"]).lower()
            # 用关键词重叠度作为相似度
            overlap = sum(1 for kw in KEYWORDS
                         if kw.lower() in s1 and kw.lower() in s2)
            if overlap >= 3:  # 至少 3 个共同关键词
                net.add_edge(sources[i], sources[j],
                              title=f"共同关键词: {overlap} 个",
                              color="#7ED321", width=2)

    # 6) 配置物理引擎(节点可以拖动)
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150},
        "forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 150}
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "dragNodes": true},
      "edges": {"smooth": {"type": "continuous"}}
    }
    """)
    return net

# ============================================================
# 7) Streamlit UI  (i18n: 中文 / English)
# ============================================================
import i18n

# --- 语言状态:URL 参数 > session_state > 默认中文 ---
_lang_from_url = st.query_params.get("lang", "")
if _lang_from_url in ("zh", "en"):
    st.session_state["lang"] = _lang_from_url
if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"
i18n.LANG = st.session_state["lang"]

# 同步给 prompt 层(LLM 也用选定语言回答)
# 注意:模块级赋值本身就是 global,不能写 global 语句
_UI_LANG = st.session_state["lang"]
SYSTEM_PROMPT = SYSTEM_PROMPT_EN if _UI_LANG == "en" else SYSTEM_PROMPT_ZH


def t(key: str, **kwargs) -> str:
    """取当前语言文案"""
    return i18n.t(key, **kwargs)


st.set_page_config(page_title="LedgerMind", page_icon="📒", layout="wide")

# ---- 右上角设置菜单(popover):语言切换 + API 连通性测试 ----
LANG_FLAGS = {"zh": "🇨🇳", "en": "🇺🇸"}
LANG_FULL = {"zh": "简体中文", "en": "English"}

# 用空左列把按钮推到最右边
_spacer, _menu_col = st.columns([6, 1])
with _menu_col:
    with st.popover(
        f"{t('menu_settings')}",
        use_container_width=True,
        help=t("menu_settings_en") if _UI_LANG == "zh" else t("menu_settings"),
    ):
        # --- 语言切换 ---
        st.markdown(f"**{t('menu_lang_section')}**")
        _lang_cols = st.columns(2)
        for _i, _opt in enumerate(("zh", "en")):
            with _lang_cols[_i]:
                _is_cur = (_opt == _UI_LANG)
                _label = f"{LANG_FLAGS[_opt]} {LANG_FULL[_opt]}"
                if st.button(
                    _label,
                    key=f"_lang_btn_{_opt}",
                    type="primary" if _is_cur else "secondary",
                    use_container_width=True,
                ):
                    if not _is_cur:
                        st.session_state["lang"] = _opt
                        st.query_params["lang"] = _opt
                        st.rerun()

        st.divider()

        # --- API 连通性测试 ---
        st.markdown(f"**{t('menu_api_section')}**")
        if st.button(t("btn_test_embed"), use_container_width=True):
            with st.spinner(t("loading_model")):
                _ok, _msg = test_embedding_api()
            if _ok:
                st.success(_msg)
            else:
                st.error(_msg)
                if ("10060" in _msg or "connection" in _msg.lower()
                        or "huggingface" in _msg.lower() or "连接" in _msg):
                    st.warning(t("hf_blocked_warning"))

        if st.button(t("btn_test_llm"), use_container_width=True):
            with st.spinner(t("calling_llm")):
                _ok, _msg = test_llm_api()
            (st.success if _ok else st.error)(_msg)

st.divider()

# --- 主标题 ---
st.title(t("app_title"))
st.caption(
    t("app_caption").format(
        llm=LLM_PROVIDER, embed=EMBED_PROVIDER, vs=VECTOR_STORE
    )
)

# ============================================================
# 侧边栏:文档上传(API 测试已移到右上角 ⚙️ 设置菜单)
# ============================================================
with st.sidebar:
    st.header(t("sidebar_docs"))
    st.caption(t("upload_hint"))
    files = st.file_uploader(
        t("upload_label"),
        type=["pdf", "html", "htm", "xml", "xbrl", "md", "markdown", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button(t("btn_build_kb"), type="primary", use_container_width=True) and files:
        try:
            total = 0
            for f in files:
                with st.spinner(t("processing_file", name=f.name)):
                    text = load_pdf(f)
                    chunks = chunk(text)
                    index_docs(chunks, f.name)
                    total += len(chunks)
            st.success(t("index_success", n=len(files), total=total))
        except Exception as e:
            st.error(f"{t('index_failed')}: {type(e).__name__}: {e}")
            st.exception(e)

    st.divider()
    if VECTOR_STORE == "faiss" and st.button(
        t("btn_clear_kb"), help=t("clear_help"), use_container_width=True
    ):
        # 注意:Streamlit 每次交互都会重跑整个脚本,模块级 _vs 在此处必定是 None,
        # 直接判断 _vs 会让清空变成静默 no-op。必须先 get_vs() 拿到实例再 reset。
        _store = get_vs()
        if _store is not None:
            _store.reset()
        st.success(t("clear_success"))
        st.rerun()

    st.caption(t("kb_size", n=get_vs().count() if get_vs() else 0))

# ============================================================
# 主区域:三个 Tab
# ============================================================
st.header(t("qa_header"))
tab1, tab2, tab3 = st.tabs([t("tab_qa"), t("tab_compare"), t("tab_graph")])

# ============ Tab 1: 普通问答 ============
with tab1:
    q = st.text_input(
        t("qa_input_label"),
        placeholder=t("qa_placeholder"),
        label_visibility="collapsed",
    )
    if q:
        try:
            with st.spinner(t("qa_retrieving")):
                ctxs = search(q, k=5)
                ans = ask(q, ctxs) if ctxs else t("kb_empty")
            st.markdown(t("qa_answer_header"))
            if not ans or not str(ans).strip():
                st.warning(t("qa_empty_answer"))
            else:
                st.write(ans)
            if ctxs:
                with st.expander(t("qa_sources", n=len(ctxs))):
                    for i, c in enumerate(ctxs):
                        st.markdown(
                            f"**[{i+1}] {c['source']}**  ·  "
                            f"{t('qa_source_relevance', score=c['score'])}"
                        )
                        st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
        except Exception as e:
            st.error(f"{t('qa_error')}: {type(e).__name__}: {e}")
            st.exception(e)

# ============ Tab 2: 多准则对比 ============
with tab2:
    st.markdown(t("compare_intro"))
    col1, col2 = st.columns([3, 1])
    with col1:
        compare_topic = st.text_input(
            t("compare_topic_label"),
            placeholder=t("compare_topic_placeholder"),
            label_visibility="collapsed",
        )
    with col2:
        standards_selected = st.multiselect(
            t("compare_standards_label"),
            options=["US GAAP", "IFRS", "PRC GAAP", "SEC Rules",
                     "Fed Regulations", "Basel"],
            default=["US GAAP", "IFRS"],
            label_visibility="collapsed",
        )

    if st.button(t("btn_generate"), type="primary"):
        if not compare_topic:
            st.warning(t("compare_empty_topic"))
        elif not standards_selected:
            st.warning(t("compare_empty_standards"))
        else:
            try:
                with st.spinner(t("compare_spinner")):
                    answer, ctxs = compare_standards(
                        compare_topic, standards_selected
                    )
                st.markdown(t("compare_result_header"))
                if not answer or not str(answer).strip():
                    st.warning(t("compare_empty_answer"))
                else:
                    st.markdown(answer)
                if ctxs:
                    with st.expander(t("compare_sources", n=len(ctxs))):
                        for i, c in enumerate(ctxs):
                            st.markdown(
                                f"**[{i+1}] {c['source']}**  ·  "
                                f"{t('qa_source_relevance', score=c['score'])}"
                            )
                            st.caption(
                                c["text"][:300] + ("..." if len(c["text"]) > 300 else "")
                            )
            except Exception as e:
                st.error(f"{t('qa_error')}: {type(e).__name__}: {e}")
                st.exception(e)

# ============ Tab 3: 知识图谱 ============
with tab3:
    st.markdown(t("graph_intro"))
    if st.button(t("btn_generate_graph")):
        try:
            with st.spinner(t("graph_building")):
                net = build_knowledge_graph()
            if net is None:
                st.warning(t("kb_empty"))
            else:
                html = net.generate_html()
                st.components.v1.html(html, height=620)
                st.caption(
                    t("graph_size", n=len(net.nodes), e=len(net.edges))
                )
        except Exception as e:
            st.error(f"{t('qa_error')}: {type(e).__name__}: {e}")
            st.exception(e)
