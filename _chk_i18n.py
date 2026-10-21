# -*- coding: utf-8 -*-
"""严格校验:主程序里 t("...") / i18n.t("...") 的 key 是否都存在于字典"""
import ast
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import i18n

src = (HERE / "rag_skeleton.py").read_text(encoding="utf-8")
tree = ast.parse(src)

# 用 AST 精确找出所有 t("key") 调用的字符串常量参数
called = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        # 函数名是 t
        fn = node.func
        fname = None
        if isinstance(fn, ast.Name):
            fname = fn.id
        elif isinstance(fn, ast.Attribute):
            fname = fn.attr
        if fname == "t" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                called.add(a0.value)

zh = set(i18n.TRANSLATIONS["zh"])
en = set(i18n.TRANSLATIONS["en"])

print(f"AST 精确提取到 {len(called)} 个 t() key")
print(f"字典中有 zh={len(zh)} en={len(en)} 个 key")
print()

missing_zh = called - zh
missing_en = called - en

if missing_zh or missing_en:
    if missing_zh:
        print(f"❌ 中文缺失: {sorted(missing_zh)}")
    if missing_en:
        print(f"❌ 英文缺失: {sorted(missing_en)}")
    sys.exit(1)
else:
    print("✅ 所有被调用的 key 在中英文字典中都存在")

unused = (zh | en) - called
print()
print(f"字典中未被主程序直接调用的 key ({len(unused)} 个,可能是 helper 或预留):")
for k in sorted(unused):
    print(f"  - {k}")