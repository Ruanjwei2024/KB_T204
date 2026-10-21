# -*- coding: utf-8 -*-
"""i18n 单元测试:检查所有 key 在中英文都有,且格式化参数一致"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import i18n

zh_keys = set(i18n.TRANSLATIONS["zh"].keys())
en_keys = set(i18n.TRANSLATIONS["en"].keys())

print("=" * 60)
print("i18n key 覆盖检查")
print("=" * 60)
print(f"中文 key 数: {len(zh_keys)}")
print(f"英文 key 数: {len(en_keys)}")

only_zh = zh_keys - en_keys
only_en = en_keys - zh_keys
if only_zh:
    print(f"\n只在中文 有的 key: {sorted(only_zh)}")
if only_en:
    print(f"\n只在英文 有的 key: {sorted(only_en)}")
if not only_zh and not only_en:
    print("\n完美!中英文 key 完全一致")

# 检查格式化参数一致
print()
print("=" * 60)
print("格式化参数一致性检查")
print("=" * 60)
import re
mismatch = []
for k in sorted(zh_keys & en_keys):
    zh_v = i18n.TRANSLATIONS["zh"][k]
    en_v = i18n.TRANSLATIONS["en"][k]
    zh_params = set(re.findall(r"\{(\w+)\}", zh_v))
    en_params = set(re.findall(r"\{(\w+)\}", en_v))
    if zh_params != en_params:
        mismatch.append((k, zh_params, en_params))

if mismatch:
    for k, z, e in mismatch:
        print(f"  [MISMATCH] {k}: zh={z} en={e}")
else:
    print("所有 key 的格式化参数一致")

# 测试取值
print()
print("=" * 60)
print("取值测试")
print("=" * 60)
i18n.LANG = "zh"
print(f"[zh] app_title = {i18n.t('app_title')[:40]}...")
i18n.LANG = "en"
print(f"[en] app_title = {i18n.t('app_title')[:50]}...")
i18n.LANG = "zh"
print(f"[zh] kb_size   = {i18n.t('kb_size', n=23)}")
i18n.LANG = "en"
print(f"[en] kb_size   = {i18n.t('kb_size', n=23)}")
i18n.LANG = "zh"
print(f"[zh] graph_size = {i18n.t('graph_size', n=10, e=15)}")
i18n.LANG = "en"
print(f"[en] graph_size = {i18n.t('graph_size', n=10, e=15)}")