import os
from pathlib import Path

wd = Path(os.environ["TMP"]) / "orange_preset_test"
from orange.service import build_service

svc = build_service(wd, load_model=False)
try:
    tree = svc.tree
    print("=== 一级节点 ===")
    for c in tree.root.children:
        print(f"  {c.name}  ({'leaf' if c.is_leaf else 'abstract'})")

    print("\n=== 逐级下钻 ===")
    from orange.tool_tree import tree_query

    r = tree_query("/tool", tree=tree)
    print("/tool ->", [c["path"] for c in r.get("children", [])])
    r = tree_query("/tool/os", tree=tree)
    print("/tool/os ->", [c["path"] for c in r.get("children", [])])
    r = tree_query("/tool/os/file", tree=tree)
    print("/tool/os/file ->", [c["path"] for c in r.get("children", [])])

    print("\n=== 叶子工具 schema ===")
    for c in r.get("children", []):
        print(f"  {c['path']}: {c.get('description', '')[:30]}")
finally:
    svc.close()
