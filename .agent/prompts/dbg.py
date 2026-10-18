"""查看 harness.db 中的会话与用量记录（诊断用）。"""

from __future__ import annotations

import sqlite3
import sys

db = sqlite3.connect(".agent/harness.db")
tables = [row[0] for row in db.execute("select name from sqlite_master where type='table'")]
print("tables:", tables)
for table in ("usage", "sessions"):
    if table not in tables:
        continue
    print(f"--- {table} ---")
    columns = [row[1] for row in db.execute(f"pragma table_info({table})")]
    print("columns:", columns)
    for row in db.execute(f"select * from {table} order by rowid desc limit 5"):
        print(row)
if "--trace" in sys.argv and "traces" in tables:
    for row in db.execute("select * from traces order by rowid desc limit 3"):
        print(row)
