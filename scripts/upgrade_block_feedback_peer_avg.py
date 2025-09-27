"""Add peer_avg_* columns to block_feedback if they do not exist (legacy DB compatibility).
Run: python scripts/upgrade_block_feedback_peer_avg.py
"""
import sqlite3, os
DB_PATH = 'test.db'
if not os.path.exists(DB_PATH):
    print('Database not found, nothing to upgrade.')
    raise SystemExit(0)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('PRAGMA table_info(block_feedback)')
cols = {row[1] for row in cur.fetchall()}
needed = [
    ('peer_avg_top1_pre','FLOAT'),
    ('peer_avg_top1_post','FLOAT'),
    ('peer_avg_top3_pre','FLOAT'),
    ('peer_avg_top3_post','FLOAT'),
]
changed = False
for name, typ in needed:
    if name not in cols:
        cur.execute(f'ALTER TABLE block_feedback ADD COLUMN {name} {typ}')
        changed = True
        print('Added column', name)
if changed:
    conn.commit()
else:
    print('All peer_avg_* columns already present.')
conn.close()
print('Upgrade complete.')
