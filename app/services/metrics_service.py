from __future__ import annotations
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.models import BlockFeedback

def update_peer_averages(db: Session):
    """Backfill or recompute peer average fields for all BlockFeedback rows.

    Useful if logic changed; averages are simple mean across OTHER users in same block.
    """
    feedbacks = db.execute(select(BlockFeedback)).scalars().all()
    if not feedbacks:
        return 0
    # Group by block_index
    by_block: dict[int, list[BlockFeedback]] = {}
    for f in feedbacks:
        by_block.setdefault(f.block_index, []).append(f)

    def avg(arr):
        vals = [x for x in arr if x is not None]
        return (sum(vals) / len(vals)) if vals else None

    for block_index, rows in by_block.items():
        for f in rows:
            peers = [r for r in rows if r.user_id != f.user_id]
            if not peers:
                continue
            f.peer_avg_top1_pre = avg([p.top1_accuracy_pre for p in peers])
            f.peer_avg_top1_post = avg([p.top1_accuracy_post for p in peers])
            f.peer_avg_top3_pre = avg([p.top3_accuracy_pre for p in peers])
            f.peer_avg_top3_post = avg([p.top3_accuracy_post for p in peers])
    db.commit()
    return len(feedbacks)
