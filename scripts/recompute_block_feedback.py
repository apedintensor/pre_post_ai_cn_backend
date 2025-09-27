"""Recompute BlockFeedback for a given user and block by deleting existing feedback and re-finalizing.

Usage:
  python -m scripts.recompute_block_feedback --user 2 --block 0
"""
from __future__ import annotations
import argparse
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.models import BlockFeedback, ReaderCaseAssignment
from app.services.game_service import finalize_block_if_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    args = parser.parse_args()
    db = SessionLocal()
    try:
        fb = db.execute(select(BlockFeedback).where(BlockFeedback.user_id == args.user, BlockFeedback.block_index == args.block)).scalar_one_or_none()
        if fb:
            db.delete(fb)
            db.commit()
            print("Deleted existing BlockFeedback.")
        # verify block exists and complete
        assigns = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == args.user, ReaderCaseAssignment.block_index == args.block)).scalars().all()
        if not assigns:
            print("No assignments for this block.")
            return
        remaining = sum(1 for a in assigns if a.completed_post_at is None)
        if remaining:
            print(f"Block incomplete: remaining_post={remaining}")
            return
        out = finalize_block_if_complete(db, args.user, args.block)
        if out:
            print(f"Recomputed: top1_pre={out.top1_accuracy_pre} top1_post={out.top1_accuracy_post} top3_pre={out.top3_accuracy_pre} top3_post={out.top3_accuracy_post}")
        else:
            print("No feedback created.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
