"""Debug a user's game blocks: prints assignments, completion status, and block feedback.

Usage:
  python -m scripts.debug_user_block --user 2

Optional:
  --block 0   # restrict to one block index
  --finalize  # attempt to finalize if a block is complete but missing feedback
"""
from __future__ import annotations
import argparse
from datetime import datetime
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.models import ReaderCaseAssignment, BlockFeedback, Case, Assessment, DiagnosisEntry
from app.services import game_service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, required=True, help="User ID")
    parser.add_argument("--block", type=int, default=None, help="Optional block index to inspect")
    parser.add_argument("--finalize", action="store_true", help="Attempt finalize if complete and missing feedback")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        user_id = args.user
        # list distinct blocks
        q = select(ReaderCaseAssignment.block_index).where(ReaderCaseAssignment.user_id == user_id).distinct().order_by(ReaderCaseAssignment.block_index.asc())
        blocks = [b for b in db.execute(q).scalars().all()]
        if not blocks:
            print(f"No assignments found for user {user_id}")
            return
        print(f"User {user_id} blocks: {blocks}")
        to_check = blocks if args.block is None else [args.block]
        for bi in to_check:
            assigns = db.execute(select(ReaderCaseAssignment).where(ReaderCaseAssignment.user_id == user_id, ReaderCaseAssignment.block_index == bi).order_by(ReaderCaseAssignment.display_order.asc())).scalars().all()
            if not assigns:
                print(f"Block {bi}: no assignments")
                continue
            total = len(assigns)
            pre_done = sum(1 for a in assigns if a.completed_pre_at is not None)
            post_done = sum(1 for a in assigns if a.completed_post_at is not None)
            remaining = total - post_done
            fb = db.execute(select(BlockFeedback).where(BlockFeedback.user_id == user_id, BlockFeedback.block_index == bi)).scalar_one_or_none()
            print(f"\nBlock {bi}: total={total} pre_done={pre_done} post_done={post_done} remaining_post={remaining} feedback_exists={bool(fb)}")
            if remaining:
                incompletes = [a for a in assigns if a.completed_post_at is None]
                for a in incompletes:
                    # show assessment phases present
                    assessments = db.execute(select(Assessment).where(Assessment.assignment_id == a.id)).scalars().all()
                    phases = sorted({ass.phase for ass in assessments})
                    print(f"  - assignment {a.id} case {a.case_id}: phases={phases} completed_pre_at={a.completed_pre_at} completed_post_at={a.completed_post_at}")
            if not remaining and not fb and args.finalize:
                print(f"  Attempting finalize for Block {bi}...")
                out = game_service.finalize_block_if_complete(db, user_id, bi)
                print(f"  Finalize result: {'created' if out else 'no-op'}")
            if fb:
                print(f"  Feedback: top1_pre={fb.top1_accuracy_pre} top1_post={fb.top1_accuracy_post} top3_pre={fb.top3_accuracy_pre} top3_post={fb.top3_accuracy_post}")
            # Detailed per-case summary
            print("  Cases:")
            for a in assigns:
                case = db.get(Case, a.case_id)
                assessments = db.execute(select(Assessment).where(Assessment.assignment_id == a.id)).scalars().all()
                pre = next((x for x in assessments if x.phase == 'PRE'), None)
                post = next((x for x in assessments if x.phase == 'POST'), None)
                def top1(ass):
                    if not ass:
                        return None
                    des = db.execute(select(DiagnosisEntry).where(DiagnosisEntry.assessment_id == ass.id).order_by(DiagnosisEntry.rank.asc())).scalars().all()
                    return des[0].diagnosis_term_id if des else None
                print(f"    - assignment {a.id} case={a.case_id} gt={case.ground_truth_diagnosis_id} pre_top1={top1(pre)} post_top1={top1(post)} pre_correct={getattr(pre,'top1_correct',None)} post_correct={getattr(post,'top1_correct',None)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
