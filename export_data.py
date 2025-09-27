import asyncio
import csv
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy import select

# Adjust sys.path to allow imports from the app package
# This assumes the script is in the 'backend' directory, and 'backend' is the project root for 'app'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from app.models.models import Base, Case, Image, Assessment, Diagnosis, DiagnosisTerm, ManagementPlan, ManagementStrategy
    from app.auth.models import User  # Assuming User model is in app.auth.models
    from app.core.config import settings
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please ensure this script is run from the 'backend' directory or that PYTHONPATH is set up correctly.")
    sys.exit(1)

# Database setup
engine = create_async_engine(settings.SQLALCHEMY_ASYNC_DATABASE_URI)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# CSV Headers
CSV_HEADERS = [
    "Ground truth Filename", "Case Image", "Responder id",
    "Preferred diagnosis", "Preferred diagnosis confidence (0-4)",
    "Differential diagnosis", "Secondary differential diagnosis",
    "Recommended management", "Recommended management specified / other",
    "Recommended management confidence (0-4)",
    "Change diagnosis after AI",
    "Preferred diagnosis after AI", "Preferred diagnosis confidence after AI (0-4)",
    "Differential diagnosis after AI", "Secondary differential diagnosis after AI",
    "Change management after AI",
    "Recommended management after AI",
    "Recommended management specified / other after AI", # Added new header
    "Recommended management confidence after AI (0-4)",
    "AI usefulness in this case"
]

def get_diagnosis_by_rank(diagnoses_list, rank_num):
    for diag in diagnoses_list:
        if diag.rank == rank_num:
            return diag.diagnosis_term.name if diag.diagnosis_term else "don't know"
    return "don't know"

async def get_data_for_export(db: AsyncSession):
    rows = []

    stmt_cases = (
        select(Case)
        .options(
            selectinload(Case.ground_truth_diagnosis),
            selectinload(Case.images),
            selectinload(Case.assessments)
            .selectinload(Assessment.user),
            selectinload(Case.assessments)
            .selectinload(Assessment.diagnoses)
            .selectinload(Diagnosis.diagnosis_term),
            selectinload(Case.assessments)
            .selectinload(Assessment.management_plan)
            .selectinload(ManagementPlan.strategy)
        )
    )
    result_cases = await db.execute(stmt_cases)
    cases = result_cases.scalars().unique().all()

    for case_obj in cases:
        if not case_obj.images:
            continue

        first_image = case_obj.images[0]
        image_filename = first_image.image_url.split('/')[-1]
        image_filename_stem, image_filename_ext = os.path.splitext(image_filename)

        ground_truth_diagnosis_name = case_obj.ground_truth_diagnosis.name if case_obj.ground_truth_diagnosis else "UnknownDiagnosis"
        
        ground_truth_filename_val = f"{image_filename_stem}_{ground_truth_diagnosis_name}{image_filename_ext}"
        study_case_id_val = image_filename

        assessments_by_user = {}
        for assessment_obj in case_obj.assessments:
            if assessment_obj.user_id not in assessments_by_user:
                assessments_by_user[assessment_obj.user_id] = {"pre": None, "post": None, "user": assessment_obj.user}
            if assessment_obj.is_post_ai:
                assessments_by_user[assessment_obj.user_id]["post"] = assessment_obj
            else:
                assessments_by_user[assessment_obj.user_id]["pre"] = assessment_obj
        
        for user_id, assessment_data in assessments_by_user.items():
            pre_assessment = assessment_data.get("pre")
            post_assessment = assessment_data.get("post")

            # We need at least a pre-assessment to make a meaningful row.
            # If post-assessment is missing, post-AI fields will be "don't know" or "N/A".
            if not pre_assessment:
                continue # Skip if no pre-assessment data for this user/case

            row_data = {header: "don't know" for header in CSV_HEADERS} # Initialize with "don't know"
            row_data["Ground truth Filename"] = ground_truth_filename_val
            row_data["Study case id"] = study_case_id_val
            row_data["Responder id"] = user_id

            # Pre-AI Assessment Data
            row_data["Preferred diagnosis"] = get_diagnosis_by_rank(pre_assessment.diagnoses, 1)
            row_data["Preferred diagnosis confidence (0-4)"] = pre_assessment.confidence_level_top1 if pre_assessment.confidence_level_top1 is not None else "don't know"
            row_data["Differential diagnosis"] = get_diagnosis_by_rank(pre_assessment.diagnoses, 2)
            row_data["Secondary differential diagnosis"] = get_diagnosis_by_rank(pre_assessment.diagnoses, 3)
            if pre_assessment.management_plan:
                row_data["Recommended management"] = pre_assessment.management_plan.strategy.name if pre_assessment.management_plan.strategy else "don't know"
                row_data["Recommended management specified / other"] = pre_assessment.management_plan.free_text if pre_assessment.management_plan.free_text else "don't know"
            row_data["Recommended management confidence (0-4)"] = pre_assessment.management_confidence if pre_assessment.management_confidence is not None else "don't know"

            # Post-AI Assessment Data (if available)
            if post_assessment:
                row_data["Preferred diagnosis after AI"] = get_diagnosis_by_rank(post_assessment.diagnoses, 1)
                row_data["Preferred diagnosis confidence after AI (0-4)"] = post_assessment.confidence_level_top1 if post_assessment.confidence_level_top1 is not None else "don't know"
                row_data["Differential diagnosis after AI"] = get_diagnosis_by_rank(post_assessment.diagnoses, 2)
                row_data["Secondary differential diagnosis after AI"] = get_diagnosis_by_rank(post_assessment.diagnoses, 3)
                if post_assessment.management_plan:
                    row_data["Recommended management after AI"] = post_assessment.management_plan.strategy.name if post_assessment.management_plan.strategy else "don't know"
                    # Populate the new field for post-AI specified management
                    row_data["Recommended management specified / other after AI"] = post_assessment.management_plan.free_text if post_assessment.management_plan.free_text else "don't know"
                else: # Ensure the new field is "don't know" if no post_assessment.management_plan
                    row_data["Recommended management after AI"] = "don't know"
                    row_data["Recommended management specified / other after AI"] = "don't know"

                row_data["Recommended management confidence after AI (0-4)"] = post_assessment.management_confidence if post_assessment.management_confidence is not None else "don't know"
                row_data["AI usefulness in this case"] = post_assessment.ai_usefulness if post_assessment.ai_usefulness else "don't know"

                # Calculate Change diagnosis after AI
                pre_diag_set = {(d.diagnosis_id, d.rank) for d in pre_assessment.diagnoses}
                post_diag_set = {(d.diagnosis_id, d.rank) for d in post_assessment.diagnoses}
                row_data["Change diagnosis after AI"] = "yes" if pre_diag_set != post_diag_set else "no"

                # Calculate Change management after AI
                pre_mgmt_strategy_id = pre_assessment.management_plan.strategy_id if pre_assessment.management_plan else None
                pre_mgmt_free_text = pre_assessment.management_plan.free_text if pre_assessment.management_plan else None # Assuming this is part of "change"
                
                post_mgmt_strategy_id = post_assessment.management_plan.strategy_id if post_assessment.management_plan else None
                post_mgmt_free_text = post_assessment.management_plan.free_text if post_assessment.management_plan else None # Assuming this is part of "change"

                management_changed = False
                if pre_mgmt_strategy_id != post_mgmt_strategy_id:
                    management_changed = True
                # Consider free text change as part of management change if desired.
                # For now, let's align with typical interpretation: strategy change is primary.
                # If free_text change also counts:
                # if pre_mgmt_free_text != post_mgmt_free_text:
                #     management_changed = True
                row_data["Change management after AI"] = "yes" if management_changed else "no"
            else:
                # If no post-assessment, "change" fields are 'no' or 'N/A'
                row_data["Change diagnosis after AI"] = "N/A (no post-AI data)"
                row_data["Change management after AI"] = "N/A (no post-AI data)"
                # Other post-AI fields remain "don't know" as initialized

            rows.append([row_data[header] for header in CSV_HEADERS])
    return rows

async def main_export():
    async with AsyncSessionLocal() as session:
        data_rows = await get_data_for_export(session)

    output_filename = "exported_data.csv"
    # Ensure the script writes to the backend directory
    output_filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_filename)

    with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADERS)
        writer.writerows(data_rows)
    print(f"Data exported to {output_filepath}")

if __name__ == "__main__":
    asyncio.run(main_export())
