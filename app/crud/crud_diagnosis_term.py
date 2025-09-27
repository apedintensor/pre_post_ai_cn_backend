# backend/app/crud/crud_diagnosis_term.py
from sqlalchemy.orm import Session
from sqlalchemy import or_, literal_column
from app.models import models
from app.schemas import schemas

def get_diagnosis_term(db: Session, term_id: int):
    return db.query(models.DiagnosisTerm).filter(models.DiagnosisTerm.id == term_id).first()

def get_diagnosis_term_by_name(db: Session, name: str):
    return db.query(models.DiagnosisTerm).filter(models.DiagnosisTerm.name == name).first()

def get_diagnosis_terms(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.DiagnosisTerm).offset(skip).limit(limit).all()

def create_diagnosis_term(db: Session, term: schemas.DiagnosisTermCreate):
    db_term = models.DiagnosisTerm(name=term.name)
    db.add(db_term)
    db.commit()
    db.refresh(db_term)
    return db_term

# Update/Delete as needed

diagnosis_term = {
    "get": get_diagnosis_term,
    "get_by_name": get_diagnosis_term_by_name,
    "get_multi": get_diagnosis_terms,
    "create": create_diagnosis_term,
}


# ------------- Synonyms CRUD -------------

def create_synonym(db: Session, synonym_in: schemas.DiagnosisSynonymCreate):
    syn = models.DiagnosisSynonym(diagnosis_term_id=synonym_in.diagnosis_term_id, synonym=synonym_in.synonym)
    db.add(syn)
    db.commit()
    db.refresh(syn)
    return syn


def list_synonyms(db: Session, term_id: int | None = None):
    q = db.query(models.DiagnosisSynonym)
    if term_id:
        q = q.filter(models.DiagnosisSynonym.diagnosis_term_id == term_id)
    return q.all()


def suggest_terms(db: Session, query: str, limit: int = 10):
    q_norm = query.strip().lower()
    if len(q_norm) < 2:
        return []

    # Match on term names
    term_matches = (
        db.query(
            models.DiagnosisTerm.id,
            models.DiagnosisTerm.name.label("label"),
            models.DiagnosisTerm.name.label("match"),
            literal_column("'name'").label("source")
        )
        .filter(models.DiagnosisTerm.name.ilike(f"%{q_norm}%"))
    )

    # Match on synonyms
    synonym_matches = (
        db.query(
            models.DiagnosisTerm.id,
            models.DiagnosisTerm.name.label("label"),
            models.DiagnosisSynonym.synonym.label("match"),
            literal_column("'synonym'").label("source")
        )
        .join(models.DiagnosisSynonym, models.DiagnosisSynonym.diagnosis_term_id == models.DiagnosisTerm.id)
        .filter(models.DiagnosisSynonym.synonym.ilike(f"%{q_norm}%"))
    )

    rows = term_matches.union_all(synonym_matches).all()

    scored = []
    for r in rows:
        name_low = r.label.lower()
        match_low = r.match.lower()
        starts = match_low.startswith(q_norm)
        in_middle = q_norm in match_low
        score = 0
        if starts:
            score += 200
        elif in_middle:
            score += 80
        if r.source == "synonym":
            score += 20  # slight boost so synonym that startsWith outranks substring name
        score -= abs(len(match_low) - len(q_norm)) * 0.5
        scored.append({
            "id": r.id,
            "label": r.label,
            "match": r.match,
            "source": r.source,
            "score": score,
        })

    # Deduplicate by term id keeping highest score
    best: dict[int, dict] = {}
    for item in scored:
        cur = best.get(item["id"]) 
        if not cur or item["score"] > cur["score"]:
            best[item["id"]] = item

    ordered = sorted(best.values(), key=lambda x: (-x["score"], x["label"]))[:limit]

    # Attach full synonym lists for returned terms (for richer client display / local fuzzy)
    if ordered:
        term_ids = [o["id"] for o in ordered]
        syn_rows = (
            db.query(models.DiagnosisSynonym)
            .filter(models.DiagnosisSynonym.diagnosis_term_id.in_(term_ids))
            .all()
        )
        syn_map: dict[int, list[str]] = {tid: [] for tid in term_ids}
        for sr in syn_rows:
            syn_map.setdefault(sr.diagnosis_term_id, []).append(sr.synonym)
        for o in ordered:
            o["synonyms"] = syn_map.get(o["id"], [])
    return ordered

synonyms = {
    "create": create_synonym,
    "list": list_synonyms,
    "suggest": suggest_terms,
}
