from app.db.session import SessionLocal
from app.models.models import DiagnosisTerm, DiagnosisSynonym
from app.crud import crud_diagnosis_term

db = SessionLocal()
term = db.query(DiagnosisTerm).filter_by(name='Basal Cell Carcinoma').first()
if not term:
	term = DiagnosisTerm(name='Basal Cell Carcinoma')
	db.add(term)
	db.commit(); db.refresh(term)

syn = db.query(DiagnosisSynonym).filter_by(synonym='BCC').first()
if not syn:
	syn = DiagnosisSynonym(diagnosis_term_id=term.id, synonym='BCC')
	db.add(syn); db.commit(); db.refresh(syn)

print('Term ID:', term.id)
print('Synonym ID:', syn.id)
print('Suggest bc ->', crud_diagnosis_term.synonyms['suggest'](db, 'bc'))
print('Suggest basal ->', crud_diagnosis_term.synonyms['suggest'](db, 'basal'))

db.close()
