import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine(
    f"mysql+pymysql://{input('Enter username: ')}:{input('Enter password: ')}@localhost/DOC")

body = pd.read_sql('SELECT * FROM body;', engine)
body_symptom = pd.read_sql('SELECT * FROM body_symptom;', engine)
disease = pd.read_sql('SELECT * FROM disease;', engine)
disease_body_symptom = pd.read_sql('SELECT * FROM disease_body_symptom;', engine)
doc_spec = pd.read_sql('SELECT * FROM doc_spec;', engine)
doctor = pd.read_sql('SELECT * FROM doctor;', engine)
doctor_diseases = pd.read_sql('SELECT * FROM doctor_diseases;', engine)
specialty = pd.read_sql('SELECT * FROM specialty;', engine)

body.to_csv('../data/body.csv', index=False)
body_symptom.to_csv('../data/body_symptom.csv', index=False)
disease.to_csv('../data/disease.csv', index=False)
disease_body_symptom.to_csv('../data/disease_body_symptom.csv', index=False)
doc_spec.to_csv('../data/doc_spec.csv', index=False)
doctor.to_csv('../data/doctor.csv', index=False)
doctor_diseases.to_csv('../data/doctor_diseases.csv', index=False)
specialty.to_csv('../data/specialty.csv', index=False)
