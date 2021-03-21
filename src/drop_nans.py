import numpy as np
import pandas as pd

# Importing data
body = pd.read_csv("../data/body.csv", error_bad_lines=False)
body_symptom = pd.read_csv("../data/body_symptom.csv", error_bad_lines=False)
disease = pd.read_csv("../data/disease.csv", error_bad_lines=False)
disease_body_symptom = pd.read_csv("../data/disease_body_symptom.csv", error_bad_lines=False)
doc_spec = pd.read_csv("../data/doc_spec.csv", error_bad_lines=False)
doctor = pd.read_csv("../data/doctor.csv", error_bad_lines=False)
doctor_diseases = pd.read_csv("../data/doctor_diseases.csv", error_bad_lines=False)
hackathon_order = pd.read_csv("../data/hackathon_order.csv", error_bad_lines=False)
specialty = pd.read_csv("../data/specialty.csv", error_bad_lines=False)
symptom = pd.read_csv("../data/symptom.csv", error_bad_lines=False)

# Datasets preprocessing
datasets = body, body_symptom, disease, disease_body_symptom, doc_spec, \
           doctor, doctor_diseases, hackathon_order, specialty, symptom

for dataset in datasets:  # Dropping columns with a lot of NaNs
    for column in dataset.isna().items():
        if np.sum(column[1:], axis=1)[0] > len(dataset) * 0.90:  # Threshold percentage of NaNs
            dataset.drop(columns=column[0], inplace=True)

body.to_csv('../data/body.csv', index=False)
body_symptom.to_csv('../data/body_symptom.csv', index=False)
disease.to_csv('../data/disease.csv', index=False)
disease_body_symptom.to_csv('../data/disease_body_symptom.csv', index=False)
doc_spec.to_csv('../data/doc_spec.csv', index=False)
doctor.to_csv('../data/doctor.csv', index=False)
doctor_diseases.to_csv('../data/doctor_diseases.csv', index=False)
hackathon_order.to_csv("../data/hackathon_order.csv", index=False)
specialty = pd.read_csv("../data/specialty.csv", error_bad_lines=False)
symptom.to_csv('../data/symptom.csv', index=False)
