import pandas as pd
import numpy as np


def find_doctor_by_specialties(specialties):
    doc_spec = pd.read_csv("../data/doc_spec.csv", error_bad_lines=False)
    specialty_df = pd.read_csv("../data/specialty.csv", error_bad_lines=False)
    isFirst = True

    for specialty in specialties:
        temp_filt = (specialty_df['name'].str.lower() == specialty.lower()) | (
            specialty_df['name'].str.lower().str.contains(specialty.lower()))
        if isFirst:
            spec_ids = list(set(specialty_df.loc[temp_filt, 'id'].to_list()))
            filt_spec_ids = [False for _ in range(len(doc_spec))]
            for spec_id in spec_ids:
                filt_spec_ids = filt_spec_ids | (doc_spec['specialty_id'] == spec_id)
            filt_doc_ids = list(set(doc_spec.loc[filt_spec_ids, 'doctor_id'].to_list()))
            isFirst = False
        else:
            new_spec_ids = list(set(specialty_df.loc[temp_filt, 'id'].to_list()))
            new_filt_spec_ids = [False for _ in range(len(doc_spec))]
            for spec_id in new_spec_ids:
                new_filt_spec_ids = new_filt_spec_ids | (doc_spec['specialty_id'] == spec_id)
            new_filt_doc_ids = list(set(doc_spec.loc[new_filt_spec_ids, 'doctor_id'].to_list()))

            temp_filt_doc_ids = []
            for existing_doc_id in filt_doc_ids:
                if existing_doc_id in new_filt_doc_ids:
                    temp_filt_doc_ids.append(existing_doc_id)
            filt_doc_ids = temp_filt_doc_ids

    return filt_doc_ids  # Doc ids


def specs_by_doc_ids(doc_ids):
    doc_spec = pd.read_csv("../data/doc_spec.csv", error_bad_lines=False)
    temp_spec_ids = [doc_spec.loc[doc_spec['doctor_id'] == doc_id, 'specialty_id'].to_list() for doc_id in doc_ids]
    spec_ids = []
    for spec_id in temp_spec_ids:
        spec_ids += spec_id
    specialty = pd.read_csv("../data/specialty.csv", error_bad_lines=False)
    specs = [specialty.loc[specialty['id'] == spec_id] for spec_id in spec_ids]

    return specs


def search(*specialties):
    if type(specialties[0]) == list:
        specialties = specialties[0]
    doc_ids = find_doctor_by_specialties(specialties)
    specs = specs_by_doc_ids(doc_ids)

    return [[spec['name'].to_list()[0], doc_id] for spec, doc_id in list(zip(specs, doc_ids))[:5]]
