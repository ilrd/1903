from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
from datacleaner import clean
import sys
import os

sys.path.append(os.getcwd())


class Predictor:
    """prepares model to use"""

    def y_to_original(self, y_prep):
        with open('utils/prep_orig_map', 'rb') as f:
            prep_orig_map = pickle.load(f)

        y_orig = []
        for y_prepi in y_prep:
            y_orig.append(prep_orig_map.get(y_prepi, -1))
        return np.array(y_orig)

    def doc_ids_by_specialties(self, specialties):
        doc_spec = pd.read_csv("../data/doc_spec.csv", error_bad_lines=False)
        orig_specialties = self.y_to_original(specialties)
        isFirst = True

        for orig_specialty in orig_specialties:
            temp_filt = doc_spec['specialty_id'] == orig_specialty
            if isFirst:
                filt_doc_ids = list(set(doc_spec.loc[temp_filt, 'doctor_id'].to_list()))
                isFirst = False
            else:
                new_filt_doc_ids = list(set(doc_spec.loc[temp_filt, 'doctor_id'].to_list()))
                temp_filt_doc_ids = []
                for existing_doc_id in filt_doc_ids:
                    if existing_doc_id in new_filt_doc_ids:
                        temp_filt_doc_ids.append(existing_doc_id)
                filt_doc_ids = temp_filt_doc_ids

        return filt_doc_ids  # Doc ids

    def specs_by_doc_ids(self, doc_ids):
        doc_spec = pd.read_csv("../data/doc_spec.csv", error_bad_lines=False)
        temp_spec_ids = [doc_spec.loc[doc_spec['doctor_id'] == doc_id, 'specialty_id'].to_list() for doc_id in doc_ids]
        spec_ids = []
        for spec_id in temp_spec_ids:
            spec_ids += spec_id
        specialty = pd.read_csv("../data/specialty.csv", error_bad_lines=False)
        specs = [specialty.loc[specialty['id'] == spec_id] for spec_id in spec_ids]

        return specs

    def predict(self, comment):
        model = load_model('utils/model3.h5')
        pred = model.predict([clean(comment)])[0]

        top5idx = []
        for i in range(5):
            top5idx.append(np.argmax(pred))
            pred[top5idx[-1]] = 0

        for i in range(1, 6):
            if not self.doc_ids_by_specialties(top5idx[:i]):
                break

        doc_ids = self.doc_ids_by_specialties(top5idx[:i - 1])
        specs = self.specs_by_doc_ids(doc_ids)

        return [spec['name'].to_list()[0] for spec in specs]
