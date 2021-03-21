# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras import Model

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# Seed everything

def seed_everything(seed=4):
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything()

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
gc.collect()

# Datasets preprocessing
datasets = body, body_symptom, disease, disease_body_symptom, doc_spec, \
           doctor, doctor_diseases, hackathon_order, specialty, symptom

from functools import partial


def select(column, x):
    try:
        int(x[column])
        return True
    except:
        return False


def non_int_to_common(column, df, x):
    try:
        return int(x[column])
    except:
        return df[column].value_counts().index[0]


df = hackathon_order[['comment', 'doctor_id']]

df = df[df.apply(partial(select, 'doctor_id'), axis=1)]

df['doctor_id'] = df['doctor_id'].astype(np.int32)

df = df.merge(right=doc_spec[['doctor_id', 'specialty_id']], how="left", on="doctor_id")
df = df[df.apply(partial(select, 'specialty_id'), axis=1)]
df['specialty_id'] = df['specialty_id'].astype(np.int32)

filt = doc_spec['specialty_id'] == 75
doc_ids = doc_spec.loc[filt, 'id']
filt_doc = doc_spec['doctor_id'] == doc_ids.iloc[1]
specialty[specialty.id == doc_spec.loc[filt_doc, 'specialty_id'].to_numpy()[0]]

# Deal with NaNs and duplicates
df = df.dropna()
df = df.drop_duplicates(subset='comment').reset_index(drop=True)

# Show all columns:
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', None)
pd.set_option('max_colwidth', None)

df[df.comment.duplicated()]
df[~df.comment.duplicated()]


# Comment preprocessing
def preprocess(raw_text):
    # keep only words
    letters_only_text = re.sub("[^a-zA-ZА-Яа-я]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("russian"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # stemmed words
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in meaningful_words]

    # join the cleaned words in a list
    cleaned_word_list = " ".join(stemmed_words)

    return cleaned_word_list


df.comment = df.comment.apply(lambda line: preprocess(line))

# Preprocess targets
y_orig_arr = df['specialty_id'].to_numpy()
y_prep_arr = pd.Series(pd.Categorical(df['specialty_id'])).cat.codes.to_numpy()
prep_orig_map = dict(set(list(zip(y_prep_arr, y_orig_arr))))
prep_orig_map = dict(sorted(prep_orig_map.items()))


def y_to_original(y_prep):
    y_orig = []
    for y_prepi in y_prep:
        y_orig.append(prep_orig_map.get(y_prepi, -1))
    return np.array(y_orig)


import pickle

with open('prep_orig_map', 'wb') as f:
    pickle.dump(prep_orig_map, f)

df['specialty_id'] = y_prep_arr
df = df[df['specialty_id'].apply(lambda x: (df['specialty_id'] == x).sum() >= 2)]

from tensorflow.keras.utils import to_categorical

y = df['specialty_id']
y = to_categorical(y)
# X
X = df['comment']

# Train test split
from sklearn.model_selection import train_test_split

X_comment_train, X_comment_test, y_train, y_test = train_test_split(X.to_numpy(), y,
                                                                    test_size=0.2, shuffle=True, stratify=y)

del X
del y
del df
for dataset in datasets:
    del dataset
gc.collect()

# Preprocess comments

stops = stopwords.words('russian')
tfidf = TfidfVectorizer(max_features=10000, stop_words=stops)
X_comment_train = tfidf.fit_transform(X_comment_train).toarray().astype(np.float32)
X_comment_test = tfidf.transform(X_comment_test).toarray().astype(np.float32)

with open('tfidf', 'wb') as f:
    pickle.dump(tfidf, f)

gc.collect()


# NN
def build_model():
    inp = L.Input(shape=X_comment_train[0].shape)
    hid = L.Dense(512, activation='relu')(inp)
    hid = L.Dense(1024, activation='relu')(hid)
    out = L.Dense(len(y_train[0]), activation='softmax')(hid)
    model = Model(inp, out)

    model.compile('adam', 'categorical_crossentropy', 'acc')

    return model


# Validation
seed_everything()
model = build_model()
model.fit(X_comment_train, y_train, batch_size=128, epochs=2, validation_split=0.2)

y_pred = model.predict(X_comment_test)

top4correct = 0
top3correct = 0
top2correct = 0
top1correct = 0
for y_predi, y_testi in zip(y_pred, y_test):
    y_predi = y_predi.copy()
    top4idx = []
    for i in range(4):
        top4idx.append(np.argmax(y_predi))
        y_predi[top4idx[-1]] = 0

    if np.argmax(y_testi) in top4idx[:1]:
        top1correct += 1

    if np.argmax(y_testi) in top4idx[:2]:
        top2correct += 1

    if np.argmax(y_testi) in top4idx[:3]:
        top3correct += 1

    if np.argmax(y_testi) in top4idx[:4]:
        top4correct += 1

top1correct / len(y_test)
top2correct / len(y_test)
top3correct / len(y_test)
top4correct / len(y_test)


def find_doctor_by_specialties(specialties):
    orig_specialties = y_to_original(specialties)
    isFirst = True

    for orig_specialty in orig_specialties:
        temp_filt = doc_spec['specialty_id'] == orig_specialty
        if isFirst:
            filt_doc_ids = list(set(doc_spec.loc[temp_filt, 'doctor_id'].to_list()))
            isFirst = False
            print("There are initially", len(filt_doc_ids), "doctors found.")
        else:
            new_filt_doc_ids = list(set(doc_spec.loc[temp_filt, 'doctor_id'].to_list()))
            print("There are new", len(new_filt_doc_ids), "doctors found.")
            temp_filt_doc_ids = []
            for existing_doc_id in filt_doc_ids:
                if existing_doc_id in new_filt_doc_ids:
                    temp_filt_doc_ids.append(existing_doc_id)
            filt_doc_ids = temp_filt_doc_ids
            print("There are", len(filt_doc_ids), "doctors left.")

    return filt_doc_ids  # Doc ids


def docs_by_id(doc_ids):
    return [doctor[doctor['id'] == doc_id] for doc_id in doc_ids]


find_doctor_by_specialties(top4idx[:3])

# Testing
model = build_model()
model.fit(X_comment_train, y_train, batch_size=128, epochs=2)

y_pred = np.argmax(model.predict(X_comment_test), axis=-1)

print(f1_score(np.argmax(y_test, axis=-1), y_pred, average='weighted'))
