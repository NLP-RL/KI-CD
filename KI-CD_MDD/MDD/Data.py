import pickle

with open('disease_symptom.p','rb') as f:
    disease_symptom = pickle.load(f)



print(disease_symptom)
