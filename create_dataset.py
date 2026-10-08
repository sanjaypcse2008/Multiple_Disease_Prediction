import pandas as pd
import random

random.seed(42)

diseases = {
    "Flu":         [1,1,1,1,1,1,0,0,0,0,0,0],
    "Common Cold": [0,1,0,0,1,0,0,0,0,0,0,0],
    "Allergy":     [0,1,0,0,1,0,0,0,0,0,0,1],
    "Migraine":    [0,0,1,1,0,0,0,0,1,0,0,0],
    "Dengue":      [1,0,1,1,0,1,0,1,0,0,0,1],
    "COVID-19":    [1,1,1,1,1,1,1,0,0,0,0,0],
    "Malaria":     [1,0,1,1,0,1,0,1,0,0,0,1],
    "Typhoid":     [1,0,1,1,0,1,0,1,1,1,1,0],
    "Asthma":      [0,1,0,0,0,0,1,0,0,0,0,0],
    "Pneumonia":   [1,1,0,1,0,1,1,0,0,0,0,0]
}

columns = [
    "Age", "Gender",
    "Fever", "Cough", "Headache", "Fatigue",
    "SoreThroat", "BodyPain", "ShortnessOfBreath",
    "Nausea", "Vomiting", "Diarrhea",
    "JointPain", "Rash", "Disease"
]

rows = []

for disease, symptoms in diseases.items():

    for _ in range(100):

        age = random.randint(5, 80)
        gender = random.randint(0, 1)

        row = [age, gender]

        for symptom in symptoms:

            value = symptom

            if random.random() < 0.08:
                value = 1 - value

            row.append(value)

        row.append(disease)
        rows.append(row)

df = pd.DataFrame(rows, columns=columns)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("dataset/improved_disease_dataset.csv", index=False)

print("Improved dataset created successfully!")
print("Total records:", len(df))
print("Total diseases:", df["Disease"].nunique())
print("Saved as dataset/improved_disease_dataset.csv")