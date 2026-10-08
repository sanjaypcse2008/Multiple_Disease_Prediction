from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

FEATURES = [
    "Age",
    "Gender",
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "SoreThroat",
    "BodyPain",
    "ShortnessOfBreath",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "JointPain",
    "Rash"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        values = []

        # Age
        age = request.form.get("Age")

        if not age:
            return render_template(
                "index.html",
                prediction="Please enter your age."
            )

        values.append(float(age))

        # Gender
        gender = request.form.get("Gender")

        if gender not in ["0", "1"]:
            return render_template(
                "index.html",
                prediction="Please select your gender."
            )

        values.append(float(gender))

        # Symptoms
        symptom_features = FEATURES[2:]

        for symptom in symptom_features:
            value = 1 if request.form.get(symptom) == "1" else 0
            values.append(value)

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Disease: {prediction}"
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction="Something went wrong. Please try again."
        )


if __name__ == "__main__":
    app.run(debug=True)