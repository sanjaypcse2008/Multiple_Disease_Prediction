from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Symptoms used by the model
symptoms = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "SoreThroat",
    "BodyPain"
]


@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms)


@app.route("/predict", methods=["POST"])
def predict():

    # Get selected symptoms from the form
    selected_symptoms = request.form.getlist("symptoms")

    # Convert symptoms into 0/1 values
    input_data = []

    for symptom in symptoms:
        if symptom in selected_symptoms:
            input_data.append(1)
        else:
            input_data.append(0)

    # Make prediction
    prediction = model.predict([input_data])[0]

    return render_template(
        "index.html",
        symptoms=symptoms,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)