from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        try:
            data = {
                "Age": float(request.form["Age"]),
                "Sex": int(request.form["Sex"]),
                "ChestPain": int(request.form["ChestPain"]),
                "RestBP": float(request.form["RestBP"]),
                "Chol": float(request.form["Chol"]),
                "Fbs": int(request.form["Fbs"]),
                "RestECG": int(request.form["RestECG"]),
                "MaxHR": float(request.form["MaxHR"]),
                "ExAng": int(request.form["ExAng"]),
                "Oldpeak": float(request.form["Oldpeak"]),
                "Slope": int(request.form["Slope"]),
                "Ca": float(request.form["Ca"]),
                "Thal": int(request.form["Thal"]),
            }

            input_df = pd.DataFrame([data])

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            if prediction == 1:
                result = "Heart Disease Detected"
                confidence = round(float(probabilities[1]) * 100, 2)
            else:
                result = "No Heart Disease"
                confidence = round(float(probabilities[0]) * 100, 2)

            print("Prediction:", prediction)
            print("Probabilities:", probabilities)
            print("Confidence:", confidence)

        except Exception as e:
            result = f"Error: {e}"
            confidence = None
            print("Error:", e)

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)