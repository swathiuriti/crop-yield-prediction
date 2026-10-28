from flask import Flask, render_template, request
import os
import pandas as pd
import joblib

app = Flask(__name__)
MODEL_PATH = os.path.join("models", "crop_yield_model.joblib")
DATA_PATH = os.path.join("data", "crop_yield.csv")

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def get_options():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            return sorted(df["Area"].dropna().astype(str).unique()), sorted(df["Item"].dropna().astype(str).unique())
        except Exception:
            pass
    return (
        ["India", "United States", "Brazil", "Australia"],
        ["Rice", "Wheat", "Maize", "Potatoes", "Soybeans"]
    )

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    areas, crops = get_options()
    model = load_model()

    if request.method == "POST":
        try:
            if model is None:
                raise ValueError("The model is not trained yet. Add the dataset and run train_model.py first.")

            area = request.form["area"]
            crop = request.form["crop"]
            year = int(request.form["year"])
            rainfall = float(request.form["rainfall"])
            pesticides = float(request.form["pesticides"])
            temperature = float(request.form["temperature"])

            input_df = pd.DataFrame([{
                "Area": area,
                "Item": crop,
                "Year": year,
                "average_rain_fall_mm_per_year": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": temperature
            }])
            predicted_yield = float(model.predict(input_df)[0])
            prediction = {
                "value": f"{predicted_yield:,.2f}",
                "unit": "hg/ha",
                "area": area,
                "crop": crop,
                "year": year
            }
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", areas=areas, crops=crops,
                           prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
