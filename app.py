
from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.joblib"
DATA_PATH = BASE_DIR / "data" / "crop_yield.csv"

# Load the model once when the application starts, not on every request.
model = None
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("Crop yield model loaded successfully.")
    else:
        print(f"Model file not found: {MODEL_PATH}")
except Exception as exc:
    print(f"Model loading failed: {exc}")

# Load dropdown options once instead of reading the CSV on every request.
try:
    if DATA_PATH.exists():
        df_options = pd.read_csv(DATA_PATH, usecols=["Area", "Item"])
        areas = sorted(df_options["Area"].dropna().astype(str).unique().tolist())
        crops = sorted(df_options["Item"].dropna().astype(str).unique().tolist())
        del df_options
    else:
        raise FileNotFoundError("Dataset not found")
except Exception as exc:
    print(f"Could not load dropdown options: {exc}")
    areas = ["India", "United States", "Brazil", "Australia"]
    crops = ["Rice", "Wheat", "Maize", "Potatoes", "Soybeans"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            if model is None:
                raise ValueError(
                    "The trained model could not be loaded. "
                    "Please check the deployment logs."
                )

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

    return render_template(
        "index.html",
        areas=areas,
        crops=crops,
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=False)
