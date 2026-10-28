CROP YIELD PREDICTION USING MACHINE LEARNING
============================================

WHAT IS INCLUDED
- app.py: Flask web application
- train_model.py: trains and compares ML models
- templates/index.html: website page
- static/style.css: website design
- data/: place your crop-yield CSV here
- models/: trained model is saved here

STEP 1 — INSTALL PYTHON
Install Python 3.11 or 3.12 if possible. During installation, tick "Add Python to PATH".
If Python is already installed and `python --version` works, you can try that installation first.

STEP 2 — OPEN THE PROJECT FOLDER
Extract the ZIP file. Open the extracted `crop_yield_prediction_project` folder.
In File Explorer, click the address bar, type `cmd`, and press Enter.
A terminal should open inside the project folder.

STEP 3 — CREATE A VIRTUAL ENVIRONMENT (recommended)
Windows:
    python -m venv venv
    venv\Scripts\activate

If activation is blocked in PowerShell, open Command Prompt using the method above and run the commands there.

STEP 4 — INSTALL PACKAGES
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

STEP 5 — ADD THE DATASET
Get a crop yield CSV whose header contains these exact columns:
    Area
    Item
    Year
    hg/ha_yield
    average_rain_fall_mm_per_year
    pesticides_tonnes
    avg_temp

Save or rename the CSV to:
    crop_yield.csv

Put it inside the project's `data` folder:
    crop_yield_prediction_project\data\crop_yield.csv

Do not put the CSV in the models or templates folder.

STEP 6 — TRAIN THE MODEL
In the terminal, run:
    python train_model.py

Wait until it compares the models and prints the best model. It should create:
    models\crop_yield_model.joblib

STEP 7 — START THE WEBSITE
Run:
    python app.py

Open Chrome and go to:
    http://127.0.0.1:5000

Keep the terminal open while using the website. To stop the server, press Ctrl+C in the terminal.

TROUBLESHOOTING
- "Dataset not found": check that the file is exactly data\crop_yield.csv.
- "CSV does not have expected columns": open the CSV and check the first row/header names.
- "Model is not trained yet": run `python train_model.py` successfully before starting the site.
- If `pip` gives a launcher/path error, use `python -m pip ...`.
- If port 5000 is busy, close the other Flask app first.

IMPORTANT PROJECT NOTES
- The model predicts yield in the dataset's unit (hg/ha).
- Accuracy and evaluation metrics depend on your dataset and split; do not claim the paper's reported score unless you reproduce it.
- This is an educational prediction tool, not an agronomy guarantee.
