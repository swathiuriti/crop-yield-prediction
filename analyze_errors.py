
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/crop_yield.csv"
MODEL_PATH = "models/crop_yield_model.joblib"

FEATURES = [
    "Area", "Item", "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes", "avg_temp"
]
TARGET = "hg/ha_yield"


def main():
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("Dataset or trained model not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    df = df[FEATURES + [TARGET]].copy()

    numeric = [
        "Year", "average_rain_fall_mm_per_year",
        "pesticides_tonnes", "avg_temp", TARGET
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric + ["Area", "Item"])
    df["Area"] = df["Area"].astype(str).str.strip()
    df["Item"] = df["Item"].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    # Recreate the same chronological test split used in training.
    years = sorted(df["Year"].unique())
    test_count = max(1, int(__import__("math").ceil(len(years) * 0.15)))
    validation_count = max(1, int(__import__("math").ceil(len(years) * 0.15)))
    train_end = len(years) - test_count - validation_count

    test_years = years[len(years) - test_count:]
    test = df[df["Year"].isin(test_years)].copy()

    model = joblib.load(MODEL_PATH)
    test["Actual"] = test[TARGET]
    test["Predicted"] = model.predict(test[FEATURES])
    test["Error"] = test["Predicted"] - test["Actual"]
    test["Absolute_Error"] = test["Error"].abs()

    print("\nTEST SET OVERALL")
    print("Test years:", test_years)
    print("Rows:", len(test))
    print("R2:", round(r2_score(test["Actual"], test["Predicted"]), 4))
    print("MAE:", round(mean_absolute_error(
        test["Actual"], test["Predicted"]), 2))
    print("RMSE:", round(mean_squared_error(
        test["Actual"], test["Predicted"]) ** 0.5, 2))

    print("\nERROR BY YEAR")
    by_year = test.groupby("Year").agg(
        Records=("Actual", "size"),
        MAE=("Absolute_Error", "mean"),
        Mean_Error=("Error", "mean"),
    ).round(2)
    print(by_year.to_string())

    print("\nERROR BY CROP — 10 WORST CROPS")
    by_crop = test.groupby("Item").agg(
        Records=("Actual", "size"),
        MAE=("Absolute_Error", "mean"),
        Mean_Error=("Error", "mean"),
    ).round(2)
    print(by_crop.sort_values("MAE", ascending=False).head(10).to_string())

    print("\nERROR BY REGION — 10 WORST REGIONS")
    by_area = test.groupby("Area").agg(
        Records=("Actual", "size"),
        MAE=("Absolute_Error", "mean"),
        Mean_Error=("Error", "mean"),
    ).round(2)
    print(by_area.sort_values("MAE", ascending=False).head(10).to_string())

    print("\n20 LARGEST INDIVIDUAL ERRORS")
    columns = [
        "Area", "Item", "Year", "Actual",
        "Predicted", "Error", "Absolute_Error"
    ]
    print(
        test.sort_values("Absolute_Error", ascending=False)
        [columns].head(20).round(2).to_string(index=False)
    )

    test.to_csv("test_predictions.csv", index=False)
    by_year.to_csv("error_by_year.csv")
    by_crop.to_csv("error_by_crop.csv")
    by_area.to_csv("error_by_region.csv")

    print("\nSaved test_predictions.csv")
    print("Saved error_by_year.csv")
    print("Saved error_by_crop.csv")
    print("Saved error_by_region.csv")


if __name__ == "__main__":
    main()
