import os
import math
import json
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_PATH = os.path.join("data", "crop_yield.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_model.joblib")

RESULTS_PATH = "model_evaluation.csv"
TEST_METRICS_PATH = "final_test_metrics.json"

TARGET = "hg/ha_yield"

FEATURES = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

NUMERIC_FEATURES = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

CATEGORICAL_FEATURES = ["Area", "Item"]


# ---------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------

def create_pipeline(estimator):
    """Build preprocessing and estimator into one pipeline."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------

def evaluate(y_true, predictions):
    """Calculate regression evaluation metrics."""

    return {
        "R2": round(float(r2_score(y_true, predictions)), 4),
        "MAE": round(float(mean_absolute_error(y_true, predictions)), 2),
        "RMSE": round(
            float(mean_squared_error(y_true, predictions) ** 0.5), 2
        ),
    }


# ---------------------------------------------------------
# MODEL CANDIDATES
# ---------------------------------------------------------

def build_candidates():
    """Build candidate models using smaller tree ensembles."""

    small_tree_settings = {
        "n_estimators": 40,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": 1,
    }

    return {
        "Median Baseline": create_pipeline(
            DummyRegressor(strategy="median")
        ),

        "Linear Regression": create_pipeline(
            LinearRegression()
        ),

        "KNN": create_pipeline(
            KNeighborsRegressor(
                n_neighbors=5,
                weights="distance",
            )
        ),

        "Gradient Boosting": create_pipeline(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        ),

        "Random Forest": create_pipeline(
            RandomForestRegressor(**small_tree_settings)
        ),

        "Extra Trees": create_pipeline(
            ExtraTreesRegressor(**small_tree_settings)
        ),

        "Extra Trees Log": TransformedTargetRegressor(
            regressor=create_pipeline(
                ExtraTreesRegressor(**small_tree_settings)
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


# ---------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------

def main():

    # 1. Check and load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found: {DATA_PATH}")
        print("Place crop_yield.csv inside the data folder.")
        return

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Remove unnecessary CSV index columns
    df = df.loc[
        :,
        ~df.columns.str.startswith("Unnamed:")
    ]

    # Check required columns
    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        print("Missing required CSV columns:", missing)
        print("Check the first row of your dataset.")
        return

    df = df[FEATURES + [TARGET]].copy()

    # 2. Clean numeric values
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET])

    # Log transformation requires nonnegative target values
    if (df[TARGET] < 0).any():
        negative_count = int((df[TARGET] < 0).sum())

        print(
            f"Found {negative_count} negative target values. "
            "The Extra Trees Log model requires nonnegative targets."
        )
        print("Check the dataset before training.")
        return

    # Remove duplicate records
    df = df.drop_duplicates()

    # 3. Clean categorical values
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    df = df.dropna(subset=CATEGORICAL_FEATURES)
    df = df.reset_index(drop=True)

    if len(df) < 100:
        print("Too few usable records. Check your dataset.")
        return

    # 4. Chronological train/validation/test split
    years = sorted(df["Year"].unique())

    if len(years) < 5:
        print("At least 5 distinct years are needed for this split.")
        return

    test_count = max(1, math.ceil(len(years) * 0.15))
    validation_count = max(1, math.ceil(len(years) * 0.15))

    train_end = len(years) - test_count - validation_count

    if train_end < 2:
        print("Not enough distinct years for train/validation/test.")
        return

    train_years = years[:train_end]

    validation_years = years[
        train_end:len(years) - test_count
    ]

    test_years = years[len(years) - test_count:]

    train_df = df[df["Year"].isin(train_years)].copy()

    validation_df = df[
        df["Year"].isin(validation_years)
    ].copy()

    test_df = df[df["Year"].isin(test_years)].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nDATASET SUMMARY")
    print("-" * 65)
    print("Usable records:", len(df))
    print("Training years:", train_years[0], "to", train_years[-1])
    print(
        "Validation years:",
        validation_years[0],
        "to",
        validation_years[-1],
    )
    print("Final test years:", test_years[0], "to", test_years[-1])

    print(
        "Train / validation / test rows:",
        len(train_df),
        "/",
        len(validation_df),
        "/",
        len(test_df),
    )

    # 5. Train and evaluate one candidate at a time
    validation_results = []
    best_name = None
    best_mae = float("inf")

    candidates = build_candidates()

    print("\nVALIDATION RESULTS")
    print("-" * 75)
    print(
        f"{'Model':22s} {'R2':>10s} "
        f"{'MAE':>14s} {'RMSE':>14s}"
    )

    while candidates:
        name = next(iter(candidates))
        estimator = candidates.pop(name)

        print(f"Training {name}...")

        estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_val)
        metrics = evaluate(y_val, predictions)

        print(
            f"{name:22s} "
            f"{metrics['R2']:10.4f} "
            f"{metrics['MAE']:14.2f} "
            f"{metrics['RMSE']:14.2f}"
        )

        validation_results.append({
            "Model": name,
            **metrics,
        })

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_name = name

        del estimator, predictions
        gc.collect()

    print("\nSelected model based on validation MAE:", best_name)

    # 6. Refit selected model on training + validation data
    # Final test years remain unseen during model selection
    combined_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    final_model = build_candidates()[best_name]

    final_model.fit(
        combined_df[FEATURES],
        combined_df[TARGET],
    )

    # 7. Evaluate on final test set
    test_predictions = final_model.predict(X_test)
    test_metrics = evaluate(y_test, test_predictions)

    print("\nFINAL TEST RESULTS — UNSEEN LATER YEARS")
    print("-" * 55)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # 8. Compare against median baseline
    baseline = create_pipeline(
        DummyRegressor(strategy="median")
    )

    baseline.fit(
        combined_df[FEATURES],
        combined_df[TARGET],
    )

    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = evaluate(y_test, baseline_predictions)

    print("\nMEDIAN BASELINE TEST RESULTS")
    print("-" * 40)

    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value}")

    if baseline_metrics["MAE"] != 0:
        improvement = (
            (
                baseline_metrics["MAE"] - test_metrics["MAE"]
            )
            / baseline_metrics["MAE"] * 100
        )
    else:
        improvement = 0.0

    print("\nTest MAE improvement over baseline:")
    print(f"{improvement:.2f}%")

    # 9. Save final model with compression
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        final_model,
        MODEL_PATH,
        compress=3,
    )

    # 10. Save validation and test results
    results = pd.DataFrame(validation_results)

    results["Final_Test_R2"] = None
    results["Final_Test_MAE"] = None
    results["Final_Test_RMSE"] = None

    selected_row = results["Model"] == best_name

    results.loc[selected_row, "Final_Test_R2"] = test_metrics["R2"]
    results.loc[selected_row, "Final_Test_MAE"] = test_metrics["MAE"]
    results.loc[selected_row, "Final_Test_RMSE"] = test_metrics["RMSE"]

    results.to_csv(RESULTS_PATH, index=False)

    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_model": best_name,
                "test_years": [
                    int(year) for year in test_years
                ],
                "test_metrics": test_metrics,
                "baseline_test_metrics": baseline_metrics,
                "mae_improvement_percent": round(
                    improvement,
                    2,
                ),
            },
            file,
            indent=4,
        )

    print("\nSAVED FILES")
    print("-" * 40)
    print("Trained model:", MODEL_PATH)
    print("Model comparison:", RESULTS_PATH)
    print("Final test metrics:", TEST_METRICS_PATH)
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()