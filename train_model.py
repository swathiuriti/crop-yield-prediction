
    # 2. Chronological train/validation/test split
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
    validation_years = years[train_end:len(years) - test_count]
    test_years = years[len(years) - test_count:]

    train_df = df[df["Year"].isin(train_years)].copy()
    validation_df = df[df["Year"].isin(validation_years)].copy()
    test_df = df[df["Year"].isin(test_years)].copy()

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = validation_df[FEATURES], validation_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print("\nDATASET SUMMARY")
    print("-" * 65)
    print("Usable records:", len(df))
    print("Training years:", train_years[0], "to", train_years[-1])
    print("Validation years:", validation_years[0], "to", validation_years[-1])
    print("Final test years:", test_years[0], "to", test_years[-1])
    print("Train / validation / test rows:", len(train_df), "/", len(validation_df), "/", len(test_df))

    # 3. Fit/evaluate one candidate at a time to avoid retaining every fitted model.
    validation_results = []
    best_name = None
    best_mae = float("inf")
    best_estimator = None

    print("\nVALIDATION RESULTS")
    print("-" * 75)
    print(f"{'Model':22s} {'R2':>10s} {'MAE':>14s} {'RMSE':>14s}")

    candidates = build_candidates()
    while candidates:
        name = next(iter(candidates))
        estimator = candidates.pop(name)  # Remove it from the dict before fitting.
        print(f"Training {name}...")
        estimator.fit(X_train, y_train)
        metrics = evaluate(y_val, estimator.predict(X_val))
        print(f"{name:22s} {metrics['R2']:10.4f} {metrics['MAE']:14.2f} {metrics['RMSE']:14.2f}")
        validation_results.append({"Model": name, **metrics})

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_name = name

        del estimator
        gc.collect()

    print("\nSelected model based on validation MAE:", best_name)

    # 4. Refit only the selected model on train + validation; keep test years unseen.
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_model = build_candidates()[best_name]
    final_model.fit(combined_df[FEATURES], combined_df[TARGET])

    test_metrics = evaluate(y_test, final_model.predict(X_test))
    print("\nFINAL TEST RESULTS — UNSEEN LATER YEARS")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    baseline = create_pipeline(DummyRegressor(strategy="median"))
    baseline.fit(combined_df[FEATURES], combined_df[TARGET])
    baseline_metrics = evaluate(y_test, baseline.predict(X_test))
    print("\nMEDIAN BASELINE TEST RESULTS")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value}")

    improvement = (
        (baseline_metrics["MAE"] - test_metrics["MAE"]) / baseline_metrics["MAE"] * 100
        if baseline_metrics["MAE"] != 0 else 0
    )
    print(f"Test MAE improvement over baseline: {improvement:.2f}%")

    # 5. Save model and evaluation outputs. Compression can reduce disk size;
    # the model still needs enough RAM when loaded by the web application.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH, compress=3)

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
        json.dump({
            "selected_model": best_name,
            "test_years": [int(year) for year in test_years],
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_metrics,
            "mae_improvement_percent": round(improvement, 2),
        }, file, indent=4)

    print("\nSAVED FILES")
    print("Trained model:", MODEL_PATH)
    print("Model comparison:", RESULTS_PATH)
    print("Final test metrics:", TEST_METRICS_PATH)
    print("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()