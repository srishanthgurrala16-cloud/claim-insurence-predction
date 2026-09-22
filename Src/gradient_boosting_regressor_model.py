"""Train and evaluate the ClaimWise Gradient Boosting Regressor."""

from sklearn.ensemble import GradientBoostingRegressor

from model_training_utils import load_prepared_data, save_model_results


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
        loss="squared_error",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = save_model_results(
        "Gradient Boosting Regressor",
        "gradient_boosting_regressor.pkl",
        model,
        x_test,
        y_test,
        predictions,
    )
    print("Gradient Boosting Regressor metrics:", metrics)


if __name__ == "__main__":
    main()
