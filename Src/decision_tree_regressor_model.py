"""Train and evaluate the ClaimWise Decision Tree Regressor."""

from sklearn.tree import DecisionTreeRegressor

from model_training_utils import load_prepared_data, save_model_results


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    model = DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = save_model_results(
        "Decision Tree Regressor",
        "decision_tree_regressor.pkl",
        model,
        x_test,
        y_test,
        predictions,
    )
    print("Decision Tree Regressor metrics:", metrics)


if __name__ == "__main__":
    main()
