"""Train and evaluate the ClaimWise Linear Regression model."""

from sklearn.linear_model import LinearRegression

from model_training_utils import load_prepared_data, save_model_results


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = save_model_results(
        "Linear Regression",
        "linear_regression.pkl",
        model,
        x_test,
        y_test,
        predictions,
    )
    print("Linear Regression metrics:", metrics)


if __name__ == "__main__":
    main()
