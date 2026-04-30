import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("Diabetes_Linear_Regression")

print("Cargando datos...")
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    print("Entrenando modelo...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"MSE: {mse:.2f}")

    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "modelo_regresion")

    # Guardamos el ID para el Quality Gate
    with open("run_info.txt", "w") as f:
        f.write(run.info.run_id)

print("Entrenamiento finalizado.")