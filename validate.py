import mlflow
import sys

print("Validando calidad del modelo...")

try:
    with open("run_info.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("Error: No se encontró run_info.txt")
    sys.exit(1)

run = mlflow.get_run(run_id)
mse = run.data.metrics.get("mse")

UMBRAL_MSE = 3000

if mse < UMBRAL_MSE:
    print(f"✅ APROBADO: MSE {mse:.2f} es menor al umbral de {UMBRAL_MSE}")
    sys.exit(0)
else:
    print(f"❌ RECHAZADO: MSE {mse:.2f} supera el umbral de {UMBRAL_MSE}")
    sys.exit(1)