#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from homework.src.trainers import select_best_knn

NEIGHBORS = [1, 3, 5, 7, 9, 11, 15, 21]

# entrenar el modelo
result = select_best_knn(NEIGHBORS)
estimator = result.estimator

print()
print(estimator, ":", sep="")

print()
print("Metricas de entrenamiento:")
print(f"  MSE: {result.train_metrics.mse}")
print(f"  MAE: {result.train_metrics.mae}")
print(f"  R2: {result.train_metrics.r2}")

print()
print("Metricas de testing:")
print(f"  MSE: {result.test_metrics.mse}")
print(f"  MAE: {result.test_metrics.mae}")
print(f"  R2: {result.test_metrics.r2}")
