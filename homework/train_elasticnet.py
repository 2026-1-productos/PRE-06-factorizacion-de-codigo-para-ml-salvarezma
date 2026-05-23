#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# importacion de librerias
from src.trainers import select_best_elasticnet

CANDIDATES = [
    (0.5, 0.5),
    (0.2, 0.2),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.3, 0.2),
]

# entrenar el modelo
result = select_best_elasticnet(CANDIDATES)
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
