import numpy as np
from src.models.core.ncube import NCube
from src.models.core.system import System
from src.models.core.solution import Solution
from src.models.enums.notation import Notation

# --- Prueba NCube ---
data = np.array([[[0.1, 0.9], [0.4, 0.6]], [[0.3, 0.7], [0.8, 0.2]]])
cubo = NCube(data, [0, 1, 2])
print("NCube creado:", cubo)
print("Shape datos:", cubo.data.shape)  # debe ser (2, 2, 2)

# --- Prueba System desde TPM manual (sistema de 2 variables) ---
# TPM estado-nodo: 4 estados x 2 variables
tpm = np.array([
    [0.5, 0.5],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
], dtype=np.float64)

sys = System(tpm, "00")
print("\nSystem creado:", sys)
print("NCubos:", sys.ncubos)

dist = sys.distribucion_marginal()
print("Distribución marginal:", dist)
print("Suma distribución:", dist.sum())  # debe ser ~1.0

# --- Prueba Solution ---
sol = Solution(
    particion=[[0], [1]],
    perdida=0.123,
    estrategia="test",
    k=2
)
print("\n", sol)
print("Es válida:", sol.es_valida())  # debe ser True
print("Todo OK")