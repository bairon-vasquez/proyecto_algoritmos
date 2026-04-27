import numpy as np
from src.utils.emd import hamming_distance, emd_pyphi
from src.models.system import System

# --- Prueba hamming ---
print("hamming(0,1):", hamming_distance(0, 1))   # 1
print("hamming(0,3):", hamming_distance(0, 3))   # 2
print("hamming(0,7):", hamming_distance(0, 7))   # 3

# --- Prueba EMD ---
u = np.array([1.0, 0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0, 1.0])
print("EMD máxima (000→111):", emd_pyphi(u, v))  # debe ser 2.0

u2 = np.array([1.0, 0.0, 0.0, 0.0])
v2 = np.array([1.0, 0.0, 0.0, 0.0])
print("EMD mínima (idénticas):", emd_pyphi(u2, v2))  # debe ser 0.0

# --- Prueba System con TPM del documento (Example 1.5) ---
# TPM estado-nodo del sistema ABC del documento 1
tpm = np.array([
    [0, 0, 0],   # estado 000
    [1, 0, 1],   # estado 100
    [0, 1, 1],   # estado 010
    [0, 1, 0],   # estado 110
    [1, 1, 0],   # estado 001
    [1, 0, 1],   # estado 101
    [0, 1, 1],   # estado 011
    [0, 1, 0],   # estado 111
], dtype=np.float64)

sys = System(tpm, "000")
print("\nSystem:", sys)
tensores = sys.obtener_tensores()
print("Tensor A (P(A=1|abc)):", tensores[0])
print("Tensor B (P(B=1|abc)):", tensores[1])
print("Tensor C (P(C=1|abc)):", tensores[2])

dist = sys.distribucion_estado_inicial()
print("Distribución estado 000:", dist)
print("Suma distribución:", dist.sum())  # debe ser 1.0
print("Todo OK")