import numpy as np
from numpy.typing import NDArray


def hamming_distance(a: int, b: int) -> int:
    """
    Distancia de Hamming entre dos enteros.
    Cuenta el número de bits distintos usando XOR.
    Ejemplo: hamming_distance(5, 3) → 5=101, 3=011, XOR=110 → 2
    """
    return bin(a ^ b).count("1")


def calcular_matriz_costos(n: int) -> NDArray[np.float64]:
    """
    Construye la matriz de costos de tamaño (2^n x 2^n)
    donde costs[i][j] = distancia de Hamming entre estado i y estado j.

    Esta es la métrica de transporte que usa EMD para medir
    cuánto "trabajo" cuesta mover probabilidad del estado i al estado j.
    """
    num_states = 2 ** n
    costs = np.zeros((num_states, num_states), dtype=np.float64)
    for i in range(num_states):
        for j in range(i + 1, num_states):
            d = hamming_distance(i, j)
            costs[i, j] = d
            costs[j, i] = d
    return costs


def emd_pyphi(
    u: NDArray[np.float64],
    v: NDArray[np.float64]
) -> float:
    """
    Calcula la Earth Mover's Distance (EMD) entre dos distribuciones
    de probabilidad u y v usando distancia Hamming como métrica de costo.

    Esta es exactamente la función del documento 1 (Listing 2.1),
    adaptada para usar POT en lugar de pyemd.

    Parámetros:
        u : distribución de probabilidad del sistema original
        v : distribución de probabilidad del sistema particionado

    Retorna:
        float : valor EMD (0 = distribuciones idénticas)
    """
    import ot

    if not all(isinstance(arr, np.ndarray) for arr in [u, v]):
        raise TypeError("u y v deben ser numpy arrays.")

    n_states = len(u)
    n_vars = int(np.log2(n_states))

    # Construir matriz de costos con distancia Hamming
    costs = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(i):
            costs[i, j] = hamming_distance(i, j)
            costs[j, i] = costs[i, j]

    # Normalizar distribuciones (POT requiere que sumen exactamente 1)
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    suma_u = u.sum()
    suma_v = v.sum()

    if suma_u > 0:
        u = u / suma_u
    if suma_v > 0:
        v = v / suma_v

    # Calcular EMD usando POT (equivalente exacto a pyemd)
    return float(ot.emd2(u, v, costs))