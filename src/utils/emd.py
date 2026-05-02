import numpy as np
from numpy.typing import NDArray


def hamming_distance(a: int, b: int) -> int:
    """Distancia de Hamming entre dos enteros."""
    return bin(a ^ b).count("1")


def emd_pyphi(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    max_support: int = 500
) -> float:
    """
    Calcula la Earth Mover's Distance (EMD) entre dos distribuciones
    de probabilidad usando distancia Hamming como métrica de costo.

    Implementación SPARSE: solo construye la matriz de costos para los
    estados con probabilidad > 0, lo que permite escalar a N=20 sin
    problemas de memoria.

    Para distribuciones densas (muchos estados con prob > 0), limita
    al top max_support estados para mantener tiempos razonables.

    Parámetros:
        u           : distribución del sistema original
        v           : distribución del sistema particionado
        max_support : máximo de estados a considerar por distribución

    Retorna:
        float : valor EMD (0 = distribuciones idénticas)

    Memoria:
        N=10:  max 500*500*8 = 2 MB   (antes: 1024*1024*8 = 8 MB)
        N=15:  max 500*500*8 = 2 MB   (antes: 32768*32768*8 = 8 GB!)
        N=20:  max 500*500*8 = 2 MB   (antes: 1M*1M*8 = 8 TB!!!)
    """
    import ot

    if not all(isinstance(arr, np.ndarray) for arr in [u, v]):
        raise TypeError("u y v deben ser numpy arrays.")

    eps = 1e-12

    # Encontrar estados con probabilidad significativa
    support_u = np.where(u > eps)[0]
    support_v = np.where(v > eps)[0]

    if len(support_u) == 0 or len(support_v) == 0:
        return 0.0

    # Limitar al top-k estados por probabilidad si el soporte es muy grande
    if len(support_u) > max_support:
        support_u = np.argsort(u)[-max_support:]
    if len(support_v) > max_support:
        support_v = np.argsort(v)[-max_support:]

    # Unión de todos los estados relevantes
    all_support = np.unique(np.concatenate([support_u, support_v]))
    m = len(all_support)

    # Construir matriz de costos SOLO para estados relevantes
    costs = np.zeros((m, m), dtype=np.float64)
    for ii in range(m):
        for jj in range(ii + 1, m):
            d = hamming_distance(int(all_support[ii]), int(all_support[jj]))
            costs[ii, jj] = d
            costs[jj, ii] = d

    # Extraer marginals para el soporte reducido
    idx_map = {int(s): i for i, s in enumerate(all_support)}
    u_small = np.zeros(m, dtype=np.float64)
    v_small = np.zeros(m, dtype=np.float64)

    for s in support_u:
        s_int = int(s)
        if s_int in idx_map:
            u_small[idx_map[s_int]] = u[s_int]
    for s in support_v:
        s_int = int(s)
        if s_int in idx_map:
            v_small[idx_map[s_int]] = v[s_int]

    # Normalizar
    su = u_small.sum()
    sv = v_small.sum()
    if su > 0:
        u_small = u_small / su
    if sv > 0:
        v_small = v_small / sv

    return float(ot.emd2(u_small, v_small, costs))
