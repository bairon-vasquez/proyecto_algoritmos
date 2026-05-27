import numpy as np
from numpy.typing import NDArray


def hamming_distance(a: int, b: int) -> int:
    """Distancia de Hamming entre dos enteros (número de bits distintos)."""
    return bin(a ^ b).count("1")


def emd_pyphi(
    u:           NDArray[np.float64],
    v:           NDArray[np.float64],
    max_support: int = 500
) -> float:
    """
    Earth Mover's Distance entre distribuciones u y v con distancia Hamming.

    Implementación SPARSE: solo construye la matriz de costos para estados
    con probabilidad > 0. Para N=15: 2 MB vs 8 GB de la versión densa.

    NOTA: Retorna el valor RAW (sin normalizar). El rango es [0, n].
    Para obtener phi normalizado en [0,1], dividir por n en el llamador.

    Justificación del rango:
    - La distancia Hamming entre dos estados de n bits está en [0, n]
    - El EMD es un promedio ponderado de distancias -> también en [0, n]
    - phi_normalizado = emd_pyphi(u, v) / n -> garantiza [0, 1]
    """
    import ot

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    eps       = 1e-12
    support_u = np.where(u > eps)[0]
    support_v = np.where(v > eps)[0]

    if len(support_u) == 0 or len(support_v) == 0:
        return 0.0

    # Limitar al top-k estados por probabilidad
    if len(support_u) > max_support:
        support_u = np.argsort(u)[-max_support:]
    if len(support_v) > max_support:
        support_v = np.argsort(v)[-max_support:]

    all_support = np.unique(np.concatenate([support_u, support_v]))
    m           = len(all_support)

    # Construir matriz de costos solo para estados relevantes
    costs = np.zeros((m, m), dtype=np.float64)
    for ii in range(m):
        for jj in range(ii + 1, m):
            d             = hamming_distance(int(all_support[ii]),
                                             int(all_support[jj]))
            costs[ii, jj] = d
            costs[jj, ii] = d

    # Extraer y normalizar marginals
    idx_map = {int(s): i for i, s in enumerate(all_support)}
    u_s = np.zeros(m, dtype=np.float64)
    v_s = np.zeros(m, dtype=np.float64)

    for s in support_u:
        si = int(s)
        if si in idx_map:
            u_s[idx_map[si]] = u[si]
    for s in support_v:
        si = int(s)
        if si in idx_map:
            v_s[idx_map[si]] = v[si]

    su = u_s.sum()
    sv = v_s.sum()
    if su > 0: u_s /= su
    if sv > 0: v_s /= sv

    return float(ot.emd2(u_s, v_s, costs))
