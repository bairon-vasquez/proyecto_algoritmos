import numpy as np
from numpy.typing import NDArray
from itertools import combinations
from src.controllers.strategies.sia import SIA
from src.models.system import System
from src.utils.emd import emd_pyphi


class GeometricSIA(SIA):
    """
    Estrategia geométrica para encontrar la bipartición óptima.

    Implementa el algoritmo del documento 2 (GeoMIP):
    1. Descompone el sistema en tensores elementales
    2. Construye la tabla de costos T usando BFS modificado
    3. Identifica biparticiones candidatas desde T
    4. Evalúa cada candidata con EMD y retorna la óptima

    Optimizaciones de memoria:
    - N <= 8 : tabla densa completa (todos los pares i,j)
    - N > 8  : solo fila del estado inicial (sparse)
    - N > 15 : heurística para reducir candidatas
    """

    def __init__(self, sistema: System):
        super().__init__(sistema)
        self.tabla_costos: dict = {}
        self.tensores: list = []

    # ==================================================================
    # PUNTO DE ENTRADA
    # ==================================================================

    def aplicar_estrategia(self) -> dict:
        n = self.sistema.n
        self.tensores = self.sistema.obtener_tensores()
        self.tabla_costos = self._calcular_tabla_costos()
        dist_orig = self.sistema.distribucion_estado_inicial()
        candidatas = self._identificar_candidatas()

        mejor_phi = float('inf')
        mejor_biparticion = (list(range(n // 2)), list(range(n // 2, n)))
        mejor_dist_part = dist_orig.copy()

        for (parte1_fut, parte1_pres, parte2_fut, parte2_pres) in candidatas:
            try:
                dist_part = self._distribucion_particionada(
                    parte1_fut, parte1_pres,
                    parte2_fut, parte2_pres
                )
                phi = emd_pyphi(dist_orig, dist_part)
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_biparticion = (
                        sorted(set(parte1_fut + parte1_pres)),
                        sorted(set(parte2_fut + parte2_pres))
                    )
                    mejor_dist_part = dist_part
                if mejor_phi < 1e-10:
                    break
            except Exception:
                continue

        return {
            'biparticion': mejor_biparticion,
            'phi': mejor_phi,
            'dist_orig': dist_orig,
            'dist_part': mejor_dist_part,
            'tiempo': 0.0
        }

    # ==================================================================
    # PASO 2: TABLA DE COSTOS T
    # ==================================================================

    def _calcular_tabla_costos(self) -> dict:
        """
        Construye la tabla de costos T para cada variable.

        Estrategia segun tamano N:
        - N <= 8 : matriz densa (num_estados x num_estados)
        - N > 8  : solo fila del estado inicial con BFS vectorizado
                   Memoria: n * 2^n * 8 bytes
                   N=10:  80 KB | N=15: 3.8 MB | N=20: 160 MB
        """
        n = self.sistema.n
        num_estados = 2 ** n
        tabla = {}
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)

        for var_idx, tensor in enumerate(self.tensores):
            if n <= 8:
                T = np.zeros((num_estados, num_estados), dtype=np.float64)
                for i in range(num_estados):
                    memo = {}
                    for j in range(num_estados):
                        T[i, j] = self._bfs_costo(i, j, tensor, n, memo)
                tabla[var_idx] = T
            else:
                fila = self._bfs_vectorizado(idx_inicio, tensor, n)
                tabla[var_idx] = {
                    'fila_inicio': fila,
                    'idx_inicio': idx_inicio
                }

        return tabla

    # ==================================================================
    # BFS RECURSIVO CON MEMOIZACION (para N <= 8)
    # ==================================================================

    def _bfs_costo(self, i, j, tensor, n, memo):
        """
        Calcula t_x(i,j) recursivamente con memoizacion.

        Formula (ecuacion 3.1 del documento 2):
            t_x(i,j) = gamma * (|X[i]-X[j]| + suma t_x(k,j) para k en N*(i,j))
        donde:
            gamma   = 2^(-dH(i,j))
            N*(i,j) = vecinos de i que acortan distancia a j en exactamente 1
        """
        if (i, j) in memo:
            return memo[(i, j)]
        if i == j:
            memo[(i, j)] = 0.0
            return 0.0

        d = self._hamming(i, j)
        gamma = 2.0 ** (-d)
        diff = abs(float(tensor[i]) - float(tensor[j]))

        if d == 1:
            costo = gamma * diff
            memo[(i, j)] = costo
            return costo

        suma_vecinos = 0.0
        for k in self._vecinos_optimos(i, j, n):
            suma_vecinos += self._bfs_costo(k, j, tensor, n, memo)

        costo = gamma * (diff + suma_vecinos)
        memo[(i, j)] = costo
        return costo

    # ==================================================================
    # BFS VECTORIZADO POR NIVELES (para N > 8)
    # ==================================================================

    def _bfs_vectorizado(self, i, tensor, n):
        """
        Calcula todos los costos t_x(i, j) para j en [0, 2^n)
        procesando nivel por nivel de distancia de Hamming.

        Identico matematicamente a _bfs_costo pero sin recursion.
        Tiempos esperados:
            N=10: ~1s  (antes: 72 minutos)
            N=15: ~5s
            N=20: ~60s
        """
        num_estados = 2 ** n
        costos = np.zeros(num_estados, dtype=np.float64)

        # Nivel 1: estados a distancia Hamming 1 de i
        for bit in range(n):
            j = i ^ (1 << bit)
            diff = abs(float(tensor[i]) - float(tensor[j]))
            costos[j] = 0.5 * diff  # gamma = 2^(-1)

        # Niveles 2..n: estados a distancia d de i
        for d in range(2, n + 1):
            gamma = 2.0 ** (-d)
            for bits in combinations(range(n), d):
                mask = 0
                for b in bits:
                    mask |= (1 << b)
                j = i ^ mask
                diff = abs(float(tensor[i]) - float(tensor[j]))
                # Vecinos optimos de j hacia i: estados a distancia d-1
                # Se obtienen flipeando uno de los bits del conjunto
                suma_vecinos = sum(costos[j ^ (1 << b)] for b in bits)
                costos[j] = gamma * (diff + suma_vecinos)

        return costos

    # ==================================================================
    # UTILIDADES GEOMETRICAS
    # ==================================================================

    def _vecinos_optimos(self, i, j, n):
        """Vecinos de i que acortan la distancia a j en exactamente 1."""
        d_ij = self._hamming(i, j)
        return [
            i ^ (1 << bit)
            for bit in range(n)
            if self._hamming(i ^ (1 << bit), j) == d_ij - 1
        ]

    @staticmethod
    def _hamming(a, b):
        """Distancia de Hamming entre dos enteros."""
        return bin(a ^ b).count("1")

    # ==================================================================
    # PASO 3: IDENTIFICAR BIPARTICIONES CANDIDATAS
    # ==================================================================

    def _identificar_candidatas(self) -> list:
        """
        Identifica biparticiones candidatas usando la tabla de costos T.

        Estrategia 1 (todos los N):
            Costos cero revelan independencia causal -> biparticion natural.

        Estrategia 2 (N <= 15):
            Genera todas las biparticiones posibles (2^(n-1) - 1).

        Estrategia 3 (N > 15):
            Heuristica: ordena variables por suma de costos y evalua
            solo las biparticiones mas prometedoras.
        """
        n = self.sistema.n
        indices = list(range(n))
        candidatas = []
        vistas = set()
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)

        # Obtener fila del estado inicial segun formato de tabla
        filas_por_var = {}
        for var_idx, T in self.tabla_costos.items():
            if isinstance(T, dict):
                filas_por_var[var_idx] = T['fila_inicio']
            else:
                filas_por_var[var_idx] = T[idx_inicio]

        # --- Estrategia 1: costos cero como guia ---
        for var_idx, fila in filas_por_var.items():
            for j, costo in enumerate(fila):
                if costo < 1e-10 and j != idx_inicio:
                    xor = idx_inicio ^ j
                    bits_distintos = [
                        bit for bit in range(n) if (xor >> bit) & 1
                    ]
                    if bits_distintos:
                        parte1 = sorted(bits_distintos)
                        parte2 = sorted(
                            [k for k in indices if k not in parte1]
                        )
                        if parte2:
                            clave = (tuple(parte1), tuple(parte2))
                            if clave not in vistas:
                                vistas.add(clave)
                                candidatas.append((
                                    parte1, parte1, parte2, parte2
                                ))

        if n <= 15:
            # --- Estrategia 2: exhaustiva ---
            for tam in range(1, n):
                for combo in combinations(indices, tam):
                    parte1 = sorted(combo)
                    parte2 = sorted(
                        [k for k in indices if k not in parte1]
                    )
                    clave = (tuple(parte1), tuple(parte2))
                    clave_inv = (tuple(parte2), tuple(parte1))
                    if clave not in vistas and clave_inv not in vistas:
                        vistas.add(clave)
                        candidatas.append((
                            parte1, parte1, parte2, parte2
                        ))
        else:
            # --- Estrategia 3: heuristica para N > 15 ---
            costos_var = [
                (fila.sum(), var_idx)
                for var_idx, fila in filas_por_var.items()
            ]
            costos_var.sort()
            vars_ord = [v for _, v in costos_var]

            # Cada variable sola vs el resto
            for _, var in costos_var:
                parte1 = [var]
                parte2 = [k for k in indices if k != var]
                clave = (tuple(parte1), tuple(parte2))
                if clave not in vistas:
                    vistas.add(clave)
                    candidatas.append((parte1, parte1, parte2, parte2))

            # Mitades crecientes segun orden de costos
            for tam in range(1, n // 2 + 1):
                parte1 = sorted(vars_ord[:tam])
                parte2 = sorted(vars_ord[tam:])
                clave = (tuple(parte1), tuple(parte2))
                if clave not in vistas:
                    vistas.add(clave)
                    candidatas.append((parte1, parte1, parte2, parte2))

            # Pares de variables con menor costo combinado
            for i_idx in range(min(5, n)):
                for j_idx in range(i_idx + 1, min(8, n)):
                    parte1 = sorted([vars_ord[i_idx], vars_ord[j_idx]])
                    parte2 = sorted(
                        [k for k in indices if k not in parte1]
                    )
                    clave = (tuple(parte1), tuple(parte2))
                    if clave not in vistas:
                        vistas.add(clave)
                        candidatas.append((
                            parte1, parte1, parte2, parte2
                        ))

        return candidatas

    # ==================================================================
    # PASO 4: DISTRIBUCION DEL SISTEMA PARTICIONADO
    # ==================================================================

    def _distribucion_particionada(
        self,
        parte1_futuro, parte1_presente,
        parte2_futuro, parte2_presente
    ):
        """
        Calcula P(sistema_particionado | estado_inicial).

        Para la biparticion (M1/P1) | (M2/P2):
        1. Marginalizar filas P2 -> dist parte 1
        2. Marginalizar filas P1 -> dist parte 2
        3. dist = dist1 x dist2  (producto de Kronecker)

        Teorema 1.2.1 del documento 1:
        P(M1+M2 t+1 | P1+P2 t) = P(M1 t+1|P1 t) * P(M2 t+1|P2 t)
        """
        n = self.sistema.n

        def calcular_dist_parte(futuro, presente):
            if not futuro or not presente:
                return np.array([1.0])

            cols = [j for j in range(n) if j in futuro]
            tpm_cols = self.sistema.tpm[:, cols]

            n_pres = len(presente)
            num_pres = 2 ** n_pres
            tpm_red = np.zeros((num_pres, len(cols)), dtype=np.float64)
            conteos = np.zeros(num_pres, dtype=int)

            for s in range(2 ** n):
                idx_red = 0
                for pos, var in enumerate(sorted(presente)):
                    bit = (s >> var) & 1
                    idx_red |= (bit << pos)
                tpm_red[idx_red] += tpm_cols[s]
                conteos[idx_red] += 1

            for i in range(num_pres):
                if conteos[i] > 0:
                    tpm_red[i] /= conteos[i]

            estado_red = ''.join(
                self.sistema.estado_inicial[v]
                for v in sorted(presente)
            )
            idx = int(estado_red[::-1], 2)
            probs = tpm_red[idx]
            dist = np.array([1.0])
            for p in probs:
                dist = np.kron(dist, np.array([1.0 - p, p]))
            return dist

        dist1 = calcular_dist_parte(parte1_futuro, parte1_presente)
        dist2 = calcular_dist_parte(parte2_futuro, parte2_presente)
        return np.kron(dist1, dist2)
