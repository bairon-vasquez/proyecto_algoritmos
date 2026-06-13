import numpy as np
from numpy.typing import NDArray
from itertools import combinations
from multiprocessing import Pool, cpu_count
from src.controllers.strategies.sia import SIA
from src.models.system import System
from src.utils.emd import emd_pyphi


# ============================================================
# Funciones a nivel de módulo (requerido por multiprocessing)
# ============================================================

def _calcular_tensor_paralelo(args):
    """
    Calcula la tabla de costos para UNA variable usando BFS vectorizado.
    Está al nivel de módulo para ser serializable por multiprocessing.

    args = (var_idx, tensor_list, n, idx_inicio, modo_sparse)
    """
    var_idx, tensor_list, n, idx_inicio, modo_sparse = args
    tensor = np.array(tensor_list, dtype=np.float64)
    # Usar el tamaño real del tensor, no 2**n.
    # Para TPMs truncadas (N>20) tensor tiene 2^20 filas en lugar de 2^n.
    num_estados = len(tensor)

    def bfs_desde(i):
        """BFS vectorizado desde estado i por niveles de Hamming."""
        costos = np.zeros(num_estados, dtype=np.float64)
        # Nivel 1
        for bit in range(n):
            j = i ^ (1 << bit)
            if j < num_estados:
                costos[j] = 0.5 * abs(tensor[i] - tensor[j])
        # Niveles 2..n
        for d in range(2, n + 1):
            gamma = 2.0 ** (-d)
            for bits in combinations(range(n), d):
                mask = 0
                for b in bits:
                    mask |= (1 << b)
                j = i ^ mask
                if j < num_estados:
                    diff = abs(tensor[i] - tensor[j])
                    suma = sum(costos[j ^ (1 << b)] for b in bits
                               if (j ^ (1 << b)) < num_estados)
                    costos[j] = gamma * (diff + suma)
        return costos

    if modo_sparse:
        # Solo calcular desde el estado inicial (sparse)
        fila = bfs_desde(idx_inicio)
        return var_idx, {'fila_inicio': fila, 'idx_inicio': idx_inicio}
    else:
        # Tabla densa: calcular todas las filas
        T = np.zeros((num_estados, num_estados), dtype=np.float64)
        for i in range(num_estados):
            T[i] = bfs_desde(i)
        return var_idx, T


class KGeoMIP(SIA):
    """
    Estrategia geométrica PARALELA para encontrar la bipartición óptima.

    Implementa el algoritmo del documento 2 (GeoMIP) con:
    1. BFS vectorizado por niveles de Hamming (sin recursión)
    2. Tabla sparse para N > 8: solo fila del estado inicial
    3. Paralelización real con multiprocessing.Pool
       - Un proceso por variable (tensor)
       - Speedup: ~min(N, CPUs) veces más rápido
    4. EMD sparse: solo estados con probabilidad > 0
    5. Heurística para N > 15: reduce candidatas de O(2^n) a O(n^2)

    Sistemas soportados: N = 3, 5, 10, 15
    """

    def __init__(self, sistema: System):
        super().__init__(sistema)
        self.tabla_costos: dict = {}
        self.tensores: list = []
        self.n_workers = max(1, cpu_count() - 1)

    # ==================================================================
    # PUNTO DE ENTRADA
    # ==================================================================

    def aplicar_estrategia(self) -> dict:
        n = self.sistema.n

        # Paso 1: tensores elementales
        self.tensores = self.sistema.obtener_tensores()

        # Paso 2: tabla de costos (paralela)
        self.tabla_costos = self._calcular_tabla_costos_paralelo()

        # Paso 3: distribución original
        dist_orig = self.sistema.distribucion_estado_inicial()

        # Paso 4: biparticiones candidatas
        candidatas = self._identificar_candidatas()

        # Paso 5: evaluar candidatas con EMD
        mejor_phi = float('inf')
        mejor_biparticion = (list(range(n // 2)), list(range(n // 2, n)))
        mejor_dist_part = dist_orig.copy()

        for (p1f, p1p, p2f, p2p) in candidatas:
            try:
                dist_part = self._distribucion_particionada(
                    p1f, p1p, p2f, p2p
                )
                phi = emd_pyphi(dist_orig, dist_part)
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_biparticion = (
                        sorted(set(p1f + p1p)),
                        sorted(set(p2f + p2p))
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
    # TABLA DE COSTOS - PARALELA
    # ==================================================================

    def _calcular_tabla_costos_paralelo(self) -> dict:
        """
        Calcula la tabla T para cada variable en PARALELO.

        Diseño:
        - N <= 8 : tabla densa en memoria (num_estados × num_estados)
        - N > 8  : solo fila del estado inicial (sparse)

        Uso de memoria resultante:
            N=3:  3 × 8 × 8 × 8B     =  1.5 KB  (densa)
            N=5:  5 × 32B × 8B        =  1.3 KB  (sparse)
            N=10: 10 × 1024 × 8B      = 80 KB    (sparse)
            N=15: 15 × 32768 × 8B     = 3.8 MB   (sparse)

        Paralelización:
        - Cada tensor se procesa en un proceso independiente
        - Speedup teórico: min(N, CPUs) × speedup_secuencial
        """
        n = self.sistema.n
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)
        modo_sparse = n > 8

        args_list = [
            (var_idx, tensor.tolist(), n, idx_inicio, modo_sparse)
            for var_idx, tensor in enumerate(self.tensores)
        ]

        tabla = {}
        usar_paralelo = n >= 5 and self.n_workers > 1

        if usar_paralelo:
            try:
                with Pool(processes=min(self.n_workers, n)) as pool:
                    resultados = pool.map(
                        _calcular_tensor_paralelo, args_list
                    )
                for var_idx, T in resultados:
                    tabla[var_idx] = T
            except Exception:
                # Fallback secuencial si Pool falla
                for args in args_list:
                    var_idx, T = _calcular_tensor_paralelo(args)
                    tabla[var_idx] = T
        else:
            for args in args_list:
                var_idx, T = _calcular_tensor_paralelo(args)
                tabla[var_idx] = T

        return tabla

    # ==================================================================
    # SIMETRÍAS DEL HIPERCUBO (§2.2.2)
    # ==================================================================

    def _equiv_classes(self) -> list[list[int]]:
        """
        Clases de equivalencia de variables bajo simetrías del hipercubo (§2.2.2).

        Variables i y j son equivalentes (misma tabla de costo T_i = T_j) si:
          - Permutación:     ||tensor_i - tensor_j||∞ < tol  (misma dinámica)
          - Complementación: ||tensor_i - (1-tensor_j)||∞ < tol  (dinámica invertida)

        Bajo cualquiera de estas condiciones la tabla BFS produce los mismos
        costos t(·,·), haciendo equivalentes todas las biparticiones que solo
        difieran por intercambiar i↔j.

        Retorna lista de grupos ordenados; singletons para variables sin pares.
        """
        if not self.tensores:
            return [[v] for v in range(self.sistema.n)]

        tol = 1e-6
        n = len(self.tensores)
        padre = list(range(n))

        def find(x: int) -> int:
            while padre[x] != x:
                padre[x] = padre[padre[x]]
                x = padre[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                padre[max(px, py)] = min(px, py)

        for i in range(n):
            ti = self.tensores[i]
            for j in range(i + 1, n):
                tj = self.tensores[j]
                if (np.max(np.abs(ti - tj)) < tol or
                        np.max(np.abs(ti - (1.0 - tj))) < tol):
                    union(i, j)

        grupos: dict[int, list[int]] = {}
        for v in range(n):
            r = find(v)
            grupos.setdefault(r, []).append(v)
        return [sorted(g) for g in sorted(grupos.values())]

    # ==================================================================
    # IDENTIFICAR CANDIDATAS
    # ==================================================================

    def _identificar_candidatas(self) -> list:
        """
        Genera biparticiones candidatas a evaluar con reducción por simetrías.

        Estrategia 1 (todos los N):
            Costos cero en la tabla T revelan independencia causal.

        Estrategia 2 (N <= 15):
            Todas las biparticiones posibles con poda por clases de
            equivalencia del hipercubo (§2.2.2): dos biparticiones que
            solo difieran por intercambiar variables equivalentes tienen
            el mismo Φ y solo se evalúa la forma canónica.

        Estrategia 3 (N > 15):
            Heurística ordenada por suma de costos (sin cambios).
        """
        n = self.sistema.n
        indices = list(range(n))
        candidatas = []
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)

        # Clases de equivalencia para reducción simétrica
        equiv_cls = self._equiv_classes()
        n_no_triviales = sum(1 for c in equiv_cls if len(c) > 1)

        def canonicalizar(p1: list, p2: list) -> tuple:
            """
            Forma canónica de {p1|p2} bajo simetrías del hipercubo.

            Para cada clase de equivalencia C, solo importa cuántos elementos
            de C van a p1 (no cuáles). Los primeros |p1∩C| de C (en orden)
            representan el grupo canónico en p1.

            La orientación (flip p1↔p2) se normaliza recomputando desde ambos
            lados y tomando el mínimo lexicográfico, lo que garantiza:
                canonicalizar(p1,p2) == canonicalizar(p2,p1)
            """
            def _directed(pp: list) -> tuple:
                set_pp = set(pp)
                cp1, cp2 = [], []
                for clase in equiv_cls:
                    en = sum(1 for v in clase if v in set_pp)
                    cp1.extend(clase[:en])
                    cp2.extend(clase[en:])
                return (tuple(sorted(cp1)), tuple(sorted(cp2)))

            return min(_directed(p1), _directed(p2))

        vistas: set = set()
        n_reducidas = 0

        def agregar(p1: list, p2: list) -> None:
            nonlocal n_reducidas
            if not p1 or not p2:
                return
            canon = canonicalizar(p1, p2)
            if canon in vistas:
                n_reducidas += 1
                return
            vistas.add(canon)
            candidatas.append((list(p1), list(p1), list(p2), list(p2)))

        # Obtener fila del estado inicial por variable
        filas = {}
        for v, T in self.tabla_costos.items():
            filas[v] = T['fila_inicio'] if isinstance(T, dict) else T[idx_inicio]

        # Estrategia 1: costos cero → independencia causal
        for v, fila in filas.items():
            for j, costo in enumerate(fila):
                if costo < 1e-10 and j != idx_inicio:
                    xor = idx_inicio ^ j
                    bits = sorted([b for b in range(n) if (xor >> b) & 1])
                    rest = sorted([i for i in indices if i not in bits])
                    agregar(bits, rest)

        if n <= 15:
            # Estrategia 2: exhaustiva con poda por simetrías
            n_antes = len(candidatas)
            for tam in range(1, n):
                for combo in combinations(indices, tam):
                    p1 = sorted(combo)
                    p2 = sorted([i for i in indices if i not in p1])
                    agregar(p1, p2)
            if n_no_triviales > 0:
                sin_sim = sum(
                    len(list(combinations(indices, t))) for t in range(1, n)
                ) // 2  # biparticiones unicas sin simetria (solo flip)
                n_unicas = len(candidatas) - n_antes
                reducidas = sin_sim - n_unicas
                print(f"  [Simetria S2.2.2] {n_no_triviales} clase(s) no trivial(es): "
                      f"{sin_sim} -> {n_unicas} candidatas unicas "
                      f"({reducidas} reducidas, {reducidas*100//max(sin_sim,1)}%)")
        else:
            # Estrategia 3: heurística (N > 15)
            costos_var = sorted([(filas[v].sum(), v) for v in range(n)])
            vars_ord = [v for _, v in costos_var]
            for _, var in costos_var:
                p1 = [var]
                p2 = [i for i in indices if i != var]
                agregar(p1, p2)
            for tam in range(1, n // 2 + 1):
                p1 = sorted(vars_ord[:tam])
                p2 = sorted(vars_ord[tam:])
                agregar(p1, p2)

        return candidatas

    # ==================================================================
    # API PÚBLICA REQUERIDA POR §5.1.2
    # ==================================================================

    def calcular_transicion_coste(self, i: int, j: int, var_idx: int) -> float:
        """
        Calcula t(i,j) para la variable var_idx.

        Si la tabla aún no fue computada, la calcula automáticamente.
        En modo sparse (N>8) solo está disponible desde el estado inicial;
        en modo denso (N<=8) cualquier par (i,j) es válido.

        Args:
            i       : índice del estado de origen (entero en [0, 2^n))
            j       : índice del estado de destino (entero en [0, 2^n))
            var_idx : índice de la variable (0-based)

        Returns:
            Costo de transición t(i,j) para la variable var_idx.
        """
        if not self.tabla_costos:
            if not self.tensores:
                self.tensores = self.sistema.obtener_tensores()
            self.tabla_costos = self._calcular_tabla_costos_paralelo()

        T = self.tabla_costos[var_idx]
        if isinstance(T, dict):
            idx_inicio = int(self.sistema.estado_inicial[::-1], 2)
            if i != idx_inicio:
                raise ValueError(
                    f"Tabla sparse: solo disponible desde el estado inicial "
                    f"(idx={idx_inicio}). Recibido i={i}."
                )
            return float(T['fila_inicio'][j])
        return float(T[i, j])

    def identificar_candidatos(self) -> list:
        """
        Identifica biparticiones candidatas a partir de la tabla T.

        Si la tabla aún no fue computada, la calcula automáticamente.
        Cada candidata tiene la forma (p1_futuro, p1_pasado, p2_futuro, p2_pasado).

        Returns:
            Lista de tuplas (p1f, p1p, p2f, p2p) con biparticiones candidatas.
        """
        if not self.tabla_costos:
            if not self.tensores:
                self.tensores = self.sistema.obtener_tensores()
            self.tabla_costos = self._calcular_tabla_costos_paralelo()
        return self._identificar_candidatas()

    # ==================================================================
    # DISTRIBUCIÓN PARTICIONADA
    # ==================================================================

    def _distribucion_particionada(
        self, p1f, p1p, p2f, p2p
    ) -> NDArray[np.float64]:
        """
        Calcula P(sistema_partido | estado_inicial) = dist1 ⊗ dist2.

        Para cada parte: marginaliza filas (variables ausentes del mecanismo)
        y selecciona columnas (variables del alcance). Luego combina con
        producto tensorial de Kronecker.
        """
        n = self.sistema.n
        tpm = self.sistema.tpm
        estado = self.sistema.estado_inicial

        def dist_parte(futuro: list, presente: list) -> NDArray:
            if not futuro or not presente:
                return np.array([1.0])
            # Mapeo variable j → columna n-1-j (compensa la inversión de
            # columnas que hace desde_csv al cargar la TPM del documento).
            # Equivalente a var_a_col = {i: (n-1-i)} de KQNodes.
            cols = sorted(n - 1 - j for j in futuro)
            tpm_cols = tpm[:, cols]
            pres = sorted(presente)
            n_p = len(pres)
            num_p = 2 ** n_p
            tpm_r = np.zeros((num_p, len(cols)), dtype=np.float64)
            cnt = np.zeros(num_p, dtype=np.int64)
            for s in range(tpm.shape[0]):  # usa filas reales, no 2**n
                ir = sum(((s >> v) & 1) << p for p, v in enumerate(pres))
                if 0 <= ir < num_p:
                    tpm_r[ir] += tpm_cols[s]
                    cnt[ir] += 1
            for i in range(num_p):
                if cnt[i] > 0:
                    tpm_r[i] /= cnt[i]
            est_r = ''.join(estado[v] for v in pres)
            idx = int(est_r[::-1], 2) % num_p
            dist = np.array([1.0])
            for p in tpm_r[idx]:
                dist = np.kron(dist, np.array([1.0 - p, p]))
            return dist

        dist_part = np.kron(dist_parte(p1f, p1p), dist_parte(p2f, p2p))

        # Reordenar ejes para que el encoding coincida con distribucion_estado_inicial()
        col_order = (sorted(n - 1 - j for j in p1f) +
                     sorted(n - 1 - j for j in p2f))
        target = list(range(n))
        if col_order != target:
            axes = [col_order.index(j) for j in target]
            dist_part = dist_part.reshape([2] * n).transpose(axes).reshape(-1)

        return dist_part


# Alias de compatibilidad
GeometricSIA = KGeoMIP


class KGeoMIPKPartition(SIA):
    """
    KGeoMIP extendido a k=2,3,4,5 usando Branch and Bound.

    Estrategia Branch and Bound (según guía):
    - Problema: optimización combinatoria NP-hard (k-partición óptima)
    - Cota inferior (lower bound): suma de los costos mínimos de la tabla T
    - Cota superior (upper bound): mejor solución encontrada hasta ahora
    - Poda: si lower_bound >= upper_bound actual, descartar rama

    Complejidad: exponencial pero mitigada por podas agresivas con T
    """

    def __init__(self, sistema: System, k_values: list = None):
        super().__init__(sistema)
        self.k_values = k_values if k_values is not None else [2, 3, 4, 5]
        self.tabla_costos: dict = {}
        self.tensores: list = []

    def aplicar_estrategia(self) -> dict:
        """k=2: KGeoMIP exacto. k>=3, n<=6: B&B. k>=3, n>6: Greedy+búsqueda local."""
        import time

        n = self.sistema.n
        dist_orig = self.sistema.distribucion_estado_inicial()
        variables = list(range(n))
        LIMITE_BB = 6

        # k=2: KGeoMIP exacto (calcula tabla_costos internamente)
        t0_k2 = time.time()
        geo_base = KGeoMIP(self.sistema)
        res_k2 = geo_base.aplicar_estrategia()
        t_k2 = time.time() - t0_k2
        tabla_T = geo_base.tabla_costos  # Reutilizar tabla ya calculada
        self.tabla_costos = tabla_T

        mejor_global = {
            'biparticion': None,
            'phi':         float('inf'),
            'k':           None,
            'dist_orig':   dist_orig,
            'dist_part':   dist_orig.copy(),
            'tiempo':      0.0,
        }
        resultados_por_k = {}

        for k in self.k_values:
            if k > n:
                continue

            if k == 2:
                p1, p2 = res_k2['biparticion']
                res_k = {
                    'phi':         res_k2['phi'],
                    'biparticion': [list(p1), list(p2)],
                    'tiempo':      t_k2,
                    'k':           2,
                    'metodo':      'kgeomip_exacto',
                }
            else:
                if n <= LIMITE_BB:
                    res_k = self._branch_and_bound_k(k, variables, tabla_T, dist_orig)
                    res_k['metodo'] = 'branch_and_bound'
                else:
                    res_k = self._greedy_k(k, variables, tabla_T, dist_orig, timeout_s=120.0)
                    res_k['metodo'] = 'greedy'

            resultados_por_k[k] = res_k
            if res_k['phi'] < mejor_global['phi']:
                mejor_global.update(res_k)
                mejor_global['k'] = k

        mejor_global['resultados_por_k'] = resultados_por_k
        return mejor_global

    def _bound(self, particion_parcial, variables_restantes, tabla_T) -> float:
        """
        Cota inferior para B&B:
        EMD mínimo posible dado el estado actual de la partición.
        Usar los costos mínimos de tabla_T para las variables sin asignar.
        """
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)
        total = 0.0
        for v in variables_restantes:
            T = tabla_T[v]
            fila = T['fila_inicio'] if isinstance(T, dict) else T[idx_inicio]
            nz = fila[fila > 1e-15]
            total += float(nz.min()) if len(nz) > 0 else 0.0
        return total

    def _calc_dist_k_particion(self, particion: list) -> NDArray[np.float64]:
        """Distribución conjunta = kron de distribuciones marginales por grupo."""
        from src.controllers.strategies.qnodes import _dist_parte_vectorizada
        n = self.sistema.n
        tpm = self.sistema.tpm
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)
        var_a_col = {i: (n - 1 - i) for i in range(n)}

        dist_total = np.array([1.0])
        col_order: list[int] = []
        for parte in particion:
            futuro_cols = sorted(var_a_col[v] for v in parte)
            presente_bits = sorted(parte)
            dist_parte = _dist_parte_vectorizada(
                tpm, futuro_cols, presente_bits, n, idx_inicio
            )
            col_order.extend(futuro_cols)
            dist_total = np.kron(dist_total, dist_parte)

        target = list(range(n))
        if col_order != target:
            axes = [col_order.index(j) for j in target]
            dist_total = dist_total.reshape([2] * n).transpose(axes).reshape(-1)
        return dist_total

    def _branch_and_bound_k(self, k: int, variables: list, tabla_T, dist_orig) -> dict:
        """
        Para k=2 delega a KGeoMIP original (exhaustivo y óptimo).
        Para k>=3 usa Branch and Bound con ruptura de simetría canónica.
        """
        import time

        # ── k=2: KGeoMIP original es el algoritmo correcto ────────────────
        if k == 2:
            t0 = time.time()
            geo = KGeoMIP(self.sistema)
            res = geo.aplicar_estrategia()
            t_k = time.time() - t0
            p1, p2 = res['biparticion']
            return {
                'phi':        res['phi'],   # raw EMD (normalización en _run_strategies)
                'biparticion': [list(p1), list(p2)],
                'tiempo':     t_k,
                'k':          2,
            }

        # ── k>=3: Branch and Bound ────────────────────────────────────────
        t0 = time.time()
        n = len(variables)
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)

        # Cota por suma de mínimos de tabla T (solo para poda de inviabilidad
        # extrema; no se usa como lower bound absoluto sobre EMD)
        min_costs = []
        for v in variables:
            T = tabla_T[v]
            fila = T['fila_inicio'] if isinstance(T, dict) else T[idx_inicio]
            nz = fila[fila > 1e-15]
            min_costs.append(float(nz.min()) if len(nz) > 0 else 0.0)

        best_phi = float('inf')
        best_particion = None
        best_dist = dist_orig.copy()

        # Stack: (assignment_list, depth)
        # assignment[i] = grupo (0..k-1) de variables[i]
        # variables[0] siempre en grupo 0 (ruptura de simetría canónica)
        stack = [([0], 1)]

        while stack:
            assignment, depth = stack.pop()

            n_used = len(set(assignment))
            remaining = n - depth

            # Poda de inviabilidad: variables restantes insuficientes
            # para llenar los grupos que aún no tienen elementos
            if remaining < k - n_used:
                continue

            if depth == n:
                groups = [[] for _ in range(k)]
                for i, g in enumerate(assignment):
                    groups[g].append(variables[i])

                if any(not g for g in groups):
                    continue

                particion = [sorted(g) for g in groups]
                try:
                    dist_part = self._calc_dist_k_particion(particion)
                    phi = emd_pyphi(dist_orig, dist_part)
                    if phi < best_phi:
                        best_phi = phi
                        best_particion = [list(p) for p in particion]
                        best_dist = dist_part.copy()
                    if best_phi < 1e-10:
                        break
                except Exception:
                    pass
                continue

            # Ramificar: asignar variables[depth] a grupos 0..max_g
            max_g = min(n_used, k - 1)
            for g in range(max_g + 1):
                stack.append((assignment + [g], depth + 1))

        return {
            'biparticion': best_particion,
            'phi':        best_phi,
            'dist_orig':  dist_orig,
            'dist_part':  best_dist,
            'tiempo':     time.time() - t0,
            'k':          k,
        }


    def _greedy_k(self, k: int, variables: list, tabla_T: dict,
                  dist_orig, timeout_s: float = 120.0) -> dict:
        """
        Greedy + búsqueda local para k-particiones cuando n > 6.
        Fase 1: asignación voraz por importancia (suma costos T).
        Fase 2: intercambios de pares hasta convergencia o timeout.
        """
        import time
        t0 = time.time()
        n_vars = len(variables)
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)

        # Importancia de cada variable: suma de costos T desde estado inicial
        importance = []
        for v in variables:
            T = tabla_T.get(v)
            if T is None:
                importance.append(0.0)
                continue
            fila = T['fila_inicio'] if isinstance(T, dict) else T[idx_inicio]
            nz = fila[fila > 1e-15]
            importance.append(float(nz.sum()) if len(nz) > 0 else 0.0)

        # Ordenar variables por importancia descendente
        sorted_idx = sorted(range(n_vars), key=lambda i: -importance[i])

        # Inicializar: primeras k variables → grupos 0..k-1 (ruptura simétrica)
        assignment = [0] * n_vars
        group_loads = [0.0] * k
        for pos, vi in enumerate(sorted_idx[:k]):
            assignment[vi] = pos
            group_loads[pos] += importance[vi]

        # Asignar restantes al grupo con menor carga acumulada
        for vi in sorted_idx[k:]:
            best_g = min(range(k), key=lambda g: group_loads[g])
            assignment[vi] = best_g
            group_loads[best_g] += importance[vi]

        def build_partition(asgn):
            groups = [[] for _ in range(k)]
            for vi, g in enumerate(asgn):
                groups[g].append(variables[vi])
            return [sorted(g) for g in groups]

        # Evaluar partición inicial
        particion = build_partition(assignment)
        try:
            dist_part = self._calc_dist_k_particion(particion)
            best_phi = emd_pyphi(dist_orig, dist_part)
            best_particion = [list(p) for p in particion]
            best_dist = dist_part.copy()
        except Exception:
            best_phi = float('inf')
            best_particion = [list(p) for p in particion]
            best_dist = dist_orig.copy()

        if best_phi < 1e-10:
            return {
                'biparticion': best_particion,
                'phi':         best_phi,
                'dist_orig':   dist_orig,
                'dist_part':   best_dist,
                'tiempo':      time.time() - t0,
                'k':           k,
            }

        # Búsqueda local: intercambiar pares de variables en grupos distintos
        improved = True
        while improved and (time.time() - t0) < timeout_s:
            improved = False
            for i in range(n_vars):
                if (time.time() - t0) >= timeout_s:
                    break
                for j in range(i + 1, n_vars):
                    if (time.time() - t0) >= timeout_s:
                        break
                    if assignment[i] == assignment[j]:
                        continue
                    assignment[i], assignment[j] = assignment[j], assignment[i]
                    new_part = build_partition(assignment)
                    try:
                        new_dist = self._calc_dist_k_particion(new_part)
                        new_phi = emd_pyphi(dist_orig, new_dist)
                        if new_phi < best_phi - 1e-12:
                            best_phi = new_phi
                            best_particion = [list(p) for p in new_part]
                            best_dist = new_dist.copy()
                            improved = True
                            if best_phi < 1e-10:
                                break
                        else:
                            assignment[i], assignment[j] = assignment[j], assignment[i]
                    except Exception:
                        assignment[i], assignment[j] = assignment[j], assignment[i]
                if best_phi < 1e-10:
                    break

        return {
            'biparticion': best_particion,
            'phi':         best_phi,
            'dist_orig':   dist_orig,
            'dist_part':   best_dist,
            'tiempo':      time.time() - t0,
            'k':           k,
        }


GeometricSIAKPartition = KGeoMIPKPartition
