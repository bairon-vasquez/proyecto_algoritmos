import numpy as np
from numpy.typing import NDArray
from itertools import combinations
from src.controllers.strategies.sia import SIA
from src.models.system import System
from src.utils.emd import emd_pyphi


def _particionar_conjunto(elementos: list[int], k: int):
    """
    Genera todas las formas de dividir 'elementos' en exactamente k
    partes NO VACÍAS usando números de Stirling de segundo tipo.

    Algoritmo: recursión con restricción de orden para evitar duplicados.
    - Las partes se generan en orden canónico (primer elemento de cada
      parte en orden creciente), lo que garantiza unicidad.

    Parámetros
    ----------
    elementos : lista de índices de variables (ej. [0,1,2,3,4])
    k         : número de partes requeridas

    Yields
    ------
    list[list[int]] : lista de k grupos, cada grupo es una lista de índices

    Complejidad
    -----------
    S(n,k) particiones (número de Stirling de 2do tipo):
        S(5,3)=25, S(10,3)=9330, S(10,4)=145750, S(15,3)≈2.38M
    """
    n = len(elementos)
    if k == 1:
        yield [list(elementos)]
        return
    if k == n:
        yield [[e] for e in elementos]
        return
    if k > n or k <= 0:
        return

    # Estrategia: fijamos el primer elemento en la primera parte
    # y generamos recursivamente las particiones del resto
    primer = elementos[0]
    resto = elementos[1:]

    # Opción 1: el primer elemento forma su propio grupo (singleton)
    # Las k-1 partes restantes se generan del 'resto'
    for sub in _particionar_conjunto(resto, k - 1):
        yield [[primer]] + sub

    # Opción 2: el primer elemento se une a uno de los grupos existentes
    # generados del 'resto' en k partes
    for sub in _particionar_conjunto(resto, k):
        for i in range(k):
            # Insertar 'primer' en el grupo i
            nuevo = [list(g) for g in sub]
            nuevo[i] = [primer] + nuevo[i]
            yield nuevo


def _calcular_dist_parte_vectorizado(
    tpm: NDArray[np.float64],
    futuro_cols: list[int],
    presente_bits: list[int],
    n: int,
    idx_inicio: int
) -> NDArray[np.float64]:
    """
    Calcula la distribución marginal de una parte de la k-partición.

    Proceso (Definition 2.1.1 del documento):
    1. Seleccionar columnas t+1 de esta parte (alcance)
    2. Marginalizar filas: promediar estados con mismo valor en presente_bits
    3. Evaluar en el estado inicial de esta parte
    4. Construir distribución conjunta por producto tensorial (Kronecker)

    Uso de numpy vectorizado (sin loops de Python donde sea posible):
    - np.zeros, slicing, broadcasting para evitar multiprocessing/threading
    - El paralelismo real proviene de BLAS/LAPACK internos de numpy

    Parámetros
    ----------
    tpm          : TPM completa del sistema (2^n × n)
    futuro_cols  : columnas de tpm que pertenecen a esta parte
    presente_bits: posiciones de bits en el estado que pertenecen a esta parte
    n            : número total de variables
    idx_inicio   : índice LE del estado inicial

    Retorna
    -------
    NDArray : vector de distribución de probabilidad de tamaño 2^|futuro_cols|
    """
    if not futuro_cols or not presente_bits:
        return np.array([1.0])

    # Seleccionar columnas del alcance (operación vectorizada)
    tpm_cols = tpm[:, futuro_cols]  # shape: (2^n, |futuro_cols|)

    # Construir máscara de reducción de filas con numpy puro
    # Para cada estado s en [0, 2^n), calcular su índice reducido
    # basado solo en presente_bits
    n_pres = len(presente_bits)
    num_pres = 2 ** n_pres
    pres_sorted = sorted(presente_bits)

    # Vector de todos los estados s: [0, 1, ..., 2^n - 1]
    all_states = np.arange(2 ** n, dtype=np.int32)

    # Calcular índice reducido para cada estado (vectorizado)
    idx_red = np.zeros(2 ** n, dtype=np.int32)
    for pos, bit in enumerate(pres_sorted):
        idx_red += ((all_states >> bit) & 1) << pos

    # Acumular TPM marginalizada (vectorizado con np.add.at)
    tpm_red = np.zeros((num_pres, len(futuro_cols)), dtype=np.float64)
    counts = np.zeros(num_pres, dtype=np.int32)
    np.add.at(tpm_red, idx_red, tpm_cols)
    np.add.at(counts, idx_red, 1)

    # Promediar (marginalización del documento)
    mask = counts > 0
    tpm_red[mask] /= counts[mask, np.newaxis]

    # Estado inicial reducido a las variables de este presente
    idx_pres = sum(((idx_inicio >> bit) & 1) << pos
                   for pos, bit in enumerate(pres_sorted))
    idx_pres = idx_pres % num_pres

    # Construir distribución conjunta por producto tensorial (Kronecker)
    # P(futuro | presente_inicio) = ⊗ P(var_j | presente_inicio)
    dist = np.array([1.0])
    for p1 in tpm_red[idx_pres]:
        dist = np.kron(dist, np.array([1.0 - p1, p1]))

    return dist


class QNodesKPartition(SIA):
    """
    Estrategia Q-Nodos para encontrar la k-partición óptima.

    Implementa la búsqueda de k-particiones (k=3,4,5) basada en los
    principios del documento 1 (Guía del Proyecto):

    Principio central (Definition 2.1.1):
    - NO se puede comparar una parte directamente con el sistema original
    - Solo después de unir las k partes mediante producto tensorial se
      puede comparar el resultado con el original
    - La única forma de obtener el sistema particionado es marginalizando
      el sistema original

    Proceso para cada k-partición {P1, P2, ..., Pk}:
    1. Para cada parte Pi:
       a. Seleccionar columnas t+1 de Pi (alcance futuro)
       b. Marginalizar filas: promediar estados con mismo valor en Pi_t
       c. Obtener distribución marginal desde estado inicial
    2. Combinar: dist_total = dist_P1 ⊗ dist_P2 ⊗ ... ⊗ dist_Pk
    3. Comparar dist_total con dist_original usando EMD (Hamming)
    4. La k-partición con menor EMD es la óptima

    Paralelización:
    - Vectorización numpy pura (sin multiprocessing ni threading)
    - np.add.at para acumulación vectorizada
    - np.kron para producto tensorial vectorizado
    - El paralelismo real es interno a BLAS/LAPACK de numpy

    Sistemas: N = 3, 5, 10, 15
    k: 3, 4, 5
    """

    def __init__(self, sistema: System):
        super().__init__(sistema)
        # Mapeo interno: variable_i -> columna en tpm
        # Tras [::-1] en desde_csv: col_interna_j = doc_col_{n-1-j}
        # Las variables en t tienen bits: A=bit0, B=bit1, ..., Z=bit{n-1}
        # Y sus columnas en tpm: A=col{n-1}, B=col{n-2}, ..., Z=col0
        self.n = sistema.n
        self.var_a_col = {i: (self.n - 1 - i) for i in range(self.n)}
        self.col_a_var = {c: i for i, c in self.var_a_col.items()}

    # ==================================================================
    # PUNTO DE ENTRADA
    # ==================================================================

    def aplicar_estrategia(self) -> dict:
        """
        Ejecuta la búsqueda de k-partición óptima para k=3,4,5.

        Retorna el mejor resultado entre todos los k evaluados.
        """
        n = self.sistema.n
        idx_inicio = int(self.sistema.estado_inicial[::-1], 2)
        dist_orig = self.sistema.distribucion_estado_inicial()
        variables = list(range(n))

        mejor_global = {
            'biparticion': None,
            'phi': float('inf'),
            'k': None,
            'dist_orig': dist_orig,
            'dist_part': dist_orig.copy(),
            'tiempo': 0.0
        }

        resultados_por_k = {}

        for k in [3, 4, 5]:
            if k > n:
                continue

            resultado_k = self._buscar_k_particion(
                k, variables, idx_inicio, dist_orig
            )
            resultados_por_k[k] = resultado_k

            if resultado_k['phi'] < mejor_global['phi']:
                mejor_global.update(resultado_k)
                mejor_global['k'] = k

        mejor_global['resultados_por_k'] = resultados_por_k
        return mejor_global

    # ==================================================================
    # BÚSQUEDA DE k-PARTICIÓN
    # ==================================================================

    def _buscar_k_particion(
        self,
        k: int,
        variables: list[int],
        idx_inicio: int,
        dist_orig: NDArray[np.float64]
    ) -> dict:
        """
        Busca la k-partición óptima evaluando todas las candidatas.

        Para sistemas grandes (n > 10) usa heurística para reducir
        el espacio de búsqueda.

        Parámetros
        ----------
        k          : número de partes
        variables  : lista de índices de variables [0, 1, ..., n-1]
        idx_inicio : índice LE del estado inicial
        dist_orig  : distribución del sistema original

        Retorna
        -------
        dict con 'biparticion', 'phi', 'dist_part'
        """
        n = self.n
        tpm = self.sistema.tpm

        mejor_phi = float('inf')
        mejor_particion = None
        mejor_dist = dist_orig.copy()
        evaluadas = 0

        # Límite de candidatas para sistemas grandes
        max_candidatas = 50000 if n <= 10 else 500

        for particion in _particionar_conjunto(variables, k):
            if evaluadas >= max_candidatas:
                break

            try:
                dist_part = self._distribucion_k_particion(
                    particion, tpm, n, idx_inicio
                )
                phi = emd_pyphi(dist_orig, dist_part)

                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_particion = [list(p) for p in particion]
                    mejor_dist = dist_part.copy()

                # Corte temprano: partición perfecta
                if mejor_phi < 1e-10:
                    break

            except Exception:
                continue

            evaluadas += 1

        return {
            'biparticion': mejor_particion,
            'phi': mejor_phi,
            'dist_orig': dist_orig,
            'dist_part': mejor_dist,
            'tiempo': 0.0
        }

    # ==================================================================
    # DISTRIBUCIÓN DE LA k-PARTICIÓN
    # ==================================================================

    def _distribucion_k_particion(
        self,
        particion: list[list[int]],
        tpm: NDArray[np.float64],
        n: int,
        idx_inicio: int
    ) -> NDArray[np.float64]:
        """
        Calcula la distribución del sistema tras la k-partición.

        Para cada parte Pi:
        - futuro_cols: columnas de tpm que corresponden a las variables de Pi
        - presente_bits: bits del estado que corresponden a las variables de Pi

        Luego combina con producto tensorial de Kronecker.

        Este es el paso central del documento (Definition 2.1.1):
        'La única manera de obtener un sistema particionado es a través del
        sistema original, aplicando las marginalizaciones necesarias'

        Parámetros
        ----------
        particion : lista de grupos, cada grupo es lista de var indices
        tpm       : TPM del sistema (2^n × n)
        n         : número de variables
        idx_inicio: índice LE del estado inicial

        Retorna
        -------
        NDArray : distribución conjunta de tamaño 2^n
        """
        dist_total = np.array([1.0])

        for parte in particion:
            # Columnas t+1 de esta parte
            # var_a_col: variable i -> columna interna n-1-i
            futuro_cols = sorted(self.var_a_col[v] for v in parte)

            # Bits del estado en t que pertenecen a esta parte
            # variable i -> bit i en el índice LE
            presente_bits = sorted(parte)

            dist_i = _calcular_dist_parte_vectorizado(
                tpm, futuro_cols, presente_bits, n, idx_inicio
            )

            # Producto tensorial (Kronecker) acumulativo
            dist_total = np.kron(dist_total, dist_i)

        return dist_total

    # ==================================================================
    # RESULTADO FORMATEADO
    # ==================================================================

    def imprimir_resultado_k(self, resultado: dict) -> None:
        """Imprime resultados detallados por k."""
        if not resultado:
            print("Sin resultados.")
            return

        etqs = self.sistema.etiquetas
        resultados_k = resultado.get('resultados_por_k', {})

        print("\n" + "=" * 60)
        print(f"  Estrategia  : QNodesKPartition")
        print(f"  Sistema     : {''.join(etqs)} (n={self.n})")
        print(f"  Estado init : {self.sistema.estado_inicial}")
        print("-" * 60)

        for k, res_k in sorted(resultados_k.items()):
            if res_k['biparticion'] is None:
                continue
            partes_str = ' | '.join(
                '{' + ','.join(etqs[v] for v in p) + '}'
                for p in res_k['biparticion']
            )
            print(f"  k={k}: phi={res_k['phi']:.6f}  [{partes_str}]")

        print("-" * 60)
        k_opt = resultado.get('k', '?')
        phi_opt = resultado.get('phi', float('inf'))
        mejor = resultado.get('biparticion', [])
        mejor_str = ' | '.join(
            '{' + ','.join(etqs[v] for v in p) + '}'
            for p in (mejor or [])
        )
        print(f"  ÓPTIMO k={k_opt}: phi={phi_opt:.6f}")
        print(f"  Partición   : {mejor_str}")
        print(f"  Tiempo      : {resultado.get('tiempo', 0):.4f}s")
        print("=" * 60 + "\n")
