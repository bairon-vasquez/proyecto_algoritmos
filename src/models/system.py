import numpy as np
from numpy.typing import NDArray
from itertools import combinations


class System:
    """
    Representa el sistema con su TPM y estado inicial.

    Implementa las tres operaciones fundamentales del documento 1:
    - Condicionamiento: fijar variables externas al subsistema
    - Marginalización: eliminar filas (t) o columnas (t+1) de la TPM
    - Producto tensorial: combinar matrices individuales por variable
    """

    def __init__(
        self,
        tpm: NDArray[np.float64],
        estado_inicial: str,
        etiquetas: list[str] | None = None
    ):
        """
        tpm           : matriz estado-nodo de shape (2^n, n)
                        cada columna j = P(Xj_t+1=1 | estado_t)
        estado_inicial: cadena binaria ej. "000", "1000"
        etiquetas     : nombres de variables ej. ['A','B','C']
        """
        self.tpm = tpm.copy()
        self.estado_inicial = estado_inicial
        self.n = len(estado_inicial)
        self.num_estados = 2 ** self.n
        self.etiquetas = etiquetas or [
            chr(ord('A') + i) for i in range(self.n)
        ]

    # ------------------------------------------------------------------
    # Carga desde CSV
    # ------------------------------------------------------------------

    @classmethod
    def desde_csv(
        cls,
        filepath: str,
        estado_inicial: str
    ) -> "System":
        """
        Carga la TPM desde un archivo CSV en formato estado-nodo.

        El CSV tiene 2^n filas y n columnas.
        Si tiene encabezado de texto se ignora automáticamente.
        """
        import pandas as pd
        df = pd.read_csv(filepath, header=None)

        # Detectar si la primera fila es encabezado de texto
        primera_fila = df.iloc[0]
        try:
            primera_fila.astype(float)
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

        tpm = df.values.astype(np.float64)

        # Si la TPM está en formato estado-estado (2^n x 2^n),
        # convertirla a estado-nodo (2^n x n)
        n = len(estado_inicial)
        if tpm.shape[1] == 2 ** n:
            tpm = cls._estado_estado_a_nodo(tpm, n)

        etiquetas = [chr(ord('A') + i) for i in range(n)]
        return cls(tpm, estado_inicial, etiquetas)

    @staticmethod
    def _estado_estado_a_nodo(
        tpm_ee: NDArray[np.float64],
        n: int
    ) -> NDArray[np.float64]:
        """
        Convierte TPM formato estado-estado (2^n x 2^n)
        a formato estado-nodo (2^n x n).

        Columna j del resultado = P(Xj=1 | estado_t), sumando
        todos los estados futuros donde el bit j vale 1.
        """
        num_states = 2 ** n
        tpm_nodo = np.zeros((num_states, n), dtype=np.float64)
        for j in range(n):
            for s in range(num_states):
                prob = 0.0
                for next_s in range(num_states):
                    # Bit j del estado next_s (orden little-endian)
                    bit = (next_s >> j) & 1
                    if bit == 1:
                        prob += tpm_ee[s, next_s]
                tpm_nodo[s, j] = prob
        return tpm_nodo

    # ------------------------------------------------------------------
    # Operación 1: Condicionamiento (Background Conditions)
    # ------------------------------------------------------------------

    def condicionar(
        self,
        indices_externos: list[int],
        valores: list[int]
    ) -> "System":
        """
        Condiciona la TPM fijando las variables externas a sus valores
        en el estado inicial. Genera el Sistema Candidato.

        Ejemplo del documento (Example 1.2):
        V={A,B,C,D}, estado inicial=1000, candidato={A,B,C}
        → condicionar con D=0 (índice 3, valor 0)

        Proceso:
        1. Filtrar filas donde las variables externas tienen los valores dados
        2. Eliminar las columnas de las variables externas (marginalizar t+1)
        """
        # Paso 1: filtrar filas
        mascara = np.ones(self.num_estados, dtype=bool)
        for idx, val in zip(indices_externos, valores):
            for s in range(self.num_estados):
                bit = (s >> idx) & 1
                if bit != val:
                    mascara[s] = False

        tpm_filtrada = self.tpm[mascara, :]

        # Paso 2: eliminar columnas de variables externas
        cols_a_conservar = [
            j for j in range(self.n)
            if j not in indices_externos
        ]
        tpm_condicionada = tpm_filtrada[:, cols_a_conservar]

        # Construir nuevo estado inicial sin las variables externas
        nuevo_estado = ''.join(
            self.estado_inicial[i]
            for i in range(self.n)
            if i not in indices_externos
        )
        nuevas_etiquetas = [
            self.etiquetas[i]
            for i in range(self.n)
            if i not in indices_externos
        ]

        return System(tpm_condicionada, nuevo_estado, nuevas_etiquetas)

    # ------------------------------------------------------------------
    # Operación 2a: Marginalización por filas (eliminar variables en t)
    # ------------------------------------------------------------------

    def marginalizar_filas(
        self,
        indices_a_eliminar: list[int]
    ) -> "System":
        """
        Marginaliza eliminando variables del tiempo t (filas).

        Proceso del documento (Example 1.3):
        1. Descartar las filas de los elementos no deseados
        2. Agrupar estados resultantes coincidentes promediando

        indices_a_eliminar: posiciones de las variables a quitar de t
        """
        n_orig = self.n
        # Variables que quedan en t
        vars_restantes = [
            i for i in range(n_orig)
            if i not in indices_a_eliminar
        ]
        n_nuevo = len(vars_restantes)
        num_estados_nuevo = 2 ** n_nuevo

        # Construir nueva TPM promediando filas con mismo estado reducido
        tpm_nueva = np.zeros(
            (num_estados_nuevo, self.tpm.shape[1]),
            dtype=np.float64
        )
        conteos = np.zeros(num_estados_nuevo, dtype=int)

        for s in range(self.num_estados):
            # Calcular el índice reducido de este estado
            idx_reducido = 0
            for pos, var in enumerate(vars_restantes):
                bit = (s >> var) & 1
                idx_reducido |= (bit << pos)
            tpm_nueva[idx_reducido] += self.tpm[s]
            conteos[idx_reducido] += 1

        # Promediar (tal como indica el documento: "promedio por índice")
        for i in range(num_estados_nuevo):
            if conteos[i] > 0:
                tpm_nueva[i] /= conteos[i]

        nuevo_estado = ''.join(
            self.estado_inicial[i] for i in vars_restantes
        )
        nuevas_etiquetas = [self.etiquetas[i] for i in vars_restantes]

        return System(tpm_nueva, nuevo_estado, nuevas_etiquetas)

    # ------------------------------------------------------------------
    # Operación 2b: Marginalización por columnas (eliminar variables en t+1)
    # ------------------------------------------------------------------

    def marginalizar_columnas(
        self,
        indices_a_eliminar: list[int]
    ) -> "System":
        """
        Marginaliza eliminando variables del tiempo t+1 (columnas).

        Proceso del documento (Example 1.4):
        Descartar las columnas correspondientes. No requiere promediado.

        indices_a_eliminar: posiciones de las variables a quitar de t+1
        """
        cols_a_conservar = [
            j for j in range(self.n)
            if j not in indices_a_eliminar
        ]
        tpm_nueva = self.tpm[:, cols_a_conservar]
        nuevas_etiquetas = [
            self.etiquetas[j] for j in cols_a_conservar
        ]
        return System(tpm_nueva, self.estado_inicial, nuevas_etiquetas)

    # ------------------------------------------------------------------
    # Operación 3: Producto Tensorial
    # ------------------------------------------------------------------

    def producto_tensorial(
        self,
        other: "System"
    ) -> NDArray[np.float64]:
        """
        Producto tensorial de dos sistemas (Definition 1.2.1).

        El número de filas se mantiene constante.
        Las columnas se multiplican elemento a elemento por fila.

        M3 = M1 ⊗ M2 donde M3[i] = producto externo de M1[i] y M2[i]

        IMPORTANTE: esto NO es el producto de Kronecker estándar.
        """
        assert self.tpm.shape[0] == other.tpm.shape[0], \
            "Ambas TPMs deben tener el mismo número de filas"

        num_filas = self.tpm.shape[0]
        n1 = self.tpm.shape[1]
        n2 = other.tpm.shape[1]
        resultado = np.zeros((num_filas, n1 * n2), dtype=np.float64)

        for i in range(num_filas):
            # Producto externo por fila: mantiene filas, multiplica columnas
            col_izq = self.tpm[i]    # shape (n1,)
            col_der = other.tpm[i]   # shape (n2,)
            resultado[i] = np.outer(col_izq, col_der).flatten()

        return resultado

    # ------------------------------------------------------------------
    # Distribución de probabilidad del estado inicial
    # ------------------------------------------------------------------

    def distribucion_estado_inicial(self) -> NDArray[np.float64]:
        """
        Calcula P(ABCt+1 | estado_inicial) usando independencia condicional.

        Según el Teorema 1.2.1:
        P(AB...Zt+1 | AB...Zt) = P(At+1|...) × P(Bt+1|...) × ... × P(Zt+1|...)

        El resultado es un vector de 2^n probabilidades sobre estados futuros,
        construido con el producto tensorial de las distribuciones individuales.
        """
        # Obtener índice del estado inicial (little-endian)
        idx = int(self.estado_inicial[::-1], 2)

        # La fila del estado inicial en la TPM da P(Xj=1) para cada variable j
        probs_1 = self.tpm[idx]  # shape (n,)

        # Construir distribución conjunta por producto tensorial
        dist = np.array([1.0])
        for j in range(self.n):
            p1 = probs_1[j]
            p0 = 1.0 - p1
            dist = np.kron(dist, np.array([p0, p1]))

        return dist

    # ------------------------------------------------------------------
    # Tensores elementales (para el algoritmo geométrico)
    # ------------------------------------------------------------------

    def obtener_tensores(self) -> list[NDArray[np.float64]]:
        """
        Descompone la TPM en n tensores elementales.

        Cada tensor elemental corresponde a una variable en t+1.
        Para la variable j: tensor[j][estado] = P(Xj=1 | estado_t)

        Retorna lista de n vectores de tamaño 2^n.
        """
        return [self.tpm[:, j].copy() for j in range(self.n)]

    # ------------------------------------------------------------------
    # Subsistema particionado (para evaluación EMD)
    # ------------------------------------------------------------------

    def construir_subsistema_particion(
        self,
        parte1_futuro: list[int],
        parte1_presente: list[int],
        parte2_futuro: list[int],
        parte2_presente: list[int]
    ) -> NDArray[np.float64]:
        """
        Construye el sistema particionado y calcula su distribución.

        Para una bipartición (M1/P1) vs (M2/P2):
        - Sistema parte 1: marginaliza variables de P2 de las filas,
          mantiene solo columnas de M1
        - Sistema parte 2: marginaliza variables de P1 de las filas,
          mantiene solo columnas de M2
        - Combina ambas partes con producto tensorial

        Retorna vector de distribución de probabilidad del sistema partido.
        """
        # Parte 1: alcance=parte1_futuro, mecanismo=parte1_presente
        vars_eliminar_p1 = [
            i for i in range(self.n)
            if i not in parte1_presente
        ]
        sys1 = self.marginalizar_filas(vars_eliminar_p1)
        sys1 = sys1.marginalizar_columnas(
            [j for j in range(sys1.n) if j not in [
                parte1_futuro.index(f) if f in parte1_futuro else -1
                for f in range(self.n)
                if f in parte1_futuro
            ]]
        )

        # Parte 2: alcance=parte2_futuro, mecanismo=parte2_presente
        vars_eliminar_p2 = [
            i for i in range(self.n)
            if i not in parte2_presente
        ]
        sys2 = self.marginalizar_filas(vars_eliminar_p2)

        # Distribuciones individuales
        dist1 = sys1.distribucion_estado_inicial()
        dist2 = sys2.distribucion_estado_inicial()

        # Producto tensorial de distribuciones (Teorema 1.2.1)
        return np.kron(dist1, dist2)

    def __repr__(self) -> str:
        return (
            f"System(n={self.n}, "
            f"estado='{self.estado_inicial}', "
            f"etiquetas={self.etiquetas}, "
            f"tpm_shape={self.tpm.shape})"
        )