import numpy as np
import pandas as pd
from numpy.typing import NDArray


class System:
    """
    Representa el sistema original con su TPM y estado inicial.
    Lee el archivo CSV y expone la matriz en formato estado-estado
    y estado-nodo.
    """

    def __init__(self, tpm: NDArray[np.float64], labels: list[str]):
        """
        tpm    : matriz estado-estado de shape (2^n, 2^n)
        labels : nombres de las variables, ej. ['A', 'B', 'C']
        """
        self.tpm = tpm
        self.labels = labels
        self.n = len(labels)
        self.num_states = 2 ** self.n

    # ------------------------------------------------------------------
    # Lectura desde CSV
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, filepath: str) -> "System":
        """
        Carga una TPM desde un archivo CSV.
        El CSV debe tener 2^n filas y 2^n columnas (sin encabezado de índice).
        Los nombres de columnas son los labels de los nodos en t+1.
        """
        df = pd.read_csv(filepath, header=0)
        labels_raw = list(df.columns)

        # Extraer solo los nombres base (quitar sufijo t+1 si existe)
        labels = [l.replace("t+1", "").replace("_t1", "").strip() for l in labels_raw]

        tpm = df.values.astype(np.float64)
        return cls(tpm, labels)

    # ------------------------------------------------------------------
    # Conversión a formato estado-nodo
    # ------------------------------------------------------------------

    def to_state_node(self) -> NDArray[np.float64]:
        """
        Convierte la TPM de formato estado-estado (2^n x 2^n)
        a formato estado-nodo (2^n x n).

        Cada columna j representa P(X_j^{t+1} = 1 | estado_t).
        """
        n = self.n
        num_states = self.num_states
        state_node = np.zeros((num_states, n), dtype=np.float64)

        for j in range(n):
            # Para la variable j, sumar probabilidades de los estados
            # donde el bit j vale 1
            for s in range(num_states):
                prob = 0.0
                for next_s in range(num_states):
                    # Bit j del estado next_s (orden big-endian)
                    bit = (next_s >> (n - 1 - j)) & 1
                    if bit == 1:
                        prob += self.tpm[s, next_s]
                state_node[s, j] = prob

        return state_node

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_state_label(self, index: int) -> str:
        """Convierte un índice entero a su representación binaria."""
        return format(index, f"0{self.n}b")

    def get_state_index(self, binary_str: str) -> int:
        """Convierte una cadena binaria a índice entero."""
        return int(binary_str, 2)

    def apply_background_condition(
        self,
        background_vars: list[int],
        background_state: list[int]
    ) -> "System":
        """
        Condiciona la TPM filtrando solo las filas donde las variables
        de fondo (background_vars) tienen los valores (background_state).

        background_vars  : índices de las variables externas
        background_state : valores (0 o 1) de esas variables
        """
        mask = np.ones(self.num_states, dtype=bool)

        for var_idx, val in zip(background_vars, background_state):
            for s in range(self.num_states):
                bit = (s >> (self.n - 1 - var_idx)) & 1
                if bit != val:
                    mask[s] = False

        filtered_rows = self.tpm[mask, :]

        # Labels del sistema candidato (sin las variables de fondo)
        candidate_labels = [
            l for i, l in enumerate(self.labels)
            if i not in background_vars
        ]

        return System(filtered_rows, candidate_labels)

    def __repr__(self) -> str:
        return (
            f"System(n={self.n}, "
            f"labels={self.labels}, "
            f"shape={self.tpm.shape})"
        )