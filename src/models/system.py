import numpy as np
import pandas as pd
from numpy.typing import NDArray


class System:
    """
    Representa el sistema con su TPM y estado inicial.

    Convención de bits: A=bit0 (LSB), B=bit1, C=bit2, ...
    Los CSVs del documento usan orden 'cba' (última variable=A).
    El método desde_csv() reordena automáticamente las filas
    para que row i = estado con índice LE i.
    """

    def __init__(
        self,
        tpm: NDArray[np.float64],
        estado_inicial: str,
        etiquetas: list[str] | None = None
    ):
        self.tpm = tpm.copy()
        self.estado_inicial = estado_inicial
        self.n = len(estado_inicial)
        self.num_estados = 2 ** self.n
        self.etiquetas = etiquetas or [
            chr(ord('A') + i) for i in range(self.n)
        ]

    @classmethod
    def desde_csv(cls, filepath: str, estado_inicial: str) -> "System":
        """
        Carga la TPM desde CSV reordenando filas al formato LE.

        El documento ordena los estados como 'cba' (última var = A),
        por lo que el CSV row i no corresponde al estado LE i.
        Esta función reordena para que row i = LE index i.
        """
        df = pd.read_csv(filepath, header=None, low_memory=False)
        primera = df.iloc[0]
        try:
            primera.astype(float)
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

        tpm_doc = df.values.astype(np.float64)
        n = len(estado_inicial)

        # Si TPM es estado-estado (2^n x 2^n), convertir a estado-nodo
        if tpm_doc.shape[1] == 2 ** n and tpm_doc.shape[0] == 2 ** n:
            tpm_doc = cls._ee_a_nodo(tpm_doc, n)

        # Reordenar filas: el doc usa orden 'cba' donde el primer dígito
        # es la ÚLTIMA variable (C para n=3, o la variable n-1 en general).
        # Fila doc_i corresponde al estado donde los bits se leen al revés:
        # doc state 'xyz' -> variable_0=z (LSB), ..., variable_{n-1}=x
        tpm_le = cls._reordenar_doc_a_le(tpm_doc, n)

        etiquetas = [chr(ord('A') + i) for i in range(n)]
        return cls(tpm_le, estado_inicial, etiquetas)

    @staticmethod
    def _reordenar_doc_a_le(
        tpm_doc: NDArray[np.float64],
        n: int
    ) -> NDArray[np.float64]:
        """
        Convierte de orden doc (cba = última var es bit0) a LE (A=bit0).

        En el documento, el estado i-ésimo en el CSV se interpreta como:
        - El dígito MÁS significativo del índice i es la PRIMERA variable
        - Pero la primera variable del doc es la ÚLTIMA (C en n=3)

        Equivalente: doc_row_i -> LE_index = int(bin(i).zfill(n)[::-1], 2)
        """
        num_estados = 2 ** n
        tpm_le = np.zeros_like(tpm_doc)
        for doc_idx in range(num_estados):
            # Convertir índice doc a índice LE
            # doc_idx en binario (n bits), luego invertir los bits
            bits = format(doc_idx, f'0{n}b')  # big-endian bit string
            bits_rev = bits[::-1]              # reverse -> LE bit string
            le_idx = int(bits_rev, 2)
            tpm_le[le_idx] = tpm_doc[doc_idx]
        return tpm_le

    @staticmethod
    def _ee_a_nodo(tpm_ee: NDArray, n: int) -> NDArray:
        num_states = 2 ** n
        tpm_nodo = np.zeros((num_states, n), dtype=np.float64)
        for j in range(n):
            for s in range(num_states):
                for ns in range(num_states):
                    if (ns >> j) & 1:
                        tpm_nodo[s, j] += tpm_ee[s, ns]
        return tpm_nodo

    def condicionar(self, indices_externos, valores):
        mascara = np.ones(self.num_estados, dtype=bool)
        for idx, val in zip(indices_externos, valores):
            for s in range(self.num_estados):
                if (s >> idx) & 1 != val:
                    mascara[s] = False
        tpm_filtrada = self.tpm[mascara, :]
        cols = [j for j in range(self.n) if j not in indices_externos]
        tpm_cond = tpm_filtrada[:, cols]
        nuevo_estado = ''.join(
            self.estado_inicial[i] for i in range(self.n)
            if i not in indices_externos
        )
        nuevas_etqs = [self.etiquetas[i] for i in range(self.n)
                       if i not in indices_externos]
        return System(tpm_cond, nuevo_estado, nuevas_etqs)

    def marginalizar_filas(self, indices_elim):
        vars_rest = [i for i in range(self.n) if i not in indices_elim]
        n_nuevo = len(vars_rest)
        num_nuevo = 2 ** n_nuevo
        tpm_nueva = np.zeros((num_nuevo, self.tpm.shape[1]), dtype=np.float64)
        conteos = np.zeros(num_nuevo, dtype=int)
        for s in range(self.num_estados):
            idx_red = sum(
                ((s >> var) & 1) << pos
                for pos, var in enumerate(vars_rest)
            )
            tpm_nueva[idx_red] += self.tpm[s]
            conteos[idx_red] += 1
        for i in range(num_nuevo):
            if conteos[i] > 0:
                tpm_nueva[i] /= conteos[i]
        nuevo_estado = ''.join(self.estado_inicial[i] for i in vars_rest)
        nuevas_etqs = [self.etiquetas[i] for i in vars_rest]
        return System(tpm_nueva, nuevo_estado, nuevas_etqs)

    def marginalizar_columnas(self, indices_elim):
        cols = [j for j in range(self.n) if j not in indices_elim]
        tpm_nueva = self.tpm[:, cols]
        nuevas_etqs = [self.etiquetas[j] for j in cols]
        return System(tpm_nueva, self.estado_inicial, nuevas_etqs)

    def obtener_tensores(self):
        return [self.tpm[:, j].copy() for j in range(self.n)]

    def distribucion_estado_inicial(self):
        idx = int(self.estado_inicial[::-1], 2)
        probs_1 = self.tpm[idx % self.num_estados]
        dist = np.array([1.0])
        for p1 in probs_1:
            dist = np.kron(dist, np.array([1.0 - p1, p1]))
        return dist

    def __repr__(self):
        return (f"System(n={self.n}, estado='{self.estado_inicial}', "
                f"etiquetas={self.etiquetas}, tpm_shape={self.tpm.shape})")
