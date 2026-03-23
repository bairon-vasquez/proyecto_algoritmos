import numpy as np
from numpy.typing import NDArray
from src.models.enums.notation import Notation


class NCube:
    """
    Representa una variable futura del sistema como un hipercubo n-dimensional.

    Cada NCube almacena P(X_j^{t+1} | estado_presente) para una variable j.
    Internamente, los datos se guardan como un tensor de shape (2, 2, ..., 2)
    con n dimensiones, donde n es el número de variables del sistema.

    Ejemplo para n=3: shape (2,2,2), donde data[a][b][c] = P(X_j=1 | A=a, B=b, C=c)
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        variables: list[int],
        notation: Notation = Notation.LITTLE_ENDIAN
    ):
        """
        data      : tensor de shape (2,)*n con las probabilidades
        variables : índices de las variables que indexan este tensor
        notation  : convención de bits (little o big endian)
        """
        self.data = data.copy()
        self.variables = list(variables)
        self.notation = notation
        self.n = len(variables)

    # ------------------------------------------------------------------
    # Operaciones fundamentales
    # ------------------------------------------------------------------

    def condicionar(
        self,
        indices_condicionados: list[int],
        estado_inicial: str
    ) -> "NCube":
        """
        Condiciona el tensor fijando el valor de ciertas variables
        según el estado inicial dado.

        indices_condicionados : posiciones en self.variables a condicionar
        estado_inicial        : cadena binaria con el estado completo del sistema

        Retorna un nuevo NCube reducido sin esas dimensiones.
        """
        data = self.data.copy()
        # Procesamos de mayor a menor índice para no desplazar ejes
        ejes_a_eliminar = sorted(indices_condicionados, reverse=True)

        for eje in ejes_a_eliminar:
            var_global = self.variables[eje]
            # Leer el bit correspondiente según la notación
            if self.notation == Notation.LITTLE_ENDIAN:
                valor_bit = int(estado_inicial[var_global])
            else:
                valor_bit = int(estado_inicial[-(var_global + 1)])
            # Seleccionar la "rebanada" del tensor en ese eje
            data = np.take(data, valor_bit, axis=eje)

        nuevas_variables = [
            v for i, v in enumerate(self.variables)
            if i not in indices_condicionados
        ]
        return NCube(data, nuevas_variables, self.notation)

    def marginalizar(self, ejes: list[int]) -> "NCube":
        """
        Marginaliza (promedia) el tensor sobre los ejes indicados.

        Se usa cuando una variable del mecanismo no participa en una
        parte de la bipartición: su efecto se integra fuera.

        Retorna un nuevo NCube colapsado en esos ejes.
        """
        data = self.data.copy()
        # Procesamos de mayor a menor para no desplazar índices
        for eje in sorted(ejes, reverse=True):
            data = np.mean(data, axis=eje)

        nuevas_variables = [
            v for i, v in enumerate(self.variables)
            if i not in ejes
        ]
        return NCube(data, nuevas_variables, self.notation)

    def aplanar(self) -> NDArray[np.float64]:
        """
        Convierte el tensor a un vector 1D de tamaño 2^n.
        Usado para calcular la distribución marginal del sistema.
        """
        return self.data.flatten(order='C')

    def copy(self) -> "NCube":
        """Retorna una copia independiente del NCube."""
        return NCube(self.data.copy(), self.variables.copy(), self.notation)

    def contiguous_data(self) -> NDArray[np.float64]:
        """Retorna los datos como array contiguo en memoria (optimización)."""
        return np.ascontiguousarray(self.data)

    def __repr__(self) -> str:
        return f"NCube(vars={self.variables}, shape={self.data.shape})"