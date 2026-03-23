import numpy as np
from numpy.typing import NDArray
from src.models.enums.notation import Notation
from src.models.core.ncube import NCube
from src.funcs.base import indices_from_mask, lil_endian


class System:
    """
    Modelo central del framework GeoMIP.

    Encapsula la TPM completa del sistema, el estado inicial y los NCubes
    (uno por cada variable futura). Expone los métodos de transformación
    que usan todas las estrategias: condicionar, substraer, bipartir.
    """

    def __init__(
        self,
        tpm: NDArray[np.float64],
        estado_inicio: str,
        notacion: Notation = Notation.LITTLE_ENDIAN
    ):
        """
        tpm          : matriz (2^n × n) en formato estado-nodo
        estado_inicio: cadena binaria con el estado inicial del sistema
        notacion     : convención de bits usada en todo el sistema
        """
        self.tpm = tpm
        self.estado_inicio = estado_inicio
        self.notacion = notacion
        self.n = len(estado_inicio)
        self.num_states = 2 ** self.n

        # Construir los NCubes: uno por cada variable futura (columna de la TPM)
        self.ncubos: list[NCube] = self._construir_ncubos()

    # ------------------------------------------------------------------
    # Construcción interna
    # ------------------------------------------------------------------

    def _construir_ncubos(self) -> list[NCube]:
        """
        Convierte cada columna de la TPM en un NCube n-dimensional.

        La TPM tiene shape (2^n, n). La columna j contiene P(X_j=1 | estado).
        Cada columna se reorganiza como tensor de shape (2,)*n.
        """
        ncubos = []
        for j in range(self.n):
            # Extraer columna j: probabilidad de X_j=1 para cada estado
            col = self.tpm[:, j]
            # Reorganizar como tensor (2, 2, ..., 2) con n dimensiones
            tensor = col.reshape([2] * self.n)
            ncubos.append(NCube(tensor, list(range(self.n)), self.notacion))
        return ncubos

    # ------------------------------------------------------------------
    # Carga desde CSV
    # ------------------------------------------------------------------

    @classmethod
    def desde_csv(
        cls,
        filepath: str,
        estado_inicio: str,
        notacion: Notation = Notation.LITTLE_ENDIAN
    ) -> "System":
        """
        Carga una TPM desde un archivo CSV y construye el System.

        El CSV debe tener 2^n filas y n columnas (formato estado-nodo),
        donde la columna j contiene P(X_j^{t+1}=1 | estado_t).
        """
        tpm = np.genfromtxt(filepath, delimiter=",")
        # Si el CSV tiene encabezado de texto, ignorar primera fila
        if np.isnan(tpm[0, 0]):
            tpm = tpm[1:, :]
        return cls(tpm, estado_inicio, notacion)

    # ------------------------------------------------------------------
    # Transformaciones del subsistema
    # ------------------------------------------------------------------

    def condicionar(self, indices: list[int]) -> "System":
        """
        Genera un candidato condicionando variables externas al subsistema.

        Las variables en 'indices' se fijan a sus valores en estado_inicio.
        Esto reduce el sistema original al subsistema de interés.
        """
        ncubos_nuevos = []
        for ncubo in self.ncubos:
            # Encontrar qué ejes de este NCube corresponden a las variables a condicionar
            ejes = [
                i for i, var in enumerate(ncubo.variables)
                if var in indices
            ]
            if ejes:
                ncubo = ncubo.condicionar(ejes, self.estado_inicio)
            ncubos_nuevos.append(ncubo)

        nuevo = System.__new__(System)
        nuevo.tpm = self.tpm
        nuevo.estado_inicio = self.estado_inicio
        nuevo.notacion = self.notacion
        nuevo.n = self.n
        nuevo.num_states = self.num_states
        nuevo.ncubos = ncubos_nuevos
        return nuevo

    def substraer(
        self,
        alcance_dims: list[int],
        mecanismo_dims: list[int]
    ) -> "System":
        """
        Deriva el subsistema seleccionando solo las variables de alcance
        y marginalizando las variables del mecanismo que no participan.

        alcance_dims   : índices de variables futuras (columnas a conservar)
        mecanismo_dims : índices de variables presentes (filas del subsistema)
        """
        # Solo conservar los NCubes correspondientes al alcance
        ncubos_sel = [self.ncubos[j] for j in alcance_dims]

        # Marginalizar las variables del mecanismo que NO están en mecanismo_dims
        ncubos_nuevos = []
        for ncubo in ncubos_sel:
            ejes_marginalizar = [
                i for i, var in enumerate(ncubo.variables)
                if var not in mecanismo_dims
            ]
            if ejes_marginalizar:
                ncubo = ncubo.marginalizar(ejes_marginalizar)
            ncubos_nuevos.append(ncubo)

        nuevo = System.__new__(System)
        nuevo.tpm = self.tpm
        nuevo.estado_inicio = self.estado_inicio
        nuevo.notacion = self.notacion
        nuevo.n = len(mecanismo_dims)
        nuevo.num_states = 2 ** nuevo.n
        nuevo.ncubos = ncubos_nuevos
        return nuevo

    def bipartir(
        self,
        alcance: list[int],
        mecanismo: list[int]
    ) -> "System":
        """
        Construye el sistema particionado: separa alcance y mecanismo
        y marginaliza las conexiones cruzadas entre las partes.

        Este es el método que implementa la "ruptura" formal de la partición.
        Para una bipartición (parte_izq, parte_der):
        - Los NCubes de alcance_izq marginalizan las variables de mecanismo_der
        - Los NCubes de alcance_der marginalizan las variables de mecanismo_izq
        """
        ncubos_particionados = []
        for j, ncubo in enumerate(self.ncubos):
            if j in alcance:
                # Marginalizar variables del mecanismo que no están en esta parte
                ejes = [
                    i for i, var in enumerate(ncubo.variables)
                    if var not in mecanismo
                ]
                if ejes:
                    ncubo = ncubo.marginalizar(ejes)
            ncubos_particionados.append(ncubo)

        nuevo = System.__new__(System)
        nuevo.tpm = self.tpm
        nuevo.estado_inicio = self.estado_inicio
        nuevo.notacion = self.notacion
        nuevo.n = self.n
        nuevo.num_states = self.num_states
        nuevo.ncubos = ncubos_particionados
        return nuevo

    def distribucion_marginal(self) -> NDArray[np.float64]:
        """
        Calcula la distribución de probabilidad del sistema para el estado inicial.

        Combina los NCubes mediante producto tensorial y evalúa en estado_inicio.
        El resultado es un vector de probabilidades sobre todos los estados futuros.
        """
        # Obtener el índice del estado inicial
        if self.notacion == Notation.LITTLE_ENDIAN:
            idx_inicio = lil_endian(self.estado_inicio)
        else:
            idx_inicio = int(self.estado_inicio, 2)

        # Producto tensorial de las distribuciones marginales de cada NCube
        dist = np.array([1.0])
        for ncubo in self.ncubos:
            flat = ncubo.aplanar()
            # Valor de probabilidad para el estado inicial en este NCube
            prob_1 = flat[idx_inicio % len(flat)]
            dist = np.kron(dist, np.array([1 - prob_1, prob_1]))

        # Normalizar para asegurar que suma 1
        total = dist.sum()
        if total > 0:
            dist = dist / total
        return dist

    def __repr__(self) -> str:
        return (
            f"System(n={self.n}, "
            f"estado='{self.estado_inicio}', "
            f"ncubos={len(self.ncubos)})"
        )