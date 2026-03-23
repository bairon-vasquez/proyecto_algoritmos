from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class Solution:
    """
    Encapsula el resultado de encontrar la k-MIP de un subsistema.

    Este objeto es el contrato de salida de todas las estrategias del framework.
    Independientemente del método usado, el resultado siempre se presenta
    en este formato estructurado.
    """

    # La partición encontrada: lista de k grupos, cada grupo es una lista de índices
    # Para bipartición: [[0, 2], [1]] significa parte_1={A,C}, parte_2={B}
    particion: list[list[int]]

    # Valor de pérdida EMD de esta partición (menor es mejor)
    perdida: float

    # Distribución del sistema original (sin partir)
    dist_original: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Distribución del sistema particionado
    dist_particionada: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Nombre de la estrategia que generó esta solución
    estrategia: str = "desconocida"

    # Tiempo de ejecución en segundos
    tiempo: float = 0.0

    # Número k de particiones
    k: int = 2

    # Metadatos adicionales (para debugging o análisis)
    metadatos: dict = field(default_factory=dict)

    def es_valida(self) -> bool:
        """Verifica que la solución tenga una partición bien formada."""
        return (
            len(self.particion) >= 2
            and all(len(parte) > 0 for parte in self.particion)
            and self.perdida >= 0
        )

    def __str__(self) -> str:
        partes = " | ".join(
            "{" + ", ".join(str(v) for v in parte) + "}"
            for parte in self.particion
        )
        return (
            f"Solucion [{self.estrategia}]\n"
            f"  Particion ({self.k} partes): {partes}\n"
            f"  Perdida (EMD): {self.perdida:.6f}\n"
            f"  Tiempo: {self.tiempo:.4f}s"
        )

    def __repr__(self) -> str:
        return f"Solution(k={self.k}, perdida={self.perdida:.6f}, estrategia='{self.estrategia}')"