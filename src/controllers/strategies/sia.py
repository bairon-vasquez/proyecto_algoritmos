from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from src.models.system import System


class SIA(ABC):
    """
    Clase base para todas las estrategias de bipartición óptima.
    Toda estrategia hereda de aquí e implementa aplicar_estrategia().
    """

    def __init__(self, sistema: System):
        self.sistema = sistema
        self.resultado = None

    @abstractmethod
    def aplicar_estrategia(self) -> dict:
        """
        Ejecuta el algoritmo y retorna un diccionario con:
        {
            'biparticion': (parte1, parte2),   # tupla de listas de índices
            'phi':         float,              # valor EMD mínimo
            'tiempo':      float,              # segundos de ejecución
            'dist_orig':   np.ndarray,         # distribución original
            'dist_part':   np.ndarray,         # distribución particionada
        }
        """
        pass

    def ejecutar(self) -> dict:
        """Punto de entrada unificado con medición de tiempo."""
        import time
        inicio = time.time()
        resultado = self.aplicar_estrategia()
        resultado['tiempo'] = time.time() - inicio
        self.resultado = resultado
        return resultado

    def imprimir_resultado(self) -> None:
        if self.resultado is None:
            print("Aún no se ha ejecutado.")
            return
        r = self.resultado
        p1, p2 = r['biparticion']
        etqs = self.sistema.etiquetas
        nombre_p1 = [etqs[i] for i in p1]
        nombre_p2 = [etqs[i] for i in p2]
        print("\n" + "=" * 55)
        print(f"  Estrategia  : {self.__class__.__name__}")
        print(f"  Sistema     : {''.join(etqs)} (n={self.sistema.n})")
        print(f"  Estado init : {self.sistema.estado_inicial}")
        print(f"  Bipartición : {nombre_p1} | {nombre_p2}")
        print(f"  Phi (EMD)   : {r['phi']:.6f}")
        print(f"  Tiempo      : {r['tiempo']:.4f} s")
        print("=" * 55 + "\n")