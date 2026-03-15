from abc import ABC, abstractmethod
from src.models.system import System


class SIA(ABC):
    """
    Interfaz común para todas las estrategias de bipartición óptima.
    SIA = System Integration Analysis.
    
    Cualquier nueva estrategia debe heredar de esta clase
    e implementar el método aplicar_estrategia().
    """

    def __init__(self, system: System):
        self.system = system
        self.result = None

    @abstractmethod
    def aplicar_estrategia(self) -> dict:
        """
        Ejecuta el algoritmo de búsqueda de bipartición óptima.

        Debe retornar un diccionario con al menos:
        {
            'biparticion': ({vars_t1}, {vars_t}),  # la bipartición encontrada
            'phi':         float,                   # valor EMD de la bipartición
            'tiempo':      float                    # segundos de ejecución
        }
        """
        pass

    def ejecutar(self) -> dict:
        """Punto de entrada unificado. Llama a aplicar_estrategia()."""
        import time
        inicio = time.time()
        resultado = self.aplicar_estrategia()
        resultado["tiempo"] = time.time() - inicio
        self.result = resultado
        return resultado

    def imprimir_resultado(self) -> None:
        """Imprime el resultado de forma legible."""
        if self.result is None:
            print("Aún no se ha ejecutado la estrategia.")
            return

        print("\n" + "=" * 50)
        print(f"  Estrategia : {self.__class__.__name__}")
        print(f"  Bipartición: {self.result.get('biparticion')}")
        print(f"  Phi (EMD)  : {self.result.get('phi'):.6f}")
        print(f"  Tiempo     : {self.result.get('tiempo'):.4f} s")
        print("=" * 50 + "\n")