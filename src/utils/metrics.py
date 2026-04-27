import time
import csv
import os
from dataclasses import dataclass, field


@dataclass
class Resultado:
    n: int
    estado_inicial: str
    biparticion: tuple
    phi: float
    tiempo: float
    estrategia: str = "GeometricSIA"


class RegistroMetricas:
    """
    Registra tiempos y resultados para cada tamaño N del sistema.
    Guarda los resultados en la carpeta results/.
    """

    def __init__(self, carpeta: str = "results"):
        self.carpeta = carpeta
        self.resultados: list[Resultado] = []
        os.makedirs(carpeta, exist_ok=True)

    def registrar(self, resultado: Resultado) -> None:
        self.resultados.append(resultado)
        self._imprimir(resultado)

    def _imprimir(self, r: Resultado) -> None:
        p1, p2 = r.biparticion
        print(f"  N={r.n:>2} | estado={r.estado_inicial} | "
              f"bipartición={list(p1)}|{list(p2)} | "
              f"phi={r.phi:.6f} | tiempo={r.tiempo:.4f}s")

    def guardar_csv(self, nombre: str = "resultados.csv") -> None:
        ruta = os.path.join(self.carpeta, nombre)
        with open(ruta, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'N', 'estado_inicial', 'parte1',
                'parte2', 'phi', 'tiempo_s', 'estrategia'
            ])
            for r in self.resultados:
                p1, p2 = r.biparticion
                writer.writerow([
                    r.n, r.estado_inicial,
                    str(list(p1)), str(list(p2)),
                    f"{r.phi:.8f}", f"{r.tiempo:.6f}",
                    r.estrategia
                ])
        print(f"\nResultados guardados en: {ruta}")

    def resumen(self) -> None:
        if not self.resultados:
            print("Sin resultados aún.")
            return
        print("\n" + "=" * 65)
        print(f"{'N':>4} | {'Tiempo(s)':>10} | {'Phi':>10} | Bipartición")
        print("-" * 65)
        for r in self.resultados:
            p1, p2 = r.biparticion
            print(f"{r.n:>4} | {r.tiempo:>10.4f} | "
                  f"{r.phi:>10.6f} | {list(p1)} | {list(p2)}")
        print("=" * 65)