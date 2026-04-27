"""
Punto de entrada principal del proyecto GeoMIP.

Ejecuta el algoritmo GeometricSIA sobre sistemas de N=3,5,10,15,20
nodos, registrando tiempos y resultados para cada tamaño.
"""
import numpy as np
import os
from src.models.system import System
from src.controllers.strategies.geometric import GeometricSIA
from src.utils.metrics import RegistroMetricas, Resultado


def generar_tpm_aleatoria(n: int, semilla: int = 42) -> np.ndarray:
    """
    Genera una TPM estado-nodo aleatoria para un sistema de n variables.
    Usada cuando no se tiene archivo CSV para ese tamaño.
    Cada columna contiene probabilidades entre 0 y 1.
    """
    rng = np.random.default_rng(semilla)
    num_estados = 2 ** n
    tpm = rng.random((num_estados, n)).astype(np.float64)
    return tpm


def ejecutar_para_n(
    n: int,
    registro: RegistroMetricas,
    carpeta_datos: str = "data"
) -> None:
    """
    Ejecuta GeometricSIA para un sistema de n variables.
    Intenta cargar desde CSV; si no existe, genera TPM aleatoria.
    """
    print(f"\n{'='*55}")
    print(f"  Ejecutando GeometricSIA para N={n}")
    print(f"{'='*55}")

    # Estado inicial: todos en 0
    estado_inicial = "0" * n

    # Intentar cargar desde CSV
    ruta_csv = os.path.join(carpeta_datos, f"N{n}C.csv")
    if os.path.exists(ruta_csv):
        print(f"  Cargando TPM desde: {ruta_csv}")
        sistema = System.desde_csv(ruta_csv, estado_inicial)
    else:
        print(f"  CSV no encontrado. Generando TPM aleatoria (n={n})")
        tpm = generar_tpm_aleatoria(n)
        sistema = System(tpm, estado_inicial)

    print(f"  Sistema: {sistema}")

    # Ejecutar estrategia
    estrategia = GeometricSIA(sistema)
    resultado = estrategia.ejecutar()
    estrategia.imprimir_resultado()

    # Registrar métricas
    registro.registrar(Resultado(
        n=n,
        estado_inicial=estado_inicial,
        biparticion=resultado['biparticion'],
        phi=resultado['phi'],
        tiempo=resultado['tiempo'],
        estrategia="GeometricSIA"
    ))


def main():
    print("\n" + "=" * 55)
    print("  PROYECTO GeoMIP — Algoritmo Geométrico")
    print("  Análisis y Diseño de Algoritmos 2025C")
    print("=" * 55)

    # Tamaños a evaluar según el documento
    tamanos = [3, 5, 10, 15, 20]

    registro = RegistroMetricas(carpeta="results")

    for n in tamanos:
        try:
            ejecutar_para_n(n, registro)
        except Exception as e:
            print(f"  ERROR en N={n}: {e}")
            import traceback
            traceback.print_exc()

    # Resumen final
    registro.resumen()
    registro.guardar_csv("resultados_geomip.csv")


if __name__ == "__main__":
    main()