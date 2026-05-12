"""
Punto de entrada principal del proyecto GeoMIP.
Ejecuta GeometricSIA (k=2) y QNodesKPartition (k=3,4,5) para N=3,5,10,15.
"""
import numpy as np
import os
import time
import multiprocessing
from src.models.system import System
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.qnodes import QNodesKPartition
from src.utils.metrics import RegistroMetricas, Resultado


def verificar_tabla_n3(sistema: System) -> bool:
    """Verifica costos contra Tabla 4.2 del documento 2."""
    geo = GeometricSIA(sistema)
    geo.tensores = sistema.obtener_tensores()
    geo.tabla_costos = geo._calcular_tabla_costos_paralelo()

    def doc_le(label: str) -> int:
        return int(label[0])*1 + int(label[1])*2 + int(label[2])*4

    tabla_doc = {
        ('A','000'):0.0, ('A','100'):0.0, ('A','010'):0.0, ('A','110'):0.0,
        ('A','001'):0.5, ('A','101'):0.375, ('A','011'):0.375, ('A','111'):0.21875,
        ('B','000'):0.0, ('B','100'):0.0, ('B','010'):0.5, ('B','110'):0.375,
        ('B','001'):0.0, ('B','101'):0.0, ('B','011'):0.375, ('B','111'):0.21875,
        ('C','000'):0.0, ('C','100'):0.5, ('C','010'):0.0, ('C','110'):0.375,
        ('C','001'):0.0, ('C','101'):0.375, ('C','011'):0.0, ('C','111'):0.21875,
    }
    var_map = {'A': 0, 'B': 1, 'C': 2}
    idx_inicio = 0

    print("\n  Verificacion Tabla 4.2:")
    print(f"  {'Transicion':<22}{'Esperado':>10}{'Calculado':>11}{'':>5}")
    print("  " + "-"*51)

    todos_ok = True
    for (var, lbl), esp in tabla_doc.items():
        j = doc_le(lbl)
        T = geo.tabla_costos[var_map[var]]
        calc = T['fila_inicio'][j] if isinstance(T, dict) else T[idx_inicio, j]
        ok = abs(calc - esp) < 1e-4
        if not ok:
            todos_ok = False
        print(f"  t_{var}(000,{lbl})   {esp:>10.5f}  {calc:>10.5f}  {'OK' if ok else 'FAIL'}")
    return todos_ok


def ejecutar_geometric(n, registro, carpeta="data"):
    """Ejecuta GeometricSIA (bipartición k=2)."""
    estado = "0" * n
    csv_path = os.path.join(carpeta, f"N{n}C.csv")
    if not os.path.exists(csv_path):
        return

    sistema = System.desde_csv(csv_path, estado)

    if n == 3:
        ok = verificar_tabla_n3(sistema)
        print(f"\n  Tabla 4.2: {'CORRECTO ✓' if ok else 'INCORRECTO ✗'}")

    estrategia = GeometricSIA(sistema)
    resultado = estrategia.ejecutar()
    estrategia.imprimir_resultado()

    registro.registrar(Resultado(
        n=n, estado_inicial=estado,
        biparticion=resultado['biparticion'],
        phi=resultado['phi'], tiempo=resultado['tiempo'],
        estrategia="GeometricSIA-k2"
    ))


def ejecutar_qnodes(n, registro, carpeta="data"):
    """Ejecuta QNodesKPartition (k=3,4,5)."""
    estado = "0" * n
    csv_path = os.path.join(carpeta, f"N{n}C.csv")
    if not os.path.exists(csv_path):
        return

    sistema = System.desde_csv(csv_path, estado)
    print(f"\n  {'='*58}")
    print(f"  QNodesKPartition para N={n}  (k=3,4,5)")
    print(f"  {'='*58}")
    print(f"  {sistema}")

    estrategia = QNodesKPartition(sistema)
    t0 = time.time()
    resultado = estrategia.aplicar_estrategia()
    resultado['tiempo'] = time.time() - t0
    estrategia.resultado = resultado
    estrategia.imprimir_resultado_k(resultado)

    # Registrar el mejor resultado global
    if resultado['biparticion']:
        registro.registrar(Resultado(
            n=n, estado_inicial=estado,
            biparticion=(resultado['biparticion'][0],
                        resultado['biparticion'][-1]),
            phi=resultado['phi'],
            tiempo=resultado['tiempo'],
            estrategia=f"QNodes-k{resultado.get('k','?')}"
        ))


def main():
    ncpus = multiprocessing.cpu_count()
    print("\n" + "="*60)
    print("  PROYECTO GeoMIP")
    print("  GeometricSIA (k=2) + QNodesKPartition (k=3,4,5)")
    print("  Analisis y Diseno de Algoritmos 2026-1")
    print("  Prof. Luz Enith")
    print(f"  CPUs: {ncpus}")
    print("="*60)

    registro_geo  = RegistroMetricas(carpeta="results")
    registro_qnod = RegistroMetricas(carpeta="results")

    for n in [3, 5, 10, 15]:
        print(f"\n{'#'*60}")
        print(f"#  N = {n}")
        print(f"{'#'*60}")

        try:
            print("\n>>> GeometricSIA (k=2):")
            ejecutar_geometric(n, registro_geo)
        except Exception as e:
            print(f"  ERROR GeometricSIA N={n}: {e}")

        try:
            print("\n>>> QNodesKPartition (k=3,4,5):")
            ejecutar_qnodes(n, registro_qnod)
        except Exception as e:
            print(f"  ERROR QNodes N={n}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("  RESUMEN GeometricSIA (k=2)")
    registro_geo.resumen()
    registro_geo.guardar_csv("resultados_geometric.csv")

    print("\n  RESUMEN QNodesKPartition (k=3,4,5)")
    registro_qnod.resumen()
    registro_qnod.guardar_csv("resultados_qnodes.csv")


if __name__ == "__main__":
    main()
