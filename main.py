"""
Punto de entrada principal del proyecto GeoMIP.
GeometricSIA (k=2) + QNodesKPartition (k=3,4,5) para N=3,5,10,15.

phi normalizado: dividir EMD raw por n -> garantiza rango [0, 1]
Demostración: max distancia Hamming entre n bits = n,
por tanto max EMD = n, y phi = EMD/n está en [0, 1].
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
    """Verifica costos calculados contra la Tabla 4.2 del documento."""
    geo = GeometricSIA(sistema)
    geo.tensores = sistema.obtener_tensores()
    geo.tabla_costos = geo._calcular_tabla_costos_paralelo()

    def doc_le(label: str) -> int:
        return int(label[0])*1 + int(label[1])*2 + int(label[2])*4

    tabla_doc = {
        ('A','000'):0.0,    ('A','100'):0.0,    ('A','010'):0.0,
        ('A','110'):0.0,    ('A','001'):0.5,    ('A','101'):0.375,
        ('A','011'):0.375,  ('A','111'):0.21875,
        ('B','000'):0.0,    ('B','100'):0.0,    ('B','010'):0.5,
        ('B','110'):0.375,  ('B','001'):0.0,    ('B','101'):0.0,
        ('B','011'):0.375,  ('B','111'):0.21875,
        ('C','000'):0.0,    ('C','100'):0.5,    ('C','010'):0.0,
        ('C','110'):0.375,  ('C','001'):0.0,    ('C','101'):0.375,
        ('C','011'):0.0,    ('C','111'):0.21875,
    }
    var_map    = {'A': 0, 'B': 1, 'C': 2}
    idx_inicio = 0

    print("\n  Verificacion Tabla 4.2:")
    print(f"  {'Transicion':<22}{'Esperado':>10}{'Calculado':>11}{'':>5}")
    print("  " + "-"*51)

    todos_ok = True
    for (var, lbl), esp in tabla_doc.items():
        j    = doc_le(lbl)
        T    = geo.tabla_costos[var_map[var]]
        calc = T['fila_inicio'][j] if isinstance(T, dict) else T[idx_inicio, j]
        ok   = abs(calc - esp) < 1e-4
        if not ok: todos_ok = False
        print(f"  t_{var}(000,{lbl})   {esp:>10.5f}  {calc:>10.5f}  {'OK' if ok else 'FAIL'}")
    return todos_ok


def ejecutar_geometric(n: int, registro: RegistroMetricas, carpeta: str = "data"):
    """Ejecuta GeometricSIA (k=2) con phi normalizado en [0,1]."""
    estado   = "0" * n
    csv_path = os.path.join(carpeta, f"N{n}C.csv")
    if not os.path.exists(csv_path):
        print(f"  AVISO: {csv_path} no encontrado.")
        return

    sistema = System.desde_csv(csv_path, estado)

    if n == 3:
        ok = verificar_tabla_n3(sistema)
        print(f"\n  Tabla 4.2: {'CORRECTO' if ok else 'INCORRECTO'}")

    estrategia = GeometricSIA(sistema)

    # Medir tiempo real incluyendo tabla T + EMD
    t0 = time.time()
    resultado = estrategia.aplicar_estrategia()
    resultado['tiempo'] = time.time() - t0
    estrategia.resultado = resultado

    # Normalizar phi: dividir por n para obtener [0, 1]
    phi_raw  = resultado['phi']
    phi_norm = phi_raw / n
    resultado['phi']     = phi_norm
    resultado['phi_raw'] = phi_raw

    # Imprimir resultado con phi normalizado
    p1, p2   = resultado['biparticion']
    etqs     = sistema.etiquetas
    nombres1 = [etqs[i] for i in p1]
    nombres2 = [etqs[i] for i in p2]
    print(f"\n{'='*55}")
    print(f"  Estrategia  : GeometricSIA")
    print(f"  Sistema     : {''.join(etqs)} (n={n})")
    print(f"  Estado init : {estado}")
    print(f"  Biparticion : {nombres1} | {nombres2}")
    print(f"  Phi [0,1]   : {phi_norm:.6f}  (raw={phi_raw:.4f})")
    print(f"  Tiempo      : {resultado['tiempo']:.4f}s")
    print(f"{'='*55}\n")

    registro.registrar(Resultado(
        n=n, estado_inicial=estado,
        biparticion=resultado['biparticion'],
        phi=phi_norm, tiempo=resultado['tiempo'],
        estrategia="GeometricSIA-k2"
    ))


def ejecutar_qnodes(n: int, registro: RegistroMetricas, carpeta: str = "data"):
    """Ejecuta QNodesKPartition (k=3,4,5) con phi normalizado en [0,1]."""
    estado   = "0" * n
    csv_path = os.path.join(carpeta, f"N{n}C.csv")
    if not os.path.exists(csv_path):
        print(f"  AVISO: {csv_path} no encontrado.")
        return

    sistema = System.desde_csv(csv_path, estado)
    print(f"\n  {'='*58}")
    print(f"  QNodesKPartition para N={n}  (k=3,4,5)")
    print(f"  {'='*58}")
    print(f"  {sistema}")

    estrategia = QNodesKPartition(sistema)
    t0         = time.time()
    resultado  = estrategia.aplicar_estrategia()
    resultado['tiempo'] = time.time() - t0
    estrategia.resultado = resultado
    estrategia.imprimir_resultado_k(resultado)

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
    print(f"  phi normalizado: EMD_raw / n -> rango [0, 1]")
    print("="*60)

    reg_geo  = RegistroMetricas(carpeta="results")
    reg_qnod = RegistroMetricas(carpeta="results")

    for n in [3, 5, 10, 15]:
        print(f"\n{'#'*60}")
        print(f"#  N = {n}")
        print(f"{'#'*60}")

        try:
            print("\n>>> GeometricSIA (k=2):")
            ejecutar_geometric(n, reg_geo)
        except Exception as e:
            print(f"  ERROR GeometricSIA N={n}: {e}")
            import traceback; traceback.print_exc()

        try:
            print("\n>>> QNodesKPartition (k=3,4,5):")
            ejecutar_qnodes(n, reg_qnod)
        except Exception as e:
            print(f"  ERROR QNodes N={n}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("  RESUMEN GeometricSIA (k=2)  [phi normalizado en [0,1]]")
    reg_geo.resumen()
    reg_geo.guardar_csv("resultados_geometric.csv")

    print("\n  RESUMEN QNodesKPartition (k=3,4,5)  [phi normalizado en [0,1]]")
    reg_qnod.resumen()
    reg_qnod.guardar_csv("resultados_qnodes.csv")


if __name__ == "__main__":
    main()
