"""
Genera reporte final consolidado de todas las pruebas ejecutadas.
Lee los CSVs de resultados y produce un resumen formateado.
"""
import os
import csv


def leer_csv(path):
    """
    Parsea el CSV de suite de forma robusta: el campo 'particion' puede
    contener comas internas (ej. [4]|[0,1,2,3]), por lo que se extrae
    como todo lo que está entre el 7º campo y los dos últimos.
    Deduplica por (prueba, estrategia) quedándose con el último registro.
    """
    if not os.path.exists(path):
        return []

    vistos = {}  # (prueba, estrategia) -> row dict
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return []

        for raw in reader:
            if len(raw) < 10:
                continue
            prueba        = raw[0].strip()
            n_sub         = raw[1].strip()
            estado_ini    = raw[2].strip()
            alcance       = raw[3].strip()
            mecanismo     = raw[4].strip()
            k             = raw[5].strip()
            estrategia    = raw[6].strip()
            tiempo_s      = raw[-1].strip()
            perdida       = raw[-2].strip()
            particion     = ','.join(raw[7:-2]).strip()

            if estrategia == 'ERROR':
                continue
            try:
                float(perdida)
                float(tiempo_s)
            except ValueError:
                continue

            row = {
                'prueba':       prueba,
                'n_sub':        n_sub,
                'estado_ini':   estado_ini,
                'alcance':      alcance,
                'mecanismo':    mecanismo,
                'k':            k,
                'estrategia':   estrategia,
                'particion':    particion,
                'perdida':      perdida,
                'tiempo_s':     tiempo_s,
            }
            vistos[(prueba, estrategia)] = row  # sobreescribe duplicados

    return sorted(vistos.values(), key=lambda r: (int(r['prueba']) if r['prueba'].isdigit() else 0,
                                                   r['estrategia']))


def imprimir_tabla(filas, titulo):
    if not filas:
        print(f"  {titulo}: sin datos")
        return
    print(f"\n{'='*90}")
    print(f"  {titulo}  ({len(filas)} resultados)")
    print(f"{'='*90}")
    print(f"{'#':>3} {'N':>4} {'Alcance':<16} {'Mecanismo':<16} "
          f"{'k':>3} {'Phi':>10} {'Tiempo(s)':>12} {'Estrategia':<10}")
    print("-"*90)
    for r in filas:
        phi_val = float(r['perdida'])
        t_val   = float(r['tiempo_s'])
        print(f"{r['prueba']:>3} "
              f"{r['n_sub']:>4} "
              f"{r['alcance']:<16} "
              f"{r['mecanismo']:<16} "
              f"{r['k']:>3} "
              f"{phi_val:>10.6f} "
              f"{t_val:>12.4f} "
              f"{r['estrategia'][:20]:<20}")


def main():
    archivos = [
        ("results/resultados_suite_N10.csv", "Suite N=10"),
        ("results/resultados_suite_N15.csv", "Suite N=15"),
        ("results/resultados_geometric.csv", "GeometricSIA modo normal"),
        ("results/resultados_qnodes.csv",    "QNodes modo normal"),
    ]

    print("\n" + "="*90)
    print("  REPORTE FINAL — Proyecto KQNodes / KGeoMIP")
    print("  Analisis y Diseno de Algoritmos 2026-1")
    print("="*90)

    total_resultados = 0
    for path, titulo in archivos:
        filas = leer_csv(path)
        imprimir_tabla(filas, titulo)
        total_resultados += len(filas)

    print(f"\n  Total resultados (deduplicados): {total_resultados}")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
