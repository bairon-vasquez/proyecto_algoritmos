"""
Punto de entrada principal del proyecto GeoMIP.
Ejecuta GeometricSIA paralelo para N=3,5,10,15,20.
"""
import numpy as np
import os
import multiprocessing
from src.models.system import System
from src.controllers.strategies.geometric import GeometricSIA
from src.utils.metrics import RegistroMetricas, Resultado


def verificar_tabla_n3(sistema: System) -> bool:
    """Verifica costos contra Tabla 4.2 del documento."""
    geo = GeometricSIA(sistema)
    geo.tensores = sistema.obtener_tensores()
    geo.tabla_costos = geo._calcular_tabla_costos_paralelo()

    # Tabla 4.2 - los labels del doc usan notación cba
    # donde el primer dígito es C (última variable)
    # Conversión: doc_label -> LE idx
    # doc '100' = C=1,B=0,A=0 -> LE = A*1+B*2+C*4 = 4
    # doc '001' = C=0,B=0,A=1 -> LE = 1
    def doc_a_le(label, n=3):
        # En el doc, label 'xyz' -> x=var_n-1,...,z=var_0 (REVERSED)
        chars = [int(c) for c in label][::-1]
        return sum(chars[i] * (2**i) for i in range(n))

    tabla_doc = {
        ('A','000'):0.0, ('A','100'):0.0, ('A','010'):0.0, ('A','110'):0.0,
        ('A','001'):0.5, ('A','101'):0.375, ('A','011'):0.375, ('A','111'):0.21875,
        ('B','000'):0.0, ('B','100'):0.0, ('B','010'):0.5, ('B','110'):0.375,
        ('B','001'):0.0, ('B','101'):0.0, ('B','011'):0.375, ('B','111'):0.21875,
        ('C','000'):0.0, ('C','100'):0.5, ('C','010'):0.0, ('C','110'):0.375,
        ('C','001'):0.0, ('C','101'):0.375, ('C','011'):0.0, ('C','111'):0.21875,
    }
    var_map = {'A':0,'B':1,'C':2}
    idx_inicio = 0

    print("\n  Verificacion Tabla 4.2:")
    print(f"  {'Transicion':<22}{'Esperado':>10}{'Calculado':>11}{'':>5}")
    print("  " + "-"*50)

    todos_ok = True
    for (var, lbl), esp in tabla_doc.items():
        j = doc_a_le(lbl)
        T = geo.tabla_costos[var_map[var]]
        calc = T['fila_inicio'][j] if isinstance(T, dict) else T[idx_inicio, j]
        ok = abs(calc - esp) < 1e-4
        if not ok: todos_ok = False
        print(f"  t_{var}(000,{lbl})   {esp:>10.5f}  {calc:>10.5f}  {'OK' if ok else 'FAIL'}")
    return todos_ok


def ejecutar_para_n(n, registro, carpeta="data"):
    print(f"\n{'='*58}")
    print(f"  GeometricSIA (paralelo) para N={n}  "
          f"[{multiprocessing.cpu_count()} CPUs disponibles]")
    print(f"{'='*58}")

    estado = "0" * n
    csv_path = os.path.join(carpeta, f"N{n}C.csv")

    if not os.path.exists(csv_path):
        print(f"  AVISO: {csv_path} no encontrado.")
        return

    print(f"  Cargando: {csv_path}")
    sistema = System.desde_csv(csv_path, estado)
    print(f"  {sistema}")

    if n == 3:
        ok = verificar_tabla_n3(sistema)
        print(f"\n  Tabla 4.2: {'CORRECTO ✓' if ok else 'INCORRECTO ✗'}")

    estrategia = GeometricSIA(sistema)
    resultado = estrategia.ejecutar()
    estrategia.imprimir_resultado()

    registro.registrar(Resultado(
        n=n,
        estado_inicial=estado,
        biparticion=resultado['biparticion'],
        phi=resultado['phi'],
        tiempo=resultado['tiempo'],
        estrategia="GeometricSIA-Paralelo"
    ))


def main():
    print("\n" + "="*58)
    print("  PROYECTO GeoMIP - Algoritmo Geometrico Paralelo")
    print("  Analisis y Diseno de Algoritmos 2025C")
    print(f"  CPUs: {multiprocessing.cpu_count()}")
    print("="*58)

    registro = RegistroMetricas(carpeta="results")

    for n in [3, 5, 10, 15, 20]:
        try:
            ejecutar_para_n(n, registro)
        except MemoryError:
            print(f"  ERROR N={n}: Memoria insuficiente.")
        except Exception as e:
            print(f"  ERROR N={n}: {e}")
            import traceback; traceback.print_exc()

    registro.resumen()
    registro.guardar_csv("resultados_geomip.csv")


if __name__ == "__main__":
    main()
