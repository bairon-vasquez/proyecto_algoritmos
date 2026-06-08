"""
Generador de casos de referencia para validación (reemplaza PyPhi).

KQNodes con k=2 y n<=20 evalúa TODAS las S(n,2) = 2^(n-1)-1 biparticiones
y devuelve la óptima garantizada — es el equivalente funcional de PyPhi
para el problema de bipartición óptima definido en el documento.

Salida: tests/casos_referencia.json

Uso:
    python tests/generar_casos_prueba.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

from src.models.system import System
from src.controllers.strategies.qnodes import KQNodes


# ─────────────────────────────────────────────────────────────────────────────
# Definición de casos de prueba
# ─────────────────────────────────────────────────────────────────────────────

# Casos N=3: todos los subsistemas son ≤3 variables (muy rápido)
CASOS_N3 = [
    ("data/N3C.csv",  "000", "ABC", "ABC"),
    ("data/N3C.csv",  "000", "ABC", "AB"),
    ("data/N3C.csv",  "000", "ABC", "BC"),
    ("data/N3C.csv",  "000", "ABC", "AC"),
    ("data/N3C.csv",  "000", "AB",  "ABC"),
    ("data/N3C.csv",  "000", "BC",  "ABC"),
    ("data/N3C.csv",  "000", "AC",  "ABC"),
    ("data/N3C.csv",  "000", "AB",  "AB"),
    ("data/N3C.csv",  "000", "BC",  "BC"),
]

# Casos N=5: subsistemas hasta 5 variables (S(5,2)=15 biparticiones, <1s por caso)
CASOS_N5 = [
    ("data/N5C.csv",  "00000", "ABCDE", "ABCDE"),
    ("data/N5C.csv",  "00000", "ABCDE", "ABCD"),
    ("data/N5C.csv",  "00000", "ABCDE", "BCDE"),
    ("data/N5C.csv",  "00000", "ABCDE", "ACE"),
    ("data/N5C.csv",  "00000", "ABCD",  "ABCDE"),
    ("data/N5C.csv",  "00000", "BCDE",  "ABCDE"),
    ("data/N5C.csv",  "00000", "ACE",   "ABCDE"),
    ("data/N5C.csv",  "00000", "ABCD",  "ABCD"),
    ("data/N5C.csv",  "00000", "ABC",   "ABCDE"),
    ("data/N5C.csv",  "00000", "ABCDE", "ABC"),
]

# Casos N=10: subsistemas reducidos ≤7 variables para ser manejables como referencia
# (KQNodes k=2 n=7 → S(7,2)=63 biparticiones, <2s por caso)
CASOS_N10_PEQUENOS = [
    ("data/N10C.csv", "1000000000", "ACEGI",   "ACEGI"),
    ("data/N10C.csv", "1000000000", "BDFHJ",   "BDFHJ"),
    ("data/N10C.csv", "1000000000", "ABCDE",   "ABCDE"),
    ("data/N10C.csv", "1000000000", "FGHIJ",   "FGHIJ"),
    ("data/N10C.csv", "1000000000", "ACE",     "ACEGI"),
    ("data/N10C.csv", "1000000000", "ABCDE",   "BDFHJ"),
    ("data/N10C.csv", "1000000000", "ACEGI",   "ABCDE"),
]


def _filtrar_vars(var_str: str, etiquetas: list) -> str:
    """Filtra variables que realmente existen en el sistema."""
    return ''.join(v for v in var_str if v in etiquetas)


def generar_referencia_caso(
    csv_path: str,
    estado: str,
    alcance: str,
    mecanismo: str,
    verbose: bool = True,
) -> dict | None:
    """
    Genera resultado de referencia para un caso usando KQNodes exhaustivo (k=2).

    Devuelve None si el archivo CSV no existe o si el subsistema es trivial (n<2).
    """
    if not os.path.exists(csv_path):
        if verbose:
            print(f"    OMITIDO — CSV no encontrado: {csv_path}")
        return None

    try:
        sistema_completo = System.desde_csv(csv_path, estado)
        etiquetas = sistema_completo.etiquetas

        # Filtrar variables que no existen en el sistema
        alc_filtrado = _filtrar_vars(alcance,   etiquetas)
        mec_filtrado = _filtrar_vars(mecanismo, etiquetas)

        if not alc_filtrado or not mec_filtrado:
            if verbose:
                print(f"    OMITIDO — alcance/mecanismo vacío tras filtro")
            return None

        subsistema = sistema_completo.construir_subsistema(
            list(alc_filtrado), list(mec_filtrado)
        )
        n_sub = subsistema.n

        if n_sub < 2:
            if verbose:
                print(f"    OMITIDO — subsistema trivial n={n_sub}")
            return None

        qn = KQNodes(subsistema)
        t0  = time.time()
        res = qn.aplicar_estrategia()
        t   = time.time() - t0

        # Extraer resultado k=2 (exhaustivo)
        res_k2 = res.get('resultados_por_k', {}).get(2, res)

        phi_norm = float(res_k2.get('phi', float('inf')))
        phi_raw  = float(res_k2.get('phi_raw', phi_norm * n_sub))
        bp       = res_k2.get('biparticion')

        if bp is None:
            if verbose:
                print(f"    FALLO — KQNodes no devolvió bipartición")
            return None

        return {
            'csv':       csv_path,
            'estado':    estado,
            'alcance':   alc_filtrado,
            'mecanismo': mec_filtrado,
            'n_sub':     n_sub,
            'referencia': {
                'biparticion': [list(p) for p in bp],
                'phi_raw':     round(phi_raw,  8),
                'phi_norm':    round(phi_norm, 8),
                'tiempo_s':    round(t,        4),
                'metodo':      'KQNodes_k2_exhaustivo',
            },
        }

    except Exception as exc:
        if verbose:
            print(f"    ERROR: {exc}")
        return {
            'error':     str(exc),
            'csv':       csv_path,
            'estado':    estado,
            'alcance':   alcance,
            'mecanismo': mecanismo,
        }


def main() -> None:
    todos = CASOS_N3 + CASOS_N5 + CASOS_N10_PEQUENOS
    casos = []

    print(f"\n{'='*60}")
    print(f"  Generando {len(todos)} casos de referencia con KQNodes exhaustivo")
    print(f"  (equivalente funcional de PyPhi para el problema MIP)")
    print(f"{'='*60}\n")

    for i, (csv, estado, alc, mec) in enumerate(todos, 1):
        print(f"  [{i:02d}/{len(todos)}]  {csv}  alc={alc}  mec={mec}")
        resultado = generar_referencia_caso(csv, estado, alc, mec, verbose=True)
        if resultado and 'referencia' in resultado:
            ref = resultado['referencia']
            bp  = ref['biparticion']
            print(f"          phi_raw={ref['phi_raw']:.6f}  "
                  f"bipart={bp}  t={ref['tiempo_s']:.3f}s")
            casos.append(resultado)
        elif resultado and 'error' in resultado:
            casos.append(resultado)   # guardar el error para diagnóstico

    salida = os.path.join(os.path.dirname(__file__), 'casos_referencia.json')
    payload = {
        'descripcion': 'Casos de referencia generados con KQNodes k=2 exhaustivo',
        'metodo_ref':  'KQNodes_k2_exacto (S(n,2) biparticiones evaluadas)',
        'total':       len(casos),
        'casos':       casos,
    }

    with open(salida, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    exitosos = sum(1 for c in casos if 'referencia' in c)
    print(f"\n  Casos generados : {exitosos} / {len(todos)}")
    print(f"  Guardado en     : {salida}")
    print()


if __name__ == '__main__':
    main()
