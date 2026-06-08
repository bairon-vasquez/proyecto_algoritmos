"""
Ejecuta validación formal de KGeoMIP contra casos de referencia.

Flujo:
    1. Carga tests/casos_referencia.json  (generado por generar_casos_prueba.py)
    2. Para cada caso: construye subsistema → ejecuta KGeoMIP → compara con ref
    3. Calcula métricas §5.2.2: tasa_acierto, E_rel, distancia Jaccard, speedup
    4. Imprime reporte y guarda results/reporte_validacion.json

Uso:
    python tests/ejecutar_validacion.py
    python tests/ejecutar_validacion.py --verbose   (detalle por caso)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from src.models.system import System
from src.controllers.strategies.geometric import KGeoMIP
from src.utils.validation import (
    validar_caso,
    reporte_validacion,
    guardar_reporte_json,
    biparticiones_equivalentes,
)


def cargar_casos(ruta: str) -> list:
    """Carga y filtra los casos válidos del archivo JSON de referencia."""
    if not os.path.exists(ruta):
        print(f"\n  Archivo no encontrado: {ruta}")
        print("  Genera primero los casos con: python tests/generar_casos_prueba.py")
        return []

    with open(ruta, encoding='utf-8') as f:
        datos = json.load(f)

    todos  = datos.get('casos', [])
    validos = [c for c in todos if 'referencia' in c and 'error' not in c]

    print(f"\n  Casos de referencia cargados : {len(validos)} / {len(todos)}")
    print(f"  Método de referencia          : {datos.get('metodo_ref', 'N/A')}\n")
    return validos


def ejecutar_validacion(casos: list, verbose: bool = False) -> list:
    """
    Ejecuta KGeoMIP en cada caso y recopila métricas de comparación.
    """
    resultados = []

    for i, caso in enumerate(casos, 1):
        csv      = caso['csv']
        estado   = caso['estado']
        alcance  = caso['alcance']
        mecanismo= caso['mecanismo']
        n_sub    = caso['n_sub']
        ref      = caso['referencia']

        phi_ref  = ref['phi_raw']
        bp_ref   = ref['biparticion']
        t_ref    = ref.get('tiempo_s', 0.0)

        etiqueta = f"[{i:02d}/{len(casos)}] alc={alcance} mec={mecanismo} n={n_sub}"

        if not os.path.exists(csv):
            print(f"  {etiqueta}  OMITIDO (CSV ausente)")
            continue

        try:
            sistema_completo = System.desde_csv(csv, estado)
            subsistema = sistema_completo.construir_subsistema(
                list(alcance), list(mecanismo)
            )

            r = validar_caso(
                sistema    = subsistema,
                estrategia_cls = KGeoMIP,
                phi_ref_raw    = phi_ref,
                bp_ref         = bp_ref,
                n              = n_sub,
                tiempo_ref_s   = t_ref,
            )
            resultados.append(r)

            if verbose:
                estado_str = "OK " if r['acierto_exacto'] else "NOK"
                print(f"  {etiqueta}  {estado_str} | "
                      f"phi_geo={r['phi_test_raw']:.5f} "
                      f"phi_ref={phi_ref:.5f} | "
                      f"E_rel={r['e_rel']*100:.1f}% | "
                      f"J={r['jaccard_dist']:.3f} | "
                      f"t={r['tiempo_s']:.3f}s | "
                      f"sp={r['speedup']:.1f}x")
            else:
                estado_str = "OK" if r['acierto_exacto'] else f"E_rel={r['e_rel']*100:.0f}%"
                print(f"  {etiqueta}  {estado_str}")

        except Exception as exc:
            print(f"  {etiqueta}  ERROR: {exc}")

    return resultados


def main() -> None:
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    ruta_casos = os.path.join(os.path.dirname(__file__), 'casos_referencia.json')
    casos      = cargar_casos(ruta_casos)

    if not casos:
        return

    print(f"{'='*60}")
    print(f"  Validando KGeoMIP en {len(casos)} casos")
    print(f"{'='*60}\n")

    resultados = ejecutar_validacion(casos, verbose=verbose)

    if not resultados:
        print("\n  Sin resultados para reportar.")
        return

    reporte = reporte_validacion(resultados, nombre_estrategia="KGeoMIP")

    ruta_reporte = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'reporte_validacion.json',
    )
    guardar_reporte_json(reporte, ruta_reporte)


if __name__ == '__main__':
    main()
