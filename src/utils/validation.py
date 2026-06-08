"""
Módulo de validación — métricas de desempeño §5.2.2 del documento GeoMIP.

Métricas implementadas:
    Tasa de acierto exacto       (hit_rate)
    Error relativo en Φ          E_rel = |Φ_opt - Φ_enc| / Φ_opt
    Distancia estructural        1 - Jaccard(biparticion_enc, biparticion_opt)
    Speedup relativo             S_rel = T_ref / T_test

Umbrales de calidad (§5.2.2, Tabla 5.1):
    Excelente: hit_rate>90%  E_rel<1%   dist_Jaccard<0.1
    Bueno:     hit_rate>80%  E_rel<5%   dist_Jaccard<0.2
    Aceptable: hit_rate>70%  E_rel<10%  dist_Jaccard<0.3
    Insuficiente: por debajo de los umbrales anteriores
"""

import json
import os
import time
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Comparadores de biparticiones
# ─────────────────────────────────────────────────────────────────────────────

def biparticiones_equivalentes(bp1: list, bp2: list) -> bool:
    """
    Dos biparticiones son equivalentes si producen la misma partición de variables.
    Maneja la simetría {S1|S2} == {S2|S1}.

    Soporta biparticiones de longitud 2 (k=2) y también listas de k partes.
    """
    if not bp1 or not bp2:
        return False
    if len(bp1) != len(bp2):
        return False

    sets1 = [frozenset(p) for p in bp1]
    sets2 = [frozenset(p) for p in bp2]

    # k=2: comparar en ambas orientaciones
    if len(bp1) == 2:
        return (sets1[0] == sets2[0] and sets1[1] == sets2[1]) or \
               (sets1[0] == sets2[1] and sets1[1] == sets2[0])

    # k>2: comparar como conjuntos de frozensets (independiente del orden)
    return set(sets1) == set(sets2)


def jaccard_similitud_biparticion(bp1: list, bp2: list) -> float:
    """
    Similitud de Jaccard entre dos biparticiones (valor en [0,1]).
    1 = idénticas, 0 = completamente distintas.

    Para k=2: max(orientación normal, orientación invertida) / 2.
    Para k>2: promedio del matching óptimo greedy.
    """
    if not bp1 or not bp2:
        return 0.0

    def _jaccard_sets(a: set, b: set) -> float:
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 1.0

    sets1 = [set(p) for p in bp1]
    sets2 = [set(p) for p in bp2]

    if len(bp1) == 2 and len(bp2) == 2:
        j_normal = (_jaccard_sets(sets1[0], sets2[0]) +
                    _jaccard_sets(sets1[1], sets2[1])) / 2
        j_flip   = (_jaccard_sets(sets1[0], sets2[1]) +
                    _jaccard_sets(sets1[1], sets2[0])) / 2
        return max(j_normal, j_flip)

    # Para k>2: matching greedy (no óptimo, pero práctico)
    usados = [False] * len(sets2)
    total  = 0.0
    for s1 in sets1:
        mejor = 0.0
        mejor_idx = -1
        for j, s2 in enumerate(sets2):
            if not usados[j]:
                j_val = _jaccard_sets(s1, s2)
                if j_val > mejor:
                    mejor     = j_val
                    mejor_idx = j
        if mejor_idx >= 0:
            usados[mejor_idx] = True
            total += mejor
    return total / len(sets1)


def jaccard_distancia_biparticion(bp1: list, bp2: list) -> float:
    """Distancia estructural: 1 − Jaccard similitud. Rango [0,1]."""
    return 1.0 - jaccard_similitud_biparticion(bp1, bp2)


# ─────────────────────────────────────────────────────────────────────────────
# Error relativo en Φ
# ─────────────────────────────────────────────────────────────────────────────

def error_relativo_phi(phi_encontrado: float, phi_optimo: float) -> float:
    """
    E_rel = |Φ_óptimo − Φ_encontrado| / Φ_óptimo  (§5.2.2, Ecuación 5.2)

    Caso especial: si Φ_óptimo ≈ 0 (bipartición perfecta):
        - Φ_encontrado ≈ 0 → E_rel = 0.0  (también perfecta)
        - Φ_encontrado > 0 → E_rel = 1.0  (peor caso relativo)
    """
    if phi_optimo < 1e-10:
        return 0.0 if phi_encontrado < 1e-10 else 1.0
    return abs(phi_encontrado - phi_optimo) / phi_optimo


# ─────────────────────────────────────────────────────────────────────────────
# Clasificación de calidad
# ─────────────────────────────────────────────────────────────────────────────

def nivel_calidad(tasa_acierto: float,
                  e_rel_prom: float,
                  jaccard_dist_prom: float) -> str:
    """
    Clasifica la implementación según los umbrales de §5.2.2, Tabla 5.1.

    Args:
        tasa_acierto      : fracción de casos con bipartición exactamente correcta
        e_rel_prom        : promedio de E_rel en todos los casos
        jaccard_dist_prom : promedio de 1-Jaccard en todos los casos
    """
    if (tasa_acierto > 0.90 and e_rel_prom < 0.01 and jaccard_dist_prom < 0.10):
        return "Excelente"
    if (tasa_acierto > 0.80 and e_rel_prom < 0.05 and jaccard_dist_prom < 0.20):
        return "Bueno"
    if (tasa_acierto > 0.70 and e_rel_prom < 0.10 and jaccard_dist_prom < 0.30):
        return "Aceptable"
    return "Insuficiente"


# ─────────────────────────────────────────────────────────────────────────────
# Validación de un caso individual
# ─────────────────────────────────────────────────────────────────────────────

def validar_caso(
    sistema,
    estrategia_cls,
    phi_ref_raw: float,
    bp_ref: list,
    n: int,
    tiempo_ref_s: float = 0.0,
) -> dict:
    """
    Ejecuta estrategia_cls sobre sistema y compara con la referencia.

    Args:
        sistema         : objeto System del subsistema a evaluar
        estrategia_cls  : clase estrategia (ej. KGeoMIP) — se instancia aquí
        phi_ref_raw     : Φ óptimo en escala raw (EMD sin normalizar)
        bp_ref          : bipartición óptima de referencia [[vars1], [vars2]]
        n               : número de variables del subsistema
        tiempo_ref_s    : tiempo de la referencia para calcular speedup

    Returns:
        dict con: acierto_exacto, e_rel, jaccard_dist, speedup, phi_test_raw,
                  bp_test, bp_ref, tiempo_s, n
    """
    t0 = time.time()
    estrategia = estrategia_cls(sistema)
    resultado  = estrategia.aplicar_estrategia()
    tiempo_test = time.time() - t0

    bp_test     = resultado.get('biparticion')
    phi_raw_ret = resultado.get('phi', float('inf'))

    # KGeoMIP devuelve phi_raw directamente; normalizar si viniera normalizado
    # (Heurística: si phi > n, definitivamente es raw, si phi <= 1.0 es normalizado)
    phi_test_raw = phi_raw_ret
    if n > 0 and phi_raw_ret <= 1.0 + 1e-9:
        # Podría ser normalizado — detectar por comparación con ref
        # Si ref > 1 y test <= 1: test está normalizado, desnormalizar
        if phi_ref_raw > 1.0 + 1e-9:
            phi_test_raw = phi_raw_ret * n

    acierto  = biparticiones_equivalentes(bp_test, bp_ref) if bp_test else False
    e_rel    = error_relativo_phi(phi_test_raw, phi_ref_raw)
    j_dist   = jaccard_distancia_biparticion(bp_test, bp_ref) if bp_test else 1.0
    speedup  = (tiempo_ref_s / tiempo_test) if tiempo_test > 1e-9 and tiempo_ref_s > 0 else 0.0

    return {
        'acierto_exacto': acierto,
        'e_rel':          e_rel,
        'jaccard_dist':   j_dist,
        'speedup':        round(speedup, 2),
        'phi_test_raw':   phi_test_raw,
        'phi_ref_raw':    phi_ref_raw,
        'bp_test':        bp_test,
        'bp_ref':         bp_ref,
        'tiempo_s':       round(tiempo_test, 6),
        'n':              n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporte agregado de validación
# ─────────────────────────────────────────────────────────────────────────────

def reporte_validacion(resultados: list,
                       nombre_estrategia: str = "KGeoMIP") -> dict:
    """
    Genera y muestra el reporte completo de validación.

    Args:
        resultados          : lista de dicts devueltos por validar_caso()
        nombre_estrategia   : etiqueta de la estrategia evaluada

    Returns:
        dict con métricas agregadas y lista detalle
    """
    if not resultados:
        print("  Sin resultados para reportar.")
        return {'error': 'Sin resultados'}

    n_total  = len(resultados)
    n_exito  = sum(1 for r in resultados if r['acierto_exacto'])
    e_rels   = [r['e_rel']        for r in resultados if r['e_rel'] < float('inf')]
    j_dists  = [r['jaccard_dist'] for r in resultados]
    tiempos  = [r['tiempo_s']     for r in resultados]
    speedups = [r['speedup']      for r in resultados if r.get('speedup', 0) > 0]

    tasa_acierto    = n_exito / n_total
    e_rel_prom      = float(np.mean(e_rels))   if e_rels   else float('inf')
    jaccard_prom    = float(np.mean(j_dists))
    tiempo_prom     = float(np.mean(tiempos))
    speedup_prom    = float(np.mean(speedups)) if speedups else 0.0
    nivel           = nivel_calidad(tasa_acierto, e_rel_prom, jaccard_prom)

    resumen = {
        'estrategia':           nombre_estrategia,
        'casos_evaluados':      n_total,
        'aciertos_exactos':     n_exito,
        'tasa_acierto':         round(tasa_acierto,  4),
        'e_rel_promedio':       round(e_rel_prom,    6),
        'jaccard_dist_promedio': round(jaccard_prom,  4),
        'tiempo_promedio_s':    round(tiempo_prom,   4),
        'speedup_promedio':     round(speedup_prom,  2),
        'nivel_calidad':        nivel,
        'detalle':              resultados,
    }

    ancho = 62
    print("\n" + "=" * ancho)
    print(f"  REPORTE DE VALIDACIÓN — {nombre_estrategia}")
    print("=" * ancho)
    print(f"  Casos evaluados       : {n_total}")
    print(f"  Aciertos exactos      : {n_exito}  ({tasa_acierto*100:.1f}%)")
    print(f"  E_rel promedio        : {e_rel_prom*100:.3f}%")
    print(f"  Dist. Jaccard prom.   : {jaccard_prom:.4f}")
    print(f"  Tiempo promedio       : {tiempo_prom:.4f} s")
    if speedup_prom > 0:
        print(f"  Speedup promedio      : {speedup_prom:.2f}×")
    print(f"  -- NIVEL DE CALIDAD -> {nivel} --")
    print("=" * ancho)

    # Tabla de umbrales
    print("\n  Tabla de umbrales (§5.2.2):")
    encabezado = f"  {'Nivel':<12} {'Tasa':>7} {'E_rel':>8} {'Jaccard':>8}"
    print(encabezado)
    print("  " + "-" * (len(encabezado) - 2))
    umbrales = [
        ("Excelente", ">90%", "<1%",  "<0.1"),
        ("Bueno",     ">80%", "<5%",  "<0.2"),
        ("Aceptable", ">70%", "<10%", "<0.3"),
    ]
    for nom, ta, er, jd in umbrales:
        marca = " <--" if nivel == nom else ""
        print(f"  {nom:<12} {ta:>7} {er:>8} {jd:>8}{marca}")
    print()

    return resumen


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────────────

def guardar_reporte_json(reporte: dict, ruta: str) -> None:
    """Guarda el reporte de validación en JSON con serialización robusta."""

    def _serial(obj: Any) -> Any:
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (frozenset, set)): return sorted(obj)
        raise TypeError(f"Tipo no serializable: {type(obj)}")

    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, default=_serial)
    print(f"  Reporte guardado en: {ruta}")
