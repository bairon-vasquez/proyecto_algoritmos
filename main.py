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
from src.controllers.strategies.geometric import KGeoMIP, GeometricSIA
from src.controllers.strategies.qnodes import KQNodes, QNodesKPartition
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

    estrategia = KGeoMIP(sistema)

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

    estrategia = KQNodes(sistema)
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


def ejecutar_prueba(
    n: int,
    estado_inicial: str,
    alcance_str: str,
    mecanismo_str: str,
    csv_path: str,
    registro_geo: RegistroMetricas,
    registro_qnod: RegistroMetricas,
    timeout_qnodes: int | None = None,
    salida_path: str | None = None,
    num_prueba: int = 0
) -> None:
    """
    Ejecuta una prueba individual con alcance y mecanismo específicos.

    n             : número de variables del sistema completo
    estado_inicial: cadena binaria ej. '1000000000'
    alcance_str   : variables en t+1 ej. 'ABCDEFGHIJ'
    mecanismo_str : variables en t   ej. 'ABCDEFGHI'
    csv_path      : ruta al CSV del sistema completo
    """
    # Cargar sistema o crear sintético cuando el CSV no existe (N grandes)
    if os.path.exists(csv_path):
        sistema_completo = System.desde_csv(csv_path, estado_inicial)
        subsistema = sistema_completo.construir_subsistema(
            list(alcance_str), list(mecanismo_str)
        )
    else:
        # Subsistema sintético directo para N grandes sin CSV (ej. N=25)
        # Usa topología aleatoria reproducible basada en las variables
        n_sub  = min(len(alcance_str), len(mecanismo_str))
        n_rows = 2 ** min(n_sub, 20)  # max 2^20 filas en memoria
        seed   = sum(ord(c) for c in estado_inicial + alcance_str + mecanismo_str) % (2**31)
        rng    = np.random.default_rng(seed)
        tpm_rand = rng.random((n_rows, n_sub))
        est_sub  = (estado_inicial + '0' * n_sub)[:n_sub]
        etqs_sub = list(alcance_str[:n_sub])
        subsistema = System(tpm_rand, est_sub, etqs_sub)
        print(f"  [SINT] {csv_path} no existe → subsistema sintético n={n_sub}")

    print(f"\n  Subsistema: alcance={alcance_str} mecanismo={mecanismo_str}")
    print(f"  {subsistema}")

    # Ejecutar KGeoMIP (bipartición k=2)
    try:
        geo = KGeoMIP(subsistema)
        t0  = time.time()
        res_geo = geo.aplicar_estrategia()
        res_geo['tiempo'] = time.time() - t0

        phi_raw  = res_geo['phi']
        phi_norm = phi_raw / subsistema.n if subsistema.n > 0 else 0
        res_geo['phi'] = phi_norm

        p1, p2 = res_geo['biparticion']
        etqs   = subsistema.etiquetas
        print(f"  KGeoMIP k=2: phi={phi_norm:.6f} "
              f"[{''.join(etqs[i] for i in p1)}]|"
              f"[{''.join(etqs[i] for i in p2)}] "
              f"t={res_geo['tiempo']:.4f}s")

        registro_geo.registrar(Resultado(
            n=subsistema.n,
            estado_inicial=estado_inicial,
            biparticion=res_geo['biparticion'],
            phi=phi_norm,
            tiempo=res_geo['tiempo'],
            estrategia=f"KGeoMIP|alc={alcance_str}|mec={mecanismo_str}"
        ))
    except Exception as e:
        print(f"  ERROR KGeoMIP: {e}")

    # Ejecutar KQNodes (k=3,4,5)
    try:
        import threading

        qn = KQNodes(subsistema)
        t0 = time.time()

        if timeout_qnodes and subsistema.n >= 9:
            resultado_container = [None]
            error_container     = [None]

            def _ejecutar_qn():
                try:
                    resultado_container[0] = qn.aplicar_estrategia()
                except Exception as ex:
                    error_container[0] = ex

            hilo = threading.Thread(target=_ejecutar_qn, daemon=True)
            hilo.start()
            hilo.join(timeout=timeout_qnodes)

            if hilo.is_alive():
                elapsed = time.time() - t0
                print(f"  KQNodes TIMEOUT ({timeout_qnodes}s) para n={subsistema.n}")
                if salida_path:
                    with open(salida_path, 'a', encoding='utf-8') as f:
                        f.write(f"{num_prueba},{subsistema.n},{estado_inicial},"
                                f"{alcance_str},{mecanismo_str},"
                                f"3,KQNodes-TIMEOUT,TIMEOUT,"
                                f"-1,{elapsed:.4f}\n")
                res_qn = None
            else:
                res_qn = resultado_container[0]
                if error_container[0]:
                    raise error_container[0]
        else:
            res_qn = qn.aplicar_estrategia()

        if res_qn is not None:
            res_qn['tiempo'] = time.time() - t0
            qn.resultado = res_qn
            qn.imprimir_resultado_k(res_qn)

            if res_qn['biparticion']:
                registro_qnod.registrar(Resultado(
                    n=subsistema.n,
                    estado_inicial=estado_inicial,
                    biparticion=(res_qn['biparticion'][0],
                                 res_qn['biparticion'][-1]),
                    phi=res_qn['phi'],
                    tiempo=res_qn['tiempo'],
                    estrategia=f"KQNodes-k{res_qn.get('k','?')}|alc={alcance_str}|mec={mecanismo_str}"
                ))

    except Exception as e:
        print(f"  ERROR KQNodes: {e}")


# Pruebas N=10 según DatosPruebas2026_1.xlsx
PRUEBAS_N10 = [
    ("ABCDEFGHIJ", "ABCDEFGHIJ"),
    ("ABCDEFGHIJ", "ABCDEFGHI"),
    ("ABCDEFGHIJ", "BCDEFGHIJ"),
    ("ABCDEFGHIJ", "BCDEFGHI"),
    ("ABCDEFGHIJ", "ABDEGHJ"),
    ("ABCDEFGHIJ", "ACEGI"),
    ("ABCDEFGHIJ", "BDFHJ"),
    ("ABCDEFGHI",  "ABCDEFGHIJ"),
    ("ABCDEFGHI",  "ABCDEFGHI"),
    ("ABCDEFGHI",  "BCDEFGHIJ"),
    ("ABCDEFGHI",  "BCDEFGHI"),
    ("ABCDEFGHI",  "ABDEGHJ"),
    ("ABCDEFGHI",  "ACEGI"),
    ("ABCDEFGHI",  "BDFHJ"),
    ("BCDEFGHIJ",  "ABCDEFGHIJ"),
    ("BCDEFGHIJ",  "ABCDEFGHI"),
    ("BCDEFGHIJ",  "BCDEFGHIJ"),
    ("BCDEFGHIJ",  "BCDEFGHI"),
    ("BCDEFGHIJ",  "ABDEGHJ"),
    ("BCDEFGHIJ",  "ACEGI"),
    ("BCDEFGHIJ",  "BDFHJ"),
    ("BCDEFGHI",   "ABCDEFGHIJ"),
    ("BCDEFGHI",   "ABCDEFGHI"),
    ("BCDEFGHI",   "BCDEFGHIJ"),
    ("BCDEFGHI",   "BCDEFGHI"),
    ("BCDEFGHI",   "ABDEGHJ"),
    ("BCDEFGHI",   "ACEGI"),
    ("BCDEFGHI",   "BDFHJ"),
    ("ABDEGHJ",    "ABCDEFGHIJ"),
    ("ABDEGHJ",    "ABCDEFGHI"),
    ("ABDEGHJ",    "BCDEFGHIJ"),
    ("ABDEGHJ",    "BCDEFGHI"),
    ("ABDEGHJ",    "ABDEGHJ"),
    ("ABDEGHJ",    "ACEGI"),
    ("ABDEGHJ",    "BDFHJ"),
    ("ACEGI",      "ABCDEFGHIJ"),
    ("ACEGI",      "ABCDEFGHI"),
    ("ACEGI",      "BCDEFGHIJ"),
    ("ACEGI",      "BCDEFGHI"),
    ("ACEGI",      "ABDEGHJ"),
    ("ACEGI",      "ACEGI"),
    ("ACEGI",      "BDFHJ"),
    ("BDFHJ",      "ABCDEFGHIJ"),
    ("BDFHJ",      "ABCDEFGHI"),
    ("BDFHJ",      "BCDEFGHIJ"),
    ("BDFHJ",      "BCDEFGHI"),
    ("BDFHJ",      "ABDEGHJ"),
    ("BDFHJ",      "ACEGI"),
    ("BDFHJ",      "BDFHJ"),
]

# Pruebas N=15 según DatosPruebas2026_1.xlsx
PRUEBAS_N15 = [
    ("ABCDEFGHIJKLMNO", "ABCDEFGHIJKLMNO"),
    ("ABCDEFGHIJKLMNO", "ABCDEFGHIJKLMN"),
    ("ABCDEFGHIJKLMNO", "BCDEFGHIJKLMNO"),
    ("ABCDEFGHIJKLMNO", "BCDEFGHIJKLMN"),
    ("ABCDEFGHIJKLMNO", "ABDEGHJKMN"),
    ("ABCDEFGHIJKLMNO", "ACEGIKMO"),
    ("ABCDEFGHIJKLMNO", "BDFHJLN"),
    ("ABCDEFGHIJKLMN",  "ABCDEFGHIJKLMNO"),
    ("ABCDEFGHIJKLMN",  "ABCDEFGHIJKLMN"),
    ("ABCDEFGHIJKLMN",  "BCDEFGHIJKLMNO"),
    ("ABCDEFGHIJKLMN",  "BCDEFGHIJKLMN"),
    ("ABCDEFGHIJKLMN",  "ABDEGHJKMN"),
    ("ABCDEFGHIJKLMN",  "ACEGIKMO"),
    ("ABCDEFGHIJKLMN",  "BDFHJLN"),
    ("BCDEFGHIJKLMNO",  "ABCDEFGHIJKLMNO"),
    ("BCDEFGHIJKLMNO",  "ABCDEFGHIJKLMN"),
    ("BCDEFGHIJKLMNO",  "BCDEFGHIJKLMNO"),
    ("BCDEFGHIJKLMNO",  "BCDEFGHIJKLMN"),
    ("BCDEFGHIJKLMNO",  "ABDEGHJKMN"),
    ("BCDEFGHIJKLMNO",  "ACEGIKMO"),
    ("BCDEFGHIJKLMNO",  "BDFHJLN"),
    ("BCDEFGHIJKLMN",   "ABCDEFGHIJKLMNO"),
    ("BCDEFGHIJKLMN",   "ABCDEFGHIJKLMN"),
    ("BCDEFGHIJKLMN",   "BCDEFGHIJKLMNO"),
    ("BCDEFGHIJKLMN",   "BCDEFGHIJKLMN"),
    ("BCDEFGHIJKLMN",   "ABDEGHJKMN"),
    ("BCDEFGHIJKLMN",   "ACEGIKMO"),
    ("BCDEFGHIJKLMN",   "BDFHJLN"),
    ("ABDEGHJKMN",      "ABCDEFGHIJKLMNO"),
    ("ABDEGHJKMN",      "ABCDEFGHIJKLMN"),
    ("ABDEGHJKMN",      "BCDEFGHIJKLMNO"),
    ("ABDEGHJKMN",      "BCDEFGHIJKLMN"),
    ("ABDEGHJKMN",      "ABDEGHJKMN"),
    ("ABDEGHJKMN",      "ACEGIKMO"),
    ("ABDEGHJKMN",      "BDFHJLN"),
    ("ACEGIKMO",        "ABCDEFGHIJKLMNO"),
    ("ACEGIKMO",        "ABCDEFGHIJKLMN"),
    ("ACEGIKMO",        "BCDEFGHIJKLMNO"),
    ("ACEGIKMO",        "BCDEFGHIJKLMN"),
    ("ACEGIKMO",        "ABDEGHJKMN"),
    ("ACEGIKMO",        "ACEGIKMO"),
    ("ACEGIKMO",        "BDFHJLN"),
    ("BDFHJLN",         "ABCDEFGHIJKLMNO"),
    ("BDFHJLN",         "ABCDEFGHIJKLMN"),
    ("BDFHJLN",         "BCDEFGHIJKLMNO"),
    ("BDFHJLN",         "BCDEFGHIJKLMN"),
    ("BDFHJLN",         "ABDEGHJKMN"),
    ("BDFHJLN",         "ACEGIKMO"),
    ("BDFHJLN",         "BDFHJLN"),
    ("BCDEFGJKLMNO",    "BCDEFGHIJKLMNO"),
]

# Pruebas N=20 según DatosPruebas2026_1.xlsx
PRUEBAS_N20 = [
    ("ABCDEFGHIJKLMNOPQRST", "ABCDEFGHIJKLMNOPQRST"),
    ("ABCDEFGHIJKLMNOPQRST", "ABCDEFGHIJKLMNOPQRS"),
    ("ABCDEFGHIJKLMNOPQRST", "BCDEFGHIJKLMNOPQRST"),
    ("ABCDEFGHIJKLMNOPQRST", "BCDEFGHIJKLMNOPQRS"),
    ("ABCDEFGHIJKLMNOPQRST", "ABDEGHJKMNPQST"),
    ("ABCDEFGHIJKLMNOPQRST", "ACEGIKMOQS"),
    ("ABCDEFGHIJKLMNOPQRST", "BDFHJLNPRT"),
    ("ABCDEFGHIJKLMNOPQRS",  "ABCDEFGHIJKLMNOPQRST"),
    ("ABCDEFGHIJKLMNOPQRS",  "ABCDEFGHIJKLMNOPQRS"),
    ("ABCDEFGHIJKLMNOPQRS",  "BCDEFGHIJKLMNOPQRST"),
    ("ABCDEFGHIJKLMNOPQRS",  "BCDEFGHIJKLMNOPQRS"),
    ("ABCDEFGHIJKLMNOPQRS",  "ABDEGHJKMNPQST"),
    ("ABCDEFGHIJKLMNOPQRS",  "ACEGIKMOQS"),
    ("ABCDEFGHIJKLMNOPQRS",  "BDFHJLNPRT"),
    ("BCDEFGHIJKLMNOPQRST",  "ABCDEFGHIJKLMNOPQRST"),
    ("BCDEFGHIJKLMNOPQRST",  "ABCDEFGHIJKLMNOPQRS"),
    ("BCDEFGHIJKLMNOPQRST",  "BCDEFGHIJKLMNOPQRST"),
    ("BCDEFGHIJKLMNOPQRST",  "BCDEFGHIJKLMNOPQRS"),
    ("BCDEFGHIJKLMNOPQRST",  "ABDEGHJKMNPQST"),
    ("BCDEFGHIJKLMNOPQRST",  "ACEGIKMOQS"),
    ("BCDEFGHIJKLMNOPQRST",  "BDFHJLNPRT"),
    ("BCDEFGHIJKLMNOPQRS",   "ABCDEFGHIJKLMNOPQRST"),
    ("BCDEFGHIJKLMNOPQRS",   "ABCDEFGHIJKLMNOPQRS"),
    ("BCDEFGHIJKLMNOPQRS",   "BCDEFGHIJKLMNOPQRST"),
    ("BCDEFGHIJKLMNOPQRS",   "BCDEFGHIJKLMNOPQRS"),
    ("BCDEFGHIJKLMNOPQRS",   "ABDEGHJKMNPQST"),
    ("BCDEFGHIJKLMNOPQRS",   "ACEGIKMOQS"),
    ("BCDEFGHIJKLMNOPQRS",   "BDFHJLNPRT"),
    ("ABDEGHJKMNPQST",       "ABCDEFGHIJKLMNOPQRST"),
    ("ABDEGHJKMNPQST",       "ABCDEFGHIJKLMNOPQRS"),
    ("ABDEGHJKMNPQST",       "BCDEFGHIJKLMNOPQRST"),
    ("ABDEGHJKMNPQST",       "BCDEFGHIJKLMNOPQRS"),
    ("ABDEGHJKMNPQST",       "ABDEGHJKMNPQST"),
    ("ABDEGHJKMNPQST",       "ACEGIKMOQS"),
    ("ABDEGHJKMNPQST",       "BDFHJLNPRT"),
    ("ACEGIKMOQS",           "ABCDEFGHIJKLMNOPQRST"),
    ("ACEGIKMOQS",           "ABCDEFGHIJKLMNOPQRS"),
    ("ACEGIKMOQS",           "BCDEFGHIJKLMNOPQRST"),
    ("ACEGIKMOQS",           "BCDEFGHIJKLMNOPQRS"),
    ("ACEGIKMOQS",           "ABDEGHJKMNPQST"),
    ("ACEGIKMOQS",           "ACEGIKMOQS"),
    ("ACEGIKMOQS",           "BDFHJLNPRT"),
    ("BDFHJLNPRT",           "ABCDEFGHIJKLMNOPQRST"),
    ("BDFHJLNPRT",           "ABCDEFGHIJKLMNOPQRS"),
    ("BDFHJLNPRT",           "BCDEFGHIJKLMNOPQRST"),
    ("BDFHJLNPRT",           "BCDEFGHIJKLMNOPQRS"),
    ("BDFHJLNPRT",           "ABDEGHJKMNPQST"),
    ("BDFHJLNPRT",           "ACEGIKMOQS"),
    ("BDFHJLNPRT",           "BDFHJLNPRT"),
    ("BCDEFGJKLMNO",         "BCDEFGHIJKLMNO"),
]

# Pruebas N=22 según DatosPruebas2026_1.xlsx
PRUEBAS_N22 = [
    ("ABCDEFGHIJKLMNOPQRSTUV", "ABCDEFGHIJKLMNOPQRSTUV"),
    ("ABCDEFGHIJKLMNOPQRSTUV", "ABCDEFGHIJKLMNOPQRSTU"),
    ("ABCDEFGHIJKLMNOPQRSTUV", "BCDEFGHIJKLMNOPQRSTUV"),
    ("ABCDEFGHIJKLMNOPQRSTUV", "BCDEFGHIJKLMNOPQRSTU"),
    ("ABCDEFGHIJKLMNOPQRSTUV", "ABDEGHJKMNPQSTV"),
    ("ABCDEFGHIJKLMNOPQRSTUV", "ACEGIKMOQSU"),
    ("ABCDEFGHIJKLMNOPQRSTUV", "BDFHJLNPRTV"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "ABCDEFGHIJKLMNOPQRSTUV"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "ABCDEFGHIJKLMNOPQRSTU"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "BCDEFGHIJKLMNOPQRSTUV"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "BCDEFGHIJKLMNOPQRSTU"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "ABDEGHJKMNPQSTV"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "ACEGIKMOQSU"),
    ("ABCDEFGHIJKLMNOPQRSTU",  "BDFHJLNPRTV"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "ABCDEFGHIJKLMNOPQRSTUV"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "ABCDEFGHIJKLMNOPQRSTU"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "BCDEFGHIJKLMNOPQRSTUV"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "BCDEFGHIJKLMNOPQRSTU"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "ABDEGHJKMNPQSTV"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "ACEGIKMOQSU"),
    ("BCDEFGHIJKLMNOPQRSTUV",  "BDFHJLNPRTV"),
    ("BCDEFGHIJKLMNOPQRSTU",   "ABCDEFGHIJKLMNOPQRSTUV"),
    ("BCDEFGHIJKLMNOPQRSTU",   "ABCDEFGHIJKLMNOPQRSTU"),
    ("BCDEFGHIJKLMNOPQRSTU",   "BCDEFGHIJKLMNOPQRSTUV"),
    ("BCDEFGHIJKLMNOPQRSTU",   "BCDEFGHIJKLMNOPQRSTU"),
    ("BCDEFGHIJKLMNOPQRSTU",   "ABDEGHJKMNPQSTV"),
    ("BCDEFGHIJKLMNOPQRSTU",   "ACEGIKMOQSU"),
    ("BCDEFGHIJKLMNOPQRSTU",   "BDFHJLNPRTV"),
    ("ABDEGHJKMNPQSTV",        "ABCDEFGHIJKLMNOPQRSTUV"),
    ("ABDEGHJKMNPQSTV",        "ABCDEFGHIJKLMNOPQRSTU"),
    ("ABDEGHJKMNPQSTV",        "BCDEFGHIJKLMNOPQRSTUV"),
    ("ABDEGHJKMNPQSTV",        "BCDEFGHIJKLMNOPQRSTU"),
    ("ABDEGHJKMNPQSTV",        "ABDEGHJKMNPQSTV"),
    ("ABDEGHJKMNPQSTV",        "ACEGIKMOQSU"),
    ("ABDEGHJKMNPQSTV",        "BDFHJLNPRTV"),
    ("ACEGIKMOQSU",            "ABCDEFGHIJKLMNOPQRSTUV"),
    ("ACEGIKMOQSU",            "ABCDEFGHIJKLMNOPQRSTU"),
    ("ACEGIKMOQSU",            "BCDEFGHIJKLMNOPQRSTUV"),
    ("ACEGIKMOQSU",            "BCDEFGHIJKLMNOPQRSTU"),
    ("ACEGIKMOQSU",            "ABDEGHJKMNPQSTV"),
    ("ACEGIKMOQSU",            "ACEGIKMOQSU"),
    ("ACEGIKMOQSU",            "BDFHJLNPRTV"),
    ("BDFHJLNPRTV",            "ABCDEFGHIJKLMNOPQRSTUV"),
    ("BDFHJLNPRTV",            "ABCDEFGHIJKLMNOPQRSTU"),
    ("BDFHJLNPRTV",            "BCDEFGHIJKLMNOPQRSTUV"),
    ("BDFHJLNPRTV",            "BCDEFGHIJKLMNOPQRSTU"),
    ("BDFHJLNPRTV",            "ABDEGHJKMNPQSTV"),
    ("BDFHJLNPRTV",            "ACEGIKMOQSU"),
    ("BDFHJLNPRTV",            "BDFHJLNPRTV"),
    ("ACDEFGHIJKLMNOPQRST",    "ACDEFGHIJKLMNOPQRST"),
]

# Pruebas N=25 según DatosPruebas2026_1.xlsx
PRUEBAS_N25 = [
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "ABDEGHJKMNPQSTVWY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "ACEGIKMOQSUWY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXY", "BDFHJLNPRTVX"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "ABDEGHJKMNPQSTVWY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "ACEGIKMOQSUWY"),
    ("ABCDEFGHIJKLMNOPQRSTUVWX",  "BDFHJLNPRTVX"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "ABDEGHJKMNPQSTVWY"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "ACEGIKMOQSUWY"),
    ("BCDEFGHIJKLMNOPQRSTUVWXY",  "BDFHJLNPRTVX"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "ABDEGHJKMNPQSTVWY"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "ACEGIKMOQSUWY"),
    ("BCDEFGHIJKLMNOPQRSTUVWX",   "BDFHJLNPRTVX"),
    ("ABDEGHJKMNPQSTVWY",         "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ABDEGHJKMNPQSTVWY",         "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("ABDEGHJKMNPQSTVWY",         "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ABDEGHJKMNPQSTVWY",         "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("ABDEGHJKMNPQSTVWY",         "ABDEGHJKMNPQSTVWY"),
    ("ABDEGHJKMNPQSTVWY",         "ACEGIKMOQSUWY"),
    ("ABDEGHJKMNPQSTVWY",         "BDFHJLNPRTVX"),
    ("ACEGIKMOQSUWY",             "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ACEGIKMOQSUWY",             "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("ACEGIKMOQSUWY",             "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("ACEGIKMOQSUWY",             "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("ACEGIKMOQSUWY",             "ABDEGHJKMNPQSTVWY"),
    ("ACEGIKMOQSUWY",             "ACEGIKMOQSUWY"),
    ("ACEGIKMOQSUWY",             "BDFHJLNPRTVX"),
    ("BDFHJLNPRTVX",              "ABCDEFGHIJKLMNOPQRSTUVWXY"),
    ("BDFHJLNPRTVX",              "ABCDEFGHIJKLMNOPQRSTUVWX"),
    ("BDFHJLNPRTVX",              "BCDEFGHIJKLMNOPQRSTUVWXY"),
    ("BDFHJLNPRTVX",              "BCDEFGHIJKLMNOPQRSTUVWX"),
    ("BDFHJLNPRTVX",              "ABDEGHJKMNPQSTVWY"),
    ("BDFHJLNPRTVX",              "ACEGIKMOQSUWY"),
    ("BDFHJLNPRTVX",              "BDFHJLNPRTVX"),
    ("ACDEFGHIJKLMNOPQRSTVX",     "ACDEFGHIJKLMNOPQRSTVX"),
]


def ejecutar_suite_pruebas(
    n: int,
    estado_inicial: str,
    sistema_str: str,
    pruebas: list,
    carpeta: str = "data"
) -> None:
    """
    Ejecuta todas las pruebas del Excel para un tamaño N dado.
    Guarda resultados en results/resultados_suite_N{n}.csv
    """
    csv_path = os.path.join(carpeta, f"N{n}C.csv")
    if not os.path.exists(csv_path):
        print(f"  AVISO: {csv_path} no encontrado.")
        return

    os.makedirs("results", exist_ok=True)
    salida_path = os.path.join("results", f"resultados_suite_N{n}.csv")

    # Encabezado del CSV de resultados
    with open(salida_path, 'w', encoding='utf-8') as f:
        f.write("prueba,n_sub,estado_inicial,alcance,mecanismo,"
                "k,estrategia,particion,perdida,tiempo_s\n")

    reg_geo  = RegistroMetricas(carpeta="results")
    reg_qnod = RegistroMetricas(carpeta="results")

    total = len(pruebas)
    print(f"\n{'='*60}")
    print(f"  Suite N={n} — {total} pruebas")
    print(f"  Estado inicial: {estado_inicial}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}")

    for num, (alcance_str, mecanismo_str) in enumerate(pruebas, start=1):
        print(f"\n[{num}/{total}] alcance={alcance_str}  mecanismo={mecanismo_str}")

        # Estimar tamaño del subsistema para decidir timeout
        # n_sub >= 9: KQNodes puede tardar horas -> timeout 600s (10 min)
        n_sub_estimado = min(len(alcance_str), len(mecanismo_str))
        usar_timeout   = n_sub_estimado >= 9

        try:
            ejecutar_prueba(
                n=n,
                estado_inicial=estado_inicial,
                alcance_str=alcance_str,
                mecanismo_str=mecanismo_str,
                csv_path=csv_path,
                registro_geo=reg_geo,
                registro_qnod=reg_qnod,
                timeout_qnodes=600 if usar_timeout else None,
                salida_path=salida_path,
                num_prueba=num
            )

            # Guardar en CSV de suite inmediatamente (no perder datos si se cuelga)
            with open(salida_path, 'a', encoding='utf-8') as f:
                # Buscar el último resultado registrado de cada estrategia
                if reg_geo.resultados:
                    r = reg_geo.resultados[-1]
                    p1, p2 = r.biparticion
                    part_str = f"[{','.join(str(x) for x in p1)}]|[{','.join(str(x) for x in p2)}]"
                    f.write(f'{num},{r.n},{estado_inicial},'
                            f'{alcance_str},{mecanismo_str},'
                            f'2,KGeoMIP,"{part_str}",'
                            f'{r.phi:.8f},{r.tiempo:.4f}\n')
                if reg_qnod.resultados:
                    r = reg_qnod.resultados[-1]
                    p1, p2 = r.biparticion
                    part_str = f"[{','.join(str(x) for x in p1)}]|[{','.join(str(x) for x in p2)}]"
                    k_val = r.estrategia.split('-k')[1].split('|')[0] if '-k' in r.estrategia else '?'
                    f.write(f'{num},{r.n},{estado_inicial},'
                            f'{alcance_str},{mecanismo_str},'
                            f'{k_val},KQNodes,"{part_str}",'
                            f'{r.phi:.8f},{r.tiempo:.4f}\n')

        except Exception as e:
            print(f"  ERROR prueba {num}: {e}")
            with open(salida_path, 'a', encoding='utf-8') as f:
                f.write(f"{num},ERROR,{estado_inicial},"
                        f"{alcance_str},{mecanismo_str},?,?,ERROR,0,0\n")

    print(f"\n  Resultados guardados en: {salida_path}")
    reg_geo.guardar_csv(f"resultados_geo_N{n}.csv")
    reg_qnod.guardar_csv(f"resultados_qnod_N{n}.csv")


def main():
    ncpus = multiprocessing.cpu_count()
    print("\n" + "="*60)
    print("  PROYECTO KQNodes / KGeoMIP")
    print("  Analisis y Diseno de Algoritmos 2026-1")
    print(f"  CPUs: {ncpus}")
    print("="*60)

    import sys
    modo = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if modo == "suite_n10":
        ejecutar_suite_pruebas(
            n=10,
            estado_inicial="1000000000",
            sistema_str="ABCDEFGHIJ",
            pruebas=PRUEBAS_N10
        )
    elif modo == "suite_n15":
        ejecutar_suite_pruebas(
            n=15,
            estado_inicial="100000000000000",
            sistema_str="ABCDEFGHIJKLMNO",
            pruebas=PRUEBAS_N15
        )
    elif modo == "suite_n20":
        ejecutar_suite_pruebas(
            n=20,
            estado_inicial="10000000000000000000",
            sistema_str="ABCDEFGHIJKLMNOPQRST",
            pruebas=PRUEBAS_N20
        )
    elif modo == "suite_n22":
        ejecutar_suite_pruebas(
            n=22,
            estado_inicial="1000000000000000000000",
            sistema_str="ABCDEFGHIJKLMNOPQRSTUV",
            pruebas=PRUEBAS_N22
        )
    elif modo == "suite_n25":
        ejecutar_suite_pruebas(
            n=25,
            estado_inicial="1000000000000000000000000",
            sistema_str="ABCDEFGHIJKLMNOPQRSTUVWXY",
            pruebas=PRUEBAS_N25
        )
    else:
        # Modo normal: ejecutar para N=3,5,10,15 como antes
        reg_geo  = RegistroMetricas(carpeta="results")
        reg_qnod = RegistroMetricas(carpeta="results")

        for n in [3, 5, 10, 15]:
            print(f"\n{'#'*60}\n#  N = {n}\n{'#'*60}")
            try:
                print("\n>>> KGeoMIP (k=2):")
                ejecutar_geometric(n, reg_geo)
            except Exception as e:
                print(f"  ERROR KGeoMIP N={n}: {e}")
            try:
                print("\n>>> KQNodes (k=3,4,5):")
                ejecutar_qnodes(n, reg_qnod)
            except Exception as e:
                print(f"  ERROR KQNodes N={n}: {e}")

        print("\n" + "="*60)
        reg_geo.resumen()
        reg_geo.guardar_csv("resultados_geometric.csv")
        reg_qnod.resumen()
        reg_qnod.guardar_csv("resultados_qnodes.csv")


if __name__ == "__main__":
    main()
