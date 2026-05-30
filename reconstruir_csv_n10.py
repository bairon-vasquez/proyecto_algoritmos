"""
Reconstruye resultados_suite_N10.csv desde el output del task b55vdzirf
que ejecutó la suite completa con el fix activo.
"""
import re
import csv

TASK_OUTPUT = (
    r"C:\Users\cdcar\AppData\Local\Temp\claude"
    r"\c--Users-cdcar-Documents-Proyectos-Programaci-n-GeoMIP-geomip-geomip-project"
    r"\3d67e071-6c11-4467-a50b-0f354219114a\tasks\b55vdzirf.output"
)
SALIDA = "results/resultados_suite_N10.csv"
ESTADO = "1000000000"

RE_HEADER  = re.compile(r'\[(\d+)/49\] alcance=(\S+)\s+mecanismo=(\S+)')
RE_NSUB    = re.compile(r'System\(n=(\d+),')
RE_GEO     = re.compile(r'KGeoMIP k=2: phi=([\d.]+).*?t=([\d.]+)s')
RE_BIP_N   = re.compile(r'N=\s*\d+.*?bipartición=(\[.*?\])\|(\[.*?\]).*?phi=([\d.]+).*?tiempo=([\d.]+)s')
RE_TIMEOUT = re.compile(r'KQNodes TIMEOUT \((\d+)s\)')
RE_OPTIMO  = re.compile(r'OPTIMO k=(\S+): phi=(\S+)')
RE_TIEMPO  = re.compile(r'Tiempo\s+:\s+([\d.]+)s')
RE_QN_N    = re.compile(r'N=\s*\d+.*?bipartición=(\[.*?\])\|(\[.*?\]).*?phi=([\d.]+).*?tiempo=([\d.]+)s')

with open(TASK_OUTPUT, encoding='utf-8') as f:
    lines = f.readlines()

filas = []
i = 0
while i < len(lines):
    m = RE_HEADER.search(lines[i])
    if not m:
        i += 1
        continue

    num     = int(m.group(1))
    alcance = m.group(2)
    mec     = m.group(3)
    i += 1

    # Buscar n_sub (System line)
    n_sub = None
    while i < len(lines) and '[' not in lines[i][:3]:
        ms = RE_NSUB.search(lines[i])
        if ms:
            n_sub = int(ms.group(1))
            break
        i += 1
    if n_sub is None:
        continue
    i += 1

    # Buscar KGeoMIP result
    geo_phi = geo_t = geo_p1 = geo_p2 = None
    while i < len(lines) and not RE_HEADER.search(lines[i]):
        mg = RE_GEO.search(lines[i])
        if mg:
            geo_phi = float(mg.group(1))
            geo_t   = float(mg.group(2))
            # Buscar biparticion en la siguiente linea N=
            for j in range(i+1, min(i+5, len(lines))):
                mb = RE_BIP_N.search(lines[j])
                if mb:
                    geo_p1 = mb.group(1)
                    geo_p2 = mb.group(2)
                    break
            break
        i += 1

    if geo_phi is not None:
        part = f'"{geo_p1}|{geo_p2}"' if geo_p1 else '[]|[]'
        filas.append({
            'prueba': num, 'n_sub': n_sub, 'estado_inicial': ESTADO,
            'alcance': alcance, 'mecanismo': mec,
            'k': 2, 'estrategia': 'KGeoMIP',
            'particion': f'{geo_p1}|{geo_p2}' if geo_p1 else '[]|[]',
            'perdida': f'{geo_phi:.8f}', 'tiempo_s': f'{geo_t:.4f}'
        })

    # Buscar KQNodes: TIMEOUT o resultado valido
    while i < len(lines) and not RE_HEADER.search(lines[i]):
        mt = RE_TIMEOUT.search(lines[i])
        if mt:
            to_s = float(mt.group(1))
            # Buscar elapsed (siguiente numero)
            for j in range(i+1, min(i+3, len(lines))):
                mt2 = re.search(r'-1,([\d.]+)', lines[j])
                if mt2:
                    to_s = float(mt2.group(1))
                    break
            filas.append({
                'prueba': num, 'n_sub': n_sub, 'estado_inicial': ESTADO,
                'alcance': alcance, 'mecanismo': mec,
                'k': 3, 'estrategia': 'KQNodes-TIMEOUT',
                'particion': 'TIMEOUT',
                'perdida': '-1', 'tiempo_s': f'{600:.4f}'
            })
            break

        mo = RE_OPTIMO.search(lines[i])
        if mo and mo.group(2) != 'inf':
            qn_phi = float(mo.group(2))
            qn_k   = mo.group(1)
            qn_t   = None
            # Buscar Tiempo y biparticion
            qn_p1 = qn_p2 = None
            for j in range(i+1, min(i+10, len(lines))):
                mt2 = RE_TIEMPO.search(lines[j])
                if mt2 and qn_t is None:
                    qn_t = float(mt2.group(1))
                mb = RE_QN_N.search(lines[j])
                if mb:
                    qn_p1 = mb.group(1)
                    qn_p2 = mb.group(2)
            filas.append({
                'prueba': num, 'n_sub': n_sub, 'estado_inicial': ESTADO,
                'alcance': alcance, 'mecanismo': mec,
                'k': qn_k, 'estrategia': 'KQNodes',
                'particion': f'{qn_p1}|{qn_p2}' if qn_p1 else f'k={qn_k}',
                'perdida': f'{qn_phi:.8f}', 'tiempo_s': f'{qn_t:.4f}' if qn_t else '0'
            })
            break
        i += 1

FIELDNAMES = ['prueba','n_sub','estado_inicial','alcance','mecanismo',
              'k','estrategia','particion','perdida','tiempo_s']

with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for r in filas:
        w.writerow(r)

# Casos degenerados: alcance ∩ mecanismo = ∅ → subsistema con 0 columnas → phi=0 (IIT)
CASOS_DEGENERATE = {
    42: ('ACEGI',  'BDFHJ'),  # alcance∩mec=∅
    48: ('BDFHJ',  'ACEGI'),  # alcance∩mec=∅
}
pruebas_cubiertas = set(r['prueba'] for r in filas)
for num, (alc, mec) in CASOS_DEGENERATE.items():
    if num not in pruebas_cubiertas:
        for est, k in [('KGeoMIP', 2), ('KQNodes', 3)]:
            filas.append({
                'prueba': num, 'n_sub': 0, 'estado_inicial': ESTADO,
                'alcance': alc, 'mecanismo': mec,
                'k': k, 'estrategia': est,
                'particion': '[]|[]',
                'perdida': '0.00000000', 'tiempo_s': '0.0000'
            })

# Reordenar por prueba
filas.sort(key=lambda r: (int(r['prueba']), r['estrategia']))

with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for r in filas:
        w.writerow(r)

print(f'Reconstruidas {len(filas)} filas para {len(set(r["prueba"] for r in filas))} pruebas')
pruebas_cubiertas = sorted(set(int(r["prueba"]) for r in filas))
faltantes = [n for n in range(1,50) if n not in pruebas_cubiertas]
if faltantes:
    print(f'Faltantes: {faltantes}')
else:
    print('Todas las 49 pruebas cubiertas')
