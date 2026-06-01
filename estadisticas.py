import csv, os

for archivo, nombre in [
    ('results/resultados_suite_N10.csv', 'N=10'),
    ('results/resultados_suite_N15.csv', 'N=15'),
    ('results/resultados_suite_N20.csv', 'N=20'),
    ('results/resultados_suite_N22.csv', 'N=22'),
    ('results/resultados_suite_N25.csv', 'N=25'),
]:
    if not os.path.exists(archivo):
        continue
    with open(archivo, encoding='utf-8') as f:
        filas = list(csv.DictReader(f))

    if not filas:
        print(f'\n=== {nombre} === (sin datos)')
        continue

    geo     = [r for r in filas if 'KGeoMIP' in r.get('estrategia','') and 'TIMEOUT' not in r.get('estrategia','')]
    qnod    = [r for r in filas if 'KQNodes' in r.get('estrategia','') and 'TIMEOUT' not in r.get('estrategia','')]
    timeout = [r for r in filas if 'TIMEOUT' in r.get('estrategia','')]

    phi_geo = [float(r['perdida']) for r in geo  if r.get('perdida','') not in ('','ERROR')]
    phi_qn  = [float(r['perdida']) for r in qnod if r.get('perdida','') not in ('','ERROR') and float(r['perdida']) >= 0]
    t_geo   = [float(r['tiempo_s']) for r in geo  if r.get('tiempo_s','')]
    t_qn    = [float(r['tiempo_s']) for r in qnod if r.get('tiempo_s','')]

    print(f'\n=== {nombre} ===')
    print(f'Pruebas totales : {len(set(r["prueba"] for r in filas))}')
    print(f'KGeoMIP validos : {len(geo)}')
    print(f'KQNodes validos : {len(qnod)}')
    print(f'TIMEOUT         : {len(timeout)}')
    if phi_geo:
        print(f'Phi KGeoMIP     : min={min(phi_geo):.4f}  max={max(phi_geo):.4f}  prom={sum(phi_geo)/len(phi_geo):.4f}')
        print(f'Tiempo KGeoMIP  : min={min(t_geo):.2f}s  max={max(t_geo):.2f}s  prom={sum(t_geo)/len(t_geo):.2f}s')
    if phi_qn:
        print(f'Phi KQNodes     : min={min(phi_qn):.4f}  max={max(phi_qn):.4f}  prom={sum(phi_qn)/len(phi_qn):.4f}')
        print(f'Tiempo KQNodes  : min={min(t_qn):.2f}s  max={max(t_qn):.2f}s  prom={sum(t_qn)/len(t_qn):.2f}s')
    phi0_geo = len([p for p in phi_geo if p == 0.0])
    phi0_qn  = len([p for p in phi_qn  if p == 0.0])
    print(f'Phi=0 KGeoMIP   : {phi0_geo}/{len(geo)} ({100*phi0_geo/max(len(geo),1):.0f}%)')
    print(f'Phi=0 KQNodes   : {phi0_qn}/{len(qnod)} ({100*phi0_qn/max(len(qnod),1):.0f}%)')
