import csv, os

def limpiar_suite_csv(path_entrada, path_salida=None):
    """
    Conserva solo la fila MÁS RECIENTE por (prueba, estrategia_base).
    'Más reciente' = última aparición en el archivo (las corridas nuevas
    se agregan al final del CSV).
    """
    if not os.path.exists(path_entrada):
        print(f'No existe: {path_entrada}')
        return

    with open(path_entrada, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        filas_raw = list(reader)

    # Limpiar filas: eliminar clave None (filas con columnas extra por commas sin escapar)
    filas = []
    for r in filas_raw:
        fila_limpia = {k: v for k, v in r.items() if k is not None}
        # Reconstruir particion desde columnas extras si las hay
        if None in r:
            extras = ','.join(r[None]) if isinstance(r[None], list) else str(r[None])
            fila_limpia['particion'] = fila_limpia.get('particion', '') + ',' + extras
        filas.append(fila_limpia)

    # Tomar la ÚLTIMA ocurrencia de cada (prueba, estrategia_base)
    vistas = {}
    for r in filas:
        prueba = r.get('prueba', '?')
        est_base = r.get('estrategia', '?').split('|')[0]
        clave = (prueba, est_base)
        vistas[clave] = r  # sobrescribe -> queda la última

    # Ordenar por número de prueba
    dedup = sorted(vistas.values(),
                   key=lambda r: (int(r.get('prueba', 0))
                                  if str(r.get('prueba', '0')).isdigit() else 0,
                                  r.get('estrategia', '')))

    salida = path_salida or path_entrada
    with open(salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dedup)

    print(f'{path_entrada}: {len(filas)} filas -> {len(dedup)} (limpias)')

limpiar_suite_csv('results/resultados_suite_N10.csv')
limpiar_suite_csv('results/resultados_suite_N15.csv')
