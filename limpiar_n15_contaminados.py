import csv

path = 'results/resultados_suite_N15.csv'

with open(path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    filas = list(reader)

# Regla robusta: si una prueba tiene KQNodes-TIMEOUT,
# cualquier KQNodes (sin TIMEOUT) para esa misma prueba es contaminacion
# del resultado registrado de la prueba anterior.
pruebas_con_timeout = {
    r['prueba']
    for r in filas
    if 'TIMEOUT' in r.get('estrategia', '')
}

def es_contaminada(r):
    prueba     = r.get('prueba', '')
    estrategia = r.get('estrategia', '')
    return (prueba in pruebas_con_timeout and
            'KQNodes' in estrategia and
            'TIMEOUT' not in estrategia)

limpias    = [r for r in filas if not es_contaminada(r)]
eliminadas = len(filas) - len(limpias)

with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(limpias)

print(f'Pruebas con TIMEOUT : {len(pruebas_con_timeout)}')
print(f'Filas eliminadas    : {eliminadas}')
print(f'Filas limpias       : {len(limpias)}')
