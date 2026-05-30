import sys
sys.path.insert(0, '.')
from src.models.system import System

sistema = System.desde_csv('data/N10C.csv', '1000000000')

casos = [
    # (alcance, mecanismo, filas_esperadas, cols_esperadas)
    ("ABCDEFGHIJ", "ABCDEFGHIJ", 1024, 10),  # completo
    ("ABCDEFGHIJ", "ABCDEFGHI",   512,  9),  # mecanismo sin J
    ("ABCDEFGHI",  "ABCDEFGHIJ", 1024,  9),  # alcance sin J
    ("ABCDEFGHI",  "ABCDEFGHI",   512,  9),  # ambos sin J
    ("ACEGI",      "ACEGI",        32,  5),  # subsistema pequeño
    ("ACEGI",      "ABCDEFGHIJ", 1024,  5),  # mecanismo completo, alcance parcial
    ("ABCDEFGHIJ", "ACEGI",        32,  5),  # mecanismo parcial, alcance completo
]

print(f"{'Alcance':<15} {'Mecanismo':<15} {'TPM shape':>12} {'Esperado':>12} {'OK':>5}")
print("-" * 65)
todos_ok = True
for alcance, mec, filas_esp, cols_esp in casos:
    sub = sistema.construir_subsistema(list(alcance), list(mec))
    filas, cols = sub.tpm.shape
    ok = (filas == filas_esp and cols == cols_esp)
    if not ok:
        todos_ok = False
    print(f"{alcance:<15} {mec:<15} {str(sub.tpm.shape):>12} "
          f"{f'({filas_esp},{cols_esp})':>12} {'OK' if ok else 'FAIL':>5}")

print()
print("Todos los casos OK:", todos_ok)
