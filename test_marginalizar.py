import sys
sys.path.insert(0, '.')
from src.models.system import System

sistema = System.desde_csv('data/N10C.csv', '1000000000')

# Caso que fallaba: alcance=ACEGI, mecanismo=BCDEFGHIJ
sub = sistema.construir_subsistema(
    alcance_vars=list('ACEGI'),
    mecanismo_vars=list('BCDEFGHIJ')
)
print('Shape TPM:', sub.tpm.shape)
print('n:', sub.n)
print('etiquetas:', sub.etiquetas)
print('n == tpm.shape[1]:', sub.n == sub.tpm.shape[1])

# Verificar que obtener_tensores no falla
tensores = sub.obtener_tensores()
print('Tensores:', len(tensores), 'shapes:', [t.shape for t in tensores])

# Verificar distribucion_estado_inicial
dist = sub.distribucion_estado_inicial()
print('dist.shape:', dist.shape, 'suma:', dist.sum())
print('Todo OK')
