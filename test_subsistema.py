import sys
sys.path.insert(0, '.')
from src.models.system import System

sistema = System.desde_csv('data/N10C.csv', '1000000000')
print('Sistema completo:', sistema)

sub = sistema.construir_subsistema(
    alcance_vars=list('ABCDEFGHIJ'),
    mecanismo_vars=list('ABCDEFGHI')
)
print('Subsistema:', sub)
print('Etiquetas subsistema:', sub.etiquetas)
print('TPM shape:', sub.tpm.shape)
