# Valor máximo de k para k-particiones (tu proyecto llega hasta k=5)
MAX_K_PARTITIONS = 5

# Valor de pérdida infinita, usado cuando una partición es inválida
INFINITE_LOSS = float("inf")

# Tamaño máximo de sistema para búsqueda exhaustiva de biparticiones
# (sistemas con más variables usarán heurísticas)
MAX_EXHAUSTIVE_BIPARTITION = 20

# Tamaño máximo para búsqueda exhaustiva de k-particiones
# (para k>2, el espacio crece mucho más rápido)
MAX_EXHAUSTIVE_KPARTITION = 10

# Tolerancia numérica para comparar valores de pérdida EMD
EPSILON = 1e-10