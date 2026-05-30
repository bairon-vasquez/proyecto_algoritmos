# Proyecto KQNodes / KGeoMIP
## Analisis y Diseno de Algoritmos 2026-1 — Prof. Luz Enith

## Estrategias implementadas
- **KGeoMIP**: algoritmo geometrico con BFS en hipercubo (k=2, biparticion optima)
- **KQNodes**: algoritmo Q-Nodos con memorizacion (k=3,4,5, k-particion optima)

## Como ejecutar

### Modo normal (N=3,5,10,15):
    python main.py

### Suite de pruebas N=10 (49 pruebas, estado inicial 1000000000):
    python main.py suite_n10

### Suite de pruebas N=15 (50 pruebas, estado inicial 100000000000000):
    python main.py suite_n15

### Generar reporte final:
    python generar_reporte.py

### Ver estadisticas:
    python estadisticas.py

## Resultados N=10 (49 pruebas completadas)
| Estrategia | Pruebas validas | Phi min | Phi max | Phi prom | Tiempo prom |
|------------|----------------|---------|---------|----------|-------------|
| KGeoMIP    | 49/49          | 0.0000  | 0.1250  | 0.0506   | 18.1s       |
| KQNodes    | 14/49          | 0.0000  | 0.3000  | 0.1485   | 122.6s      |

Nota: KQNodes no aplica cuando el subsistema tiene < 3 variables (no hay k=3 particion).
Los TIMEOUT (5 casos) son subsistemas n>=9 que superaron el limite de 600s.

## Archivos de resultados
- results/resultados_suite_N10.csv : Suite completa N=10
- results/resultados_suite_N15.csv : Suite completa N=15
- results/resultados_geo_N10.csv   : KGeoMIP detalle N=10
- results/resultados_qnod_N10.csv  : KQNodes detalle N=10

## Dependencias
    pip install numpy pandas scipy POT openpyxl pytest

## Estructura del proyecto
    src/models/system.py                    : TPM, marginalizacion, subsistema
    src/controllers/strategies/geometric.py : KGeoMIP (antes GeometricSIA)
    src/controllers/strategies/qnodes.py    : KQNodes (antes QNodesKPartition)
    src/controllers/strategies/sia.py       : Clase base abstracta
    src/utils/emd.py                        : EMD sparse con POT
    src/utils/metrics.py                    : Registro de resultados
    data/N10C.csv, data/N15C.csv            : TPMs de prueba
