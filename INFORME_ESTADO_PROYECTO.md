# Informe de Estado del Proyecto GeoMIP
## Análisis y Diseño de Algoritmos — 2026-1

> **Fecha de revisión:** 2026-06-07  
> **Revisado por:** Claude Sonnet 4.6 (Senior Dev)  
> **Fuentes:** `1_Guía_Proyecto_ADAV1_2_0.pdf` (GeoMIP), `2_GeoMIP.pdf` (Generalidades)

---

## Resumen Ejecutivo

El proyecto implementa la **Estrategia Geométrica (KGeoMIP)** y el algoritmo **KQNodes** para encontrar biparticiones óptimas de sistemas de variables binarias, minimizando la función de Mínima Pérdida de Información (MIP) medida con EMD+Hamming. El avance es **alto** (~85% de los requisitos funcionales del documento), pero existen brechas críticas en validación comparativa, datos de prueba y métricas de desempeño formales.

---

## 1. Matriz de Requisitos vs. Implementación

### 1.1 Fundamentos teóricos (PDF `2_GeoMIP.pdf`)

| Requisito | Estado | Archivo / Componente |
|-----------|--------|----------------------|
| TPM formato estado-estado | ✅ Implementado | `system.py:_ee_a_nodo()` |
| TPM formato estado-nodo | ✅ Implementado | `system.py:_desde_csv_numpy()` |
| Carga de CSV con detección de encabezado | ✅ Implementado | `system.py:desde_csv()` |
| Sistema candidato / condiciones de fondo | ✅ Implementado | `system.py:condicionar()` |
| Marginalización de filas (tiempo t) | ✅ Implementado | `system.py:marginalizar_filas()` |
| Marginalización de columnas (tiempo t+1) | ✅ Implementado | `system.py:marginalizar_columnas()` |
| Construcción de subsistema (alcance + mecanismo) | ✅ Implementado | `system.py:construir_subsistema()` |
| Independencia condicional / producto tensorial | ✅ Implementado | `system.py:distribucion_estado_inicial()` |
| Distancia Hamming | ✅ Implementado | `emd.py:hamming_distance()` |
| EMD (Earth Mover's Distance) | ✅ Implementado | `emd.py:emd_pyphi()` con `ot.emd2` |
| Bipartición k=2, fórmula P(u,v)=2^(u+v-1)−1 | ✅ Implementado | `geometric.py:_identificar_candidatas()` |
| k-particiones k=3,4,5 | ✅ Implementado | `qnodes.py:KQNodes`, `geometric.py:KGeoMIPKPartition` |

---

### 1.2 Estrategia Geométrica (PDF `1_Guía_Proyecto_ADAV1_2_0.pdf`)

| Requisito (Sección) | Estado | Observaciones |
|---------------------|--------|---------------|
| **§1.1** Representación del sistema como hipercubo | ✅ Implementado | BFS por niveles de Hamming |
| **§2.1** Descomposición en tensores elementales | ✅ Implementado | `system.py:obtener_tensores()` |
| **§2.2.1** Distancia Hamming y adyacencia | ✅ Implementado | `emd.py:hamming_distance()` |
| **§2.2.2** Invariancia dimensional / simetrías | ❌ No implementado | Solo mencionado como optimización futura |
| **§3.1** Función de costo `t(i,j) = γ·(|X[i]-X[j]| + Σt(k,j))` | ✅ Implementado | `geometric.py:bfs_desde()` dentro de `_calcular_tensor_paralelo()` |
| **§3.1.1** Factor de decrecimiento exponencial `γ = 2^(-d)` | ✅ Implementado | `gamma = 2.0 ** (-d)` (línea 34) |
| **§3.1.2** BFS modificado para exploración del espacio | ✅ Implementado | BFS vectorizado (sin recursión) |
| **§3.2.1** Distribuciones marginales como proyecciones | ✅ Implementado | `geometric.py:_distribucion_particionada()` |
| **§4.2** Verificación contra Tabla 4.2 del documento | ✅ Implementado | `main.py:verificar_tabla_n3()` |
| **§5.1.1** Clase `GeometricSIA` heredando de `SIA` | ✅ Implementado | `KGeoMIP` + alias `GeometricSIA` |
| **§5.1.1** Método `aplicar_estrategia()` | ✅ Implementado | `geometric.py:KGeoMIP.aplicar_estrategia()` |
| **§5.1.2** Método `calcular_transicion_coste()` (nombrado explícitamente) | ⚠️ Parcial | Implementado como `bfs_desde()` local, no expuesto como método público de clase |
| **§5.1.2** Método `identificar_candidatos()` (nombrado explícitamente) | ⚠️ Parcial | Implementado como `_identificar_candidatas()` (privado, nombre diferente) |
| **§5.1.2** Caché de cálculos (memoización) | ✅ Implementado | `qnodes.py:_cache_dist` |
| **§5.1.2** Tabla de transiciones sparse | ✅ Implementado | Modo sparse activado para N>8 |
| **§5.2.1** Conjunto de pruebas `tests/PruebasIniciales.xlsx` | ❌ Ausente | El archivo Excel **no existe** en el repositorio |
| **§5.2.1** Comparación con PyPhi (referencia estándar) | ❌ No implementado | No hay integración con PyPhi |
| **§5.2.2** Tasa de acierto exacto | ❌ No implementado | No hay lógica de comparación automática |
| **§5.2.2** Error relativo en Φ (`E_rel`) | ❌ No implementado | No se calcula `E_rel` |
| **§5.2.2** Distancia estructural (Jaccard) entre biparticiones | ❌ No implementado | No implementado |
| **§5.2.2** Speedup relativo vs. PyPhi (`S_rel`) | ❌ No implementado | No se mide ni reporta |
| **§5.2.2** Uso de memoria del algoritmo | ❌ No implementado | No se registra memoria peak |

---

## 2. Lo que está Implementado (Detalle)

### 2.1 Núcleo Algorítmico

#### `src/models/system.py` — Modelo del Sistema
- **Carga de TPM** desde CSV con detección automática de encabezado y conversión estado-estado → estado-nodo.
- **Inversión de columnas** para corregir convención documental (columna A del CSV = variable de mayor índice internamente).
- **Soporte PySpark** para archivos >40 MB (N≥20), con fallback a numpy si Spark no está disponible.
- **`construir_subsistema(alcance, mecanismo)`**: pipeline completo de condicionamiento → marginalización filas → marginalización columnas, gestionando el caso mecanismo ⊃ alcance.

#### `src/controllers/strategies/geometric.py` — KGeoMIP

**Clase `KGeoMIP` (alias `GeometricSIA`)** — bipartición k=2:
1. Obtiene tensores elementales de la TPM.
2. Calcula tabla de costos T con BFS vectorizado por niveles de Hamming:
   - **Nivel 1** (d=1): `costo = 0.5 × |X[i] - X[j]|`
   - **Niveles 2..n** (d≥2): `costo = γ × (|X[i]-X[j]| + Σ costos_vecinos)`
   - **Modo sparse** para N>8: solo calcula la fila del estado inicial.
   - **Paralelismo real** con `multiprocessing.Pool`, un proceso por variable.
3. Identifica candidatas: ceros de la tabla T (independencia causal) + exhaustivo para N≤15 + heurística para N>15.
4. Evalúa con `emd_pyphi` sparse y retorna la bipartición con menor Φ.

**Clase `KGeoMIPKPartition`** — k-particiones k=2,3,4,5:
- k=2: delega a `KGeoMIP` exacto.
- k≥3, n≤6: Branch & Bound con ruptura de simetría canónica.
- k≥3, n>6: Greedy por importancia (suma costos T) + búsqueda local por intercambios.

#### `src/controllers/strategies/qnodes.py` — KQNodes

**Clase `KQNodes`** — k-particiones k=2,3,4,5:
- k=2, n≤20: exhaustivo Stirling.
- k≥3, n≤7: exhaustivo con números de Stirling del segundo tipo.
- k>2, n>7: Greedy (round-robin) + fase de movimientos + fase de intercambios.
- **Memoización de distribuciones marginales**: `frozenset(vars)` → distribución; speedup real ~82× para N=10,k=3.
- Φ normalizado: `phi_raw / n` → garantiza rango [0,1].

#### `src/utils/emd.py` — EMD Sparse
- Implementación sparse: construye matriz de costos solo para estados con p>0.
- Usa `ot.emd2` (POT library) para resolver el problema de transporte óptimo.
- Soporta hasta 500 estados en soporte por distribución (configurable).

### 2.2 Infraestructura y Pruebas

#### `main.py`
- Suites de pruebas para N=3, 5, 10, 15, 20, 22, 25 (49 casos por N).
- **Verificación automática de Tabla 4.2** del documento para N=3.
- Guardado incremental de resultados en CSV (evita pérdidas por timeout).
- Timeout de 600 s para KQNodes en subsistemas grandes (n≥9).
- Modo CLI por argumento: `suite_n10`, `suite_n15`, `suite_n20`, `suite_n22`, `suite_n25`.

#### `interfaz.py` — GUI Tkinter
- Interfaz científica dark-mode completamente funcional.
- Configuraciones predefinidas para N=10, 15, 20, 22, 25.
- Ejecución en hilo separado con cola de mensajes (no bloquea UI).
- Soporte para ambas estrategias (KGeoMIP y KQNodes).

#### `src/utils/metrics.py`
- Registro de resultados (`RegistroMetricas`) con exportación a CSV.
- Campos: N, estado inicial, bipartición, Φ, tiempo, estrategia.

#### Datos de entrada
| Archivo | Estado |
|---------|--------|
| `data/N3C.csv` | ✅ Presente |
| `data/N5C.csv` | ✅ Presente |
| `data/N10C.csv` | ✅ Presente |
| `data/N15C.csv` | ✅ Presente |
| `data/N20C.csv` | ❌ Ausente |
| `data/N22C.csv` | ❌ Ausente |
| `data/N25C.csv` | ❌ Ausente |

#### Resultados generados
Los archivos en `results/` confirman que las suites N=10, N=15 y N=20 se ejecutaron exitosamente:
- `results/resultados_suite_N10.csv`, `results/resultados_suite_N15.csv`, `results/resultados_suite_N20.csv`
- Archivos de log y resultados segmentados por estrategia.

---

## 3. Lo que Falta (Brechas Críticas)

### 3.1 Archivo de Pruebas Oficial (`tests/PruebasIniciales.xlsx`) — CRÍTICO
El documento §5.2.1 exige explícitamente este archivo como conjunto de pruebas canónico con **biparticiones óptimas precalculadas por PyPhi**. Sin él es imposible:
- Validar la tasa de acierto exacto.
- Calcular el error relativo en Φ.
- Medir distancia estructural (Jaccard).

**Impacto:** Sin este archivo no es posible demostrar cumplimiento de los umbrales de calidad del §5.2.2 (Excelente >90%, Bueno >80%, Aceptable >70%).

### 3.2 Integración con PyPhi — CRÍTICO
El documento establece PyPhi como referencia estándar ("standar" [sic]) para validación. La brecha tiene dos dimensiones:
- No se ejecuta PyPhi como oráculo para comparación.
- No se implementa ningún módulo de benchmark automatizado.

### 3.3 Métricas de Desempeño Formales (§5.2.2) — IMPORTANTE
Las siguientes métricas están definidas en el documento pero **no tienen código de cálculo**:

```
E_rel = |Φ_óptimo - Φ_encontrado| / Φ_óptimo        (Error relativo en Φ)
S_rel = T_PyPhi / T_Geometric                         (Speedup relativo)
d_Jaccard(P_found, P_opt)                             (Distancia estructural)
```
No existe ningún módulo que compute estas métricas sobre los CSV de resultados.

### 3.4 Métodos Públicos con Nombres del Documento — MENOR
El §5.1.2 especifica los métodos auxiliares por nombre:
- `calcular_transicion_coste(...)` → existe como función local `bfs_desde()` dentro de `_calcular_tensor_paralelo`. No es un método de instancia accesible externamente.
- `identificar_candidatos(...)` → existe como `_identificar_candidatas()` (privado, nombre diferente al especificado).

### 3.5 Datos de Entrada para N=20, 22, 25 — IMPORTANTE
Los archivos `N20C.csv`, `N22C.csv`, `N25C.csv` no están presentes. La interfaz los referencia como configuraciones predefinidas, pero sin ellos las suites para N≥20 generan subsistemas sintéticos aleatorios en lugar de usar la topología del documento.

### 3.6 Explotación de Simetrías del Hipercubo — BAJA PRIORIDAD
El §2.2.2 describe la explotación de automorfismos del hipercubo (permutaciones + complementaciones) para reducir el espacio de búsqueda. No está implementado; el algoritmo evalúa clases equivalentes de forma redundante.

### 3.7 Tests Unitarios Completos — MENOR
- `tests/test_geometric.py` tiene un método `test_tabla_costos_variable_A` pero llama a `geo._calcular_tabla_costos()` (nombre antiguo) — incompatible con la API actual que usa `_calcular_tabla_costos_paralelo()`.
- No hay tests para: `System`, `KQNodes`, `emd_pyphi`, `marginalizar_filas`, `marginalizar_columnas`.

---

## 4. Análisis de Complejidad Real vs. Teórica

| Aspecto | Documento (§5.1.1) | Implementación Real |
|---------|-------------------|---------------------|
| Complejidad temporal | O(n·2^n) | O(n·2^n) para modo sparse; O(n·2^(2n)) para tabla densa (N≤8) |
| Complejidad espacial | O(1) para tabla completa | O(n·2^n) sparse (fila por variable); O(n·4^n) densa (N≤8) |
| Paralelismo | Mencionado como optimización | ✅ Implementado con `multiprocessing.Pool` |
| Simetrías | Mencionado como optimización | ❌ No implementado |
| Memoización | No mencionada explícitamente | ✅ Implementada en KQNodes (~82× speedup) |

---

## 5. Tabla de Prioridades para Completar el Proyecto

| Prioridad | Tarea | Esfuerzo Estimado |
|-----------|-------|-------------------|
| 🔴 Alta | Obtener/generar `tests/PruebasIniciales.xlsx` con resultados PyPhi de referencia | Alto |
| 🔴 Alta | Implementar módulo de validación: tasa de acierto, `E_rel`, distancia Jaccard | Medio |
| 🟠 Media | Conseguir o generar `data/N20C.csv`, `N22C.csv`, `N25C.csv` | Bajo (datos) |
| 🟠 Media | Exponer `calcular_transicion_coste()` e `identificar_candidatos()` como métodos públicos con los nombres exactos del documento | Bajo |
| 🟠 Media | Corregir test `tests/test_geometric.py` (renombrar llamada a `_calcular_tabla_costos_paralelo`) | Muy bajo |
| 🟡 Baja | Calcular y reportar `S_rel` (speedup vs. PyPhi) y uso de memoria | Medio |
| 🟡 Baja | Añadir tests unitarios para `System`, `KQNodes` y `emd_pyphi` | Medio |
| 🟢 Opcional | Implementar reducción por simetrías del hipercubo (§2.2.2) | Alto |

---

## 6. Evaluación Global

| Dimensión | Calificación | Justificación |
|-----------|-------------|---------------|
| **Núcleo algorítmico** | ✅ 95% | BFS, función de costo, tabla T, candidatas, EMD: todos correctos y verificados contra Tabla 4.2 |
| **Modelo del sistema** | ✅ 95% | TPM, marginalización, condicionamiento, subsistemas: completos y correctos |
| **k-Particiones** | ✅ 90% | k=2,3,4,5 implementados; exacto para n≤7(k≥3) y heurístico para n>7 |
| **Escalabilidad** | ✅ 85% | Paralelo con Pool, sparse para N>8, Spark para N≥20 (sin datos N≥20) |
| **Validación formal** | ❌ 10% | Sin PyPhi, sin Excel de referencia, sin métricas E_rel/Jaccard/S_rel |
| **Pruebas unitarias** | ⚠️ 30% | Un test roto; faltan tests para la mayoría de módulos |
| **Interfaz de usuario** | ✅ 90% | GUI funcional, dark-mode, opciones completas |
| **Datos de prueba** | ⚠️ 50% | N=3,5,10,15 presentes; N=20,22,25 ausentes |

---

## 7. Conclusión

El proyecto cumple sólidamente la parte **computacional y algorítmica** del enunciado. La implementación de `KGeoMIP` es correcta (verificada numéricamente contra la Tabla 4.2 del documento), eficiente (paralelismo, modo sparse) y escalable (hasta N=25 con datos sintéticos). El principal déficit no es de código sino de **infraestructura de validación**: sin el archivo `PruebasIniciales.xlsx` con respuestas de referencia de PyPhi, no es posible demostrar formalmente que el algoritmo cumple los umbrales de calidad exigidos en §5.2.2. Esta es la brecha más urgente a cerrar.
