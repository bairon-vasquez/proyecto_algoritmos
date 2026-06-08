"""
Tests del modelo System: inicialización, marginalizaciones, condicionamiento,
construcción de subsistemas y distribución de estado inicial.
"""
import numpy as np
import pytest
from src.models.system import System


# TPM del caso de estudio (sección 4.1 del documento 2), sin inversión de columnas.
# Misma convención que test_geometric.py: se pasa directamente al constructor.
TPM_DOC = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
], dtype=np.float64)


def test_sistema_inicializacion():
    """System inicializa n, etiquetas y shape correctamente."""
    s = System(TPM_DOC, "000")
    assert s.n == 3
    assert s.etiquetas == ["A", "B", "C"]
    assert s.tpm.shape == (8, 3)
    assert s.num_estados == 8
    assert s.estado_inicial == "000"


def test_distribucion_estado_inicial():
    """P(X_t+1 | estado 000) = [1,0,0,0,0,0,0,0] para TPM trivial."""
    s = System(TPM_DOC, "000")
    dist = s.distribucion_estado_inicial()
    assert dist.shape == (8,)
    expected = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(dist, expected, atol=1e-10)


def test_marginalizar_filas():
    """Marginalizar variable B (índice 1) reduce filas a 4 y ajusta etiquetas."""
    s = System(TPM_DOC, "000")
    mf = s.marginalizar_filas([1])
    assert mf.tpm.shape == (4, 3)
    assert mf.n == 2
    assert mf.etiquetas == ["A", "C"]
    # Promedios esperados: filas {0,2} → [0,0.5,0], {1,3} → [1,0.5,0], etc.
    expected = np.array([
        [0.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.5, 1.0],
    ])
    np.testing.assert_allclose(mf.tpm, expected, atol=1e-10)


def test_marginalizar_columnas():
    """Marginalizar columna B (índice 1) reduce columnas a 2."""
    s = System(TPM_DOC, "000")
    mc = s.marginalizar_columnas([1])
    assert mc.tpm.shape == (8, 2)
    assert mc.n == 2
    assert mc.etiquetas == ["A", "C"]
    # Columnas 0 y 2 de TPM_DOC
    np.testing.assert_allclose(mc.tpm, TPM_DOC[:, [0, 2]], atol=1e-10)


def test_condicionar():
    """Condicionar B=0 mantiene 4 estados y columnas A,C."""
    s = System(TPM_DOC, "000")
    cd = s.condicionar([1], [0])
    assert cd.tpm.shape == (4, 2)
    assert cd.n == 2
    assert cd.etiquetas == ["A", "C"]
    # Estados con bit1=0: índices 0,1,4,5; columnas 0 y 2
    expected = TPM_DOC[[0, 1, 4, 5]][:, [0, 2]]
    np.testing.assert_allclose(cd.tpm, expected, atol=1e-10)


def test_construir_subsistema_alc_menor():
    """Subsistema alcance=['A','B'] mecanismo=['A','B','C'] → shape=(4,2)."""
    s = System(TPM_DOC, "000")
    sub = s.construir_subsistema(["A", "B"], ["A", "B", "C"])
    assert sub.tpm.shape == (4, 2)
    assert sub.n == 2
    assert sub.etiquetas == ["A", "B"]


def test_construir_subsistema_igual():
    """Subsistema con alcance == mecanismo == sistema completo no cambia shape."""
    s = System(TPM_DOC, "000")
    sub = s.construir_subsistema(["A", "B", "C"], ["A", "B", "C"])
    assert sub.tpm.shape == (8, 3)
    assert sub.n == 3


def test_desde_csv():
    """desde_csv carga N3C.csv, aplica inversión de columnas y retorna n=3."""
    s = System.desde_csv("data/N3C.csv", "000")
    assert s.n == 3
    assert s.tpm.shape == (8, 3)
    assert s.etiquetas == ["A", "B", "C"]
    dist = s.distribucion_estado_inicial()
    assert dist.shape == (8,)
    assert abs(dist.sum() - 1.0) < 1e-9


if __name__ == "__main__":
    test_sistema_inicializacion()
    test_distribucion_estado_inicial()
    test_marginalizar_filas()
    test_marginalizar_columnas()
    test_condicionar()
    test_construir_subsistema_alc_menor()
    test_construir_subsistema_igual()
    test_desde_csv()
    print("\n✓ Todos los tests de System pasaron")
