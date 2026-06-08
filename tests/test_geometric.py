"""
Tests de validación del algoritmo GeometricSIA.
Verifica con el caso de estudio exacto del documento 2 (capítulo 4).
"""
import numpy as np
import pytest
from src.models.system import System
from src.controllers.strategies.geometric import GeometricSIA
from src.utils.emd import emd_pyphi


# TPM del caso de estudio del documento 2 (sección 4.1)
# Sistema ABC con estado inicial 000
# Tensores: A=[0,1,0,1,0,1,0,1], B=[0,0,1,1,0,0,1,1], C=[0,0,0,0,1,1,1,1]
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


def test_tabla_costos_variable_A():
    """
    Valida los costos calculados para variable A desde estado 000.
    Valores esperados de la Tabla 4.2 del documento 2.
    """
    sys = System(TPM_DOC, "000")
    geo = GeometricSIA(sys)
    geo.tensores = sys.obtener_tensores()
    geo.tabla_costos = geo._calcular_tabla_costos_paralelo()

    T_A = geo.tabla_costos[0]  # variable A
    idx = 0  # estado 000

    # Tabla 4.2: costos desde 000 para variable A
    assert abs(T_A[idx, 0]) < 1e-6,  "t_A(000,000) debe ser 0"
    assert abs(T_A[idx, 4]) < 1e-6,  "t_A(000,100) debe ser 0"
    assert abs(T_A[idx, 2]) < 1e-6,  "t_A(000,010) debe ser 0"
    assert abs(T_A[idx, 6]) < 1e-6,  "t_A(000,110) debe ser 0"
    assert abs(T_A[idx, 1] - 0.5) < 1e-6,   "t_A(000,001) debe ser 0.5"
    assert abs(T_A[idx, 5] - 0.375) < 1e-6, "t_A(000,101) debe ser 0.375"
    assert abs(T_A[idx, 3] - 0.375) < 1e-6, "t_A(000,011) debe ser 0.375"
    assert abs(T_A[idx, 7] - 0.21875) < 1e-4, "t_A(000,111) debe ser ~0.219"
    print("✓ Tabla costos variable A correcta")


def test_tabla_costos_variable_B():
    """Valida costos para variable B. Tabla 4.2 del documento."""
    sys = System(TPM_DOC, "000")
    geo = GeometricSIA(sys)
    geo.tensores = sys.obtener_tensores()
    geo.tabla_costos = geo._calcular_tabla_costos_paralelo()

    T_B = geo.tabla_costos[1]
    idx = 0

    assert abs(T_B[idx, 0]) < 1e-6,  "t_B(000,000)=0"
    assert abs(T_B[idx, 4]) < 1e-6,  "t_B(000,100)=0"
    assert abs(T_B[idx, 2] - 0.5) < 1e-6,   "t_B(000,010)=0.5"
    assert abs(T_B[idx, 1]) < 1e-6,  "t_B(000,001)=0"
    assert abs(T_B[idx, 5]) < 1e-6,  "t_B(000,101)=0"
    assert abs(T_B[idx, 6] - 0.375) < 1e-6, "t_B(000,110)=0.375"
    assert abs(T_B[idx, 3] - 0.375) < 1e-6, "t_B(000,011)=0.375"
    print("✓ Tabla costos variable B correcta")


def test_tabla_costos_variable_C():
    """Valida costos para variable C. Tabla 4.2 del documento."""
    sys = System(TPM_DOC, "000")
    geo = GeometricSIA(sys)
    geo.tensores = sys.obtener_tensores()
    geo.tabla_costos = geo._calcular_tabla_costos_paralelo()

    T_C = geo.tabla_costos[2]
    idx = 0

    assert abs(T_C[idx, 0]) < 1e-6,  "t_C(000,000)=0"
    assert abs(T_C[idx, 4] - 0.5) < 1e-6,   "t_C(000,100)=0.5"
    assert abs(T_C[idx, 2]) < 1e-6,  "t_C(000,010)=0"
    assert abs(T_C[idx, 1]) < 1e-6,  "t_C(000,001)=0"
    assert abs(T_C[idx, 6] - 0.375) < 1e-6, "t_C(000,110)=0.375"
    assert abs(T_C[idx, 5] - 0.375) < 1e-6, "t_C(000,101)=0.375"
    assert abs(T_C[idx, 3]) < 1e-6,  "t_C(000,011)=0"
    print("✓ Tabla costos variable C correcta")


def test_estrategia_completa():
    """Verifica que la estrategia completa retorna resultado válido."""
    sys = System(TPM_DOC, "000")
    geo = GeometricSIA(sys)
    resultado = geo.ejecutar()

    assert 'biparticion' in resultado
    assert 'phi' in resultado
    assert resultado['phi'] >= 0
    p1, p2 = resultado['biparticion']
    assert len(p1) > 0
    assert len(p2) > 0
    assert set(p1) | set(p2) == set(range(3))
    assert set(p1) & set(p2) == set()
    geo.imprimir_resultado()
    print("✓ Estrategia completa correcta")


def test_emd_simetria():
    """Verifica propiedad de simetría de la EMD."""
    u = np.array([0.5, 0.3, 0.1, 0.1])
    v = np.array([0.1, 0.1, 0.3, 0.5])
    assert abs(emd_pyphi(u, v) - emd_pyphi(v, u)) < 1e-8
    print("✓ EMD simétrica")


def test_emd_identicas():
    """EMD entre distribuciones idénticas debe ser 0."""
    u = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(emd_pyphi(u, u)) < 1e-8
    print("✓ EMD = 0 para distribuciones idénticas")


if __name__ == "__main__":
    test_tabla_costos_variable_A()
    test_tabla_costos_variable_B()
    test_tabla_costos_variable_C()
    test_estrategia_completa()
    test_emd_simetria()
    test_emd_identicas()
    print("\n✓ Todos los tests pasaron")