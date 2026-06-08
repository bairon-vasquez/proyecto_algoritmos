"""
Tests de KQNodes: generador de k-particiones (Stirling) y búsqueda de MIP.
Verifica phi normalizado, consistencia phi_raw = phi_norm × n, y rangos válidos.
"""
import numpy as np
import pytest
from src.controllers.strategies.qnodes import _particionar_conjunto, KQNodes
from src.models.system import System


# ─────────────────────────────────────────────────────────────────────────────
# Tests de _particionar_conjunto (números de Stirling S(n,k))
# ─────────────────────────────────────────────────────────────────────────────

def test_particionar_k2_n3():
    """S(3,2) = 3 biparticiones de {0,1,2} en 2 partes no vacías."""
    partes = list(_particionar_conjunto([0, 1, 2], 2))
    assert len(partes) == 3
    # Cada partición debe tener exactamente 2 grupos
    for p in partes:
        assert len(p) == 2
        assert all(len(g) > 0 for g in p)
    # La unión de grupos debe ser {0,1,2}
    for p in partes:
        assert set(p[0]) | set(p[1]) == {0, 1, 2}
        assert set(p[0]) & set(p[1]) == set()


def test_particionar_k3_n3():
    """S(3,3) = 1: única tripartición de {0,1,2} es [[0],[1],[2]]."""
    partes = list(_particionar_conjunto([0, 1, 2], 3))
    assert len(partes) == 1
    assert partes[0] == [[0], [1], [2]]


def test_particionar_k2_n4():
    """S(4,2) = 7 biparticiones de {0,1,2,3}."""
    partes = list(_particionar_conjunto([0, 1, 2, 3], 2))
    assert len(partes) == 7
    for p in partes:
        assert set(p[0]) | set(p[1]) == {0, 1, 2, 3}
        assert set(p[0]) & set(p[1]) == set()


def test_particionar_k1():
    """k=1 siempre produce una sola partición con todos los elementos."""
    partes = list(_particionar_conjunto([0, 1, 2], 1))
    assert len(partes) == 1
    assert partes[0] == [[0, 1, 2]]


def test_particionar_k_mayor_n():
    """k > n no produce particiones (imposible tener partes no vacías)."""
    partes = list(_particionar_conjunto([0, 1], 3))
    assert len(partes) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests de KQNodes con datos reales
# ─────────────────────────────────────────────────────────────────────────────

def test_kqnodes_n3_phi_cero():
    """Sistema N3C completo (alcance=mecanismo=ABC) tiene phi=0 (sistema separable)."""
    s = System.desde_csv("data/N3C.csv", "000")
    kq = KQNodes(s)
    res = kq.aplicar_estrategia()
    assert res["phi"] < 1e-9, f"phi_norm esperado 0, obtenido {res['phi']}"
    assert res["phi_raw"] < 1e-9, f"phi_raw esperado 0, obtenido {res['phi_raw']}"


def test_kqnodes_phi_raw_consistencia():
    """phi_raw == phi_norm × n para el resultado global."""
    s = System.desde_csv("data/N3C.csv", "000")
    kq = KQNodes(s)
    res = kq.aplicar_estrategia()
    n = s.n
    assert abs(res["phi_raw"] - res["phi"] * n) < 1e-9


def test_kqnodes_phi_en_rango():
    """phi normalizado debe estar en [0, 1]."""
    s = System.desde_csv("data/N3C.csv", "000")
    kq = KQNodes(s)
    res = kq.aplicar_estrategia()
    assert 0.0 <= res["phi"] <= 1.0 + 1e-9


def test_kqnodes_biparticion_valida():
    """La bipartición retornada cubre todas las variables sin solapamiento."""
    s = System.desde_csv("data/N3C.csv", "000")
    kq = KQNodes(s)
    res = kq.aplicar_estrategia()
    bp = res["biparticion"]
    assert bp is not None
    todos = set()
    for parte in bp:
        assert len(parte) > 0
        for v in parte:
            assert v not in todos, f"variable {v} duplicada"
            todos.add(v)
    assert todos == set(range(s.n))


def test_kqnodes_subsistema_phi_no_cero():
    """Subsistema AB (alcance=mecanismo=AB) tiene phi_norm=0.25 (phi_raw=0.5)."""
    s = System.desde_csv("data/N3C.csv", "000")
    sub = s.construir_subsistema(["A", "B"], ["A", "B"])
    kq = KQNodes(sub)
    res = kq.aplicar_estrategia()
    assert abs(res["phi"] - 0.25) < 1e-6, f"phi_norm esperado 0.25, obtenido {res['phi']}"
    assert abs(res["phi_raw"] - 0.5) < 1e-6, f"phi_raw esperado 0.5, obtenido {res['phi_raw']}"


def test_kqnodes_n5_phi():
    """Sistema N5C completo tiene phi_norm=0.05 (phi_raw=0.25)."""
    s = System.desde_csv("data/N5C.csv", "00000")
    kq = KQNodes(s)
    res = kq.aplicar_estrategia()
    assert abs(res["phi"] - 0.05) < 1e-6, f"phi_norm esperado 0.05, obtenido {res['phi']}"
    assert abs(res["phi_raw"] - 0.25) < 1e-6, f"phi_raw esperado 0.25, obtenido {res['phi_raw']}"


def test_kqnodes_resultados_por_k():
    """aplicar_estrategia retorna resultados_por_k con al menos k=2."""
    s = System.desde_csv("data/N3C.csv", "000")
    kq = KQNodes(s)
    res = kq.aplicar_estrategia()
    assert "resultados_por_k" in res
    assert 2 in res["resultados_por_k"]
    res_k2 = res["resultados_por_k"][2]
    assert "phi" in res_k2
    assert "biparticion" in res_k2


if __name__ == "__main__":
    test_particionar_k2_n3()
    test_particionar_k3_n3()
    test_particionar_k2_n4()
    test_particionar_k1()
    test_particionar_k_mayor_n()
    test_kqnodes_n3_phi_cero()
    test_kqnodes_phi_raw_consistencia()
    test_kqnodes_phi_en_rango()
    test_kqnodes_biparticion_valida()
    test_kqnodes_subsistema_phi_no_cero()
    test_kqnodes_n5_phi()
    test_kqnodes_resultados_por_k()
    print("\n✓ Todos los tests de KQNodes pasaron")
