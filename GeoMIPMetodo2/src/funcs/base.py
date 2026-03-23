import numpy as np
from itertools import combinations
from numpy.typing import NDArray


def lil_endian(binary_str: str) -> int:
    """
    Convierte una cadena binaria en little-endian a su índice entero.
    
    En little-endian, el primer caracter es el bit de menor peso.
    Ejemplo: "001" → 0*2^0 + 0*2^1 + 1*2^2 = 4
    
    Esto es diferente al int() estándar de Python que usa big-endian:
    int("001", 2) = 1, pero lil_endian("001") = 4.
    """
    return int(binary_str[::-1], 2)


def hamming_distance(a: int, b: int) -> int:
    """
    Calcula la distancia de Hamming entre dos estados representados como enteros.
    
    La distancia de Hamming es el número de bits distintos.
    Se calcula con XOR (bits que difieren quedan en 1) y luego contando los 1s.
    Ejemplo: hamming_distance(5, 3) → 5=101, 3=011, XOR=110 → 2 bits distintos.
    """
    return bin(a ^ b).count("1")


def get_neighbors(state: int, n: int) -> list[int]:
    """
    Retorna los estados adyacentes a distancia de Hamming 1.
    
    Son los estados que se obtienen flipeando exactamente un bit.
    Para el estado 'state' con 'n' variables, hay exactamente n vecinos.
    Ejemplo: get_neighbors(0b010, 3) → [0b011, 0b000, 0b110]
    """
    return [state ^ (1 << i) for i in range(n)]


def get_states_at_level(initial: int, level: int, n: int) -> list[int]:
    """
    Retorna todos los estados a distancia de Hamming 'level' desde 'initial'.
    
    Esto define los "niveles" del hipercubo que el algoritmo recorre BFS-style.
    Se generan flipeando exactamente 'level' bits del estado inicial.
    """
    states = []
    for bits_to_flip in combinations(range(n), level):
        # Crear máscara con los bits a flipear
        mask = 0
        for bit in bits_to_flip:
            mask |= (1 << bit)
        states.append(initial ^ mask)
    return states


def get_labels(n: int) -> list[str]:
    """
    Genera etiquetas para las variables del sistema: ['A', 'B', 'C', ...].
    Para más de 26 variables usa 'A0', 'A1', etc.
    """
    if n <= 26:
        return [chr(ord('A') + i) for i in range(n)]
    return [f"A{i}" for i in range(n)]


def get_restricted_combinations(elements: list, min_size: int = 1) -> list[list]:
    """
    Genera todas las combinaciones no vacías de una lista de elementos,
    excluyendo el conjunto completo (para evitar particiones triviales).
    
    Usado para generar los candidatos de bipartición de mecanismo y alcance.
    """
    result = []
    for size in range(min_size, len(elements)):
        for combo in combinations(elements, size):
            result.append(list(combo))
    return result


def indices_from_mask(mask: str) -> list[int]:
    """
    Convierte una máscara binaria a la lista de índices donde vale '1'.
    Ejemplo: "10110" → [0, 2, 3]
    """
    return [i for i, bit in enumerate(mask) if bit == '1']


def mask_from_indices(indices: list[int], length: int) -> str:
    """
    Convierte una lista de índices a una máscara binaria de longitud 'length'.
    Ejemplo: indices=[0,2], length=4 → "1010"
    """
    mask = ['0'] * length
    for i in indices:
        mask[i] = '1'
    return ''.join(mask)