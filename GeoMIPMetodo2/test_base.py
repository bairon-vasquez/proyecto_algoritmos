from src.models.enums.notation import Notation
from src.models.enums.distance import Distance
from src.constants.base import MAX_K_PARTITIONS, EPSILON
from src.funcs.base import lil_endian, hamming_distance, get_neighbors

print('lil_endian("001") =', lil_endian('001'))        # debe ser 4
print('hamming(5, 3) =', hamming_distance(5, 3))        # debe ser 2
print('vecinos de 0 (n=3):', get_neighbors(0, 3))       # debe ser [1, 2, 4]
print('Notacion:', Notation.LITTLE_ENDIAN.value)
print('MAX_K:', MAX_K_PARTITIONS)
print('EPSILON:', EPSILON)
print('Todo OK')