from enum import Enum

class Notation(Enum):
    """
    Define el orden de interpretación de bits en los estados binarios.
    
    LITTLE_ENDIAN: el bit de menor peso está a la izquierda.
    Por ejemplo, estado "001" → variable_0=0, variable_1=0, variable_2=1.
    Esta es la convención estándar usada en todo el framework.
    """
    LITTLE_ENDIAN = "little"
    BIG_ENDIAN    = "big"