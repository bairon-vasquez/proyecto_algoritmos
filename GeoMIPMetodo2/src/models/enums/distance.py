from enum import Enum

class Distance(Enum):
    """
    Tipos de distancia soportados para calcular costos entre estados.
    
    HAMMING: número de bits distintos entre dos estados binarios.
    Es la métrica central del framework GeoMIP.
    """
    HAMMING = "hamming"