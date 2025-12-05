from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def calculate_anomaly(image_bytes):
    """
    - carregar imagem original
    - carregar imagem reconstruída
    - converter para CIELAB
    - calcular ΔE2000 pixel a pixel
    - gerar métricas como Top1% e Top2%
    """
    return {"top2_mean": 0.0, "top1_energy": 0.0}
