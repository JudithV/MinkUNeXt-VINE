from thop import profile
from config import PARAMS
from model.minkunext import model

# Entrada de prueba
input_tensor = torch.randn(1, 3, 224, 224).to(PARAMS.device)

# Cálculo de FLOPs y parámetros
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
print(f"Parámetros: {params/1e6:.2f} millones")

