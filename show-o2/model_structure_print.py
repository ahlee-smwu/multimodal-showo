import torch
import torch.nn as nn
from models import Showo2Qwen2_5
from utils import get_config

config = get_config()

model = Showo2Qwen2_5(**config.model.showo)
state_dict = torch.load('1st_show-o2-1.5b-downstream-mixed-modality-432x432/pytorch_model.bin', map_location='cpu')
"""model.load_state_dict(state_dict)

# 구조 및 frozen 상태 동일하게 확인
print("모델 전체 구조:")
# print(model)

print("\n각 파라미터의 requires_grad 상태:")
for name, param in model.named_modules():
    print(f"{name:20s} | requires_grad: {param.requires_grad}")"""

layer_names = set()
for param_name in state_dict.keys():
    # 'xxx.weight' 또는 'xxx.bias' 등에서 마지막 '.' 앞까지만 추출
    layer_name = '.'.join(param_name.split('.')[:-1])
    layer_names.add(layer_name)

for name in layer_names:
    print(name)