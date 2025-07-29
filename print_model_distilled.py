import torch

# 모델 경로 설정
model_path = "/home/sichoi/multimodal-showo/pre-distilled_semantic_layers.pt"
output_txt_path = "model_layers_pre-distilled_semantic.txt"

# state_dict 로드
state_dict = torch.load(model_path, map_location="cpu")

# 파일로 저장
with open(output_txt_path, "w") as f:
    f.write("=== Layer names and shapes from pytorch_model.bin ===\n\n")
    for name, param in state_dict.items():
        shape = tuple(param.shape)
        line = f"{name}: {shape}"
        print(line)
        f.write(line + "\n")

print(f"\n✅ 모델 파라미터 정보가 '{output_txt_path}'에 저장되었습니다.")
