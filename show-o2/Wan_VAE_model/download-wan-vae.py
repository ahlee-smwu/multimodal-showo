from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "Wan-AI/Wan2.1-T2V-14B"
filename = "Wan2.1_VAE.pth"

# 1) 캐시에 다운로드
cached_path = hf_hub_download(repo_id=repo_id, filename=filename)

# 2) 현재 작업 디렉토리로 복사
dest_path = os.path.join(os.getcwd(), filename)
print(dest_path)
shutil.copyfile(cached_path, dest_path)

# 3) 캐시 파일 삭제
if os.path.exists(cached_path):
    os.remove(cached_path)

print(f"File copied to current directory: {dest_path}")
print(f"Cache file deleted: {cached_path}")