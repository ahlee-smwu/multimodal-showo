from huggingface_hub import snapshot_download
import os
import shutil

# 1. Hugging Face 레포 ID
repo_id = "showlab/show-o2-7B"

# 2. 전체 파일 다운로드 (캐시에 저장됨)
local_dir = snapshot_download(repo_id=repo_id)

# ✅ 3. 복사할 경로: 'show-o2-1.5B/' 폴더로 지정
target_dir = os.path.join(os.getcwd(), repo_id.split("/")[-1])  # "show-o2-1.5B"

# 4. 기존에 있다면 제거
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# 5. 복사
shutil.copytree(local_dir, target_dir)

print(f"✅ Repository '{repo_id}' downloaded and copied to: {target_dir}")
