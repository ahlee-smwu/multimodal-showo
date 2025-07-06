from datasets import load_dataset
import cv2
import os

# 저장 폴더
save_dir = ""
os.makedirs(save_dir, exist_ok=True)
print('line')

# 스트리밍 모드로 데이터셋 로드
dataset = load_dataset("ZhangHanXD/AvaMERG", split="train", streaming=True)

# 3개 샘플만 처리
for i, example in enumerate(dataset):
    if i >= 3:
        break

    video_url = example['video_v5_0']  # 컬럼 이름이 정확한지 확인 필요
    video_id = example['video_id'] if 'video_id' in example else f"sample_{i}"

    print(f"Processing sample {i+1}: {video_id}")

    # OpenCV는 URL 직접 처리 불가하므로, 먼저 로컬에 저장해야 함
    video_path = os.path.join(save_dir, f"{video_id}.mp4")

    # URL에서 비디오 파일 다운로드 (requests 이용)
    import requests
    if not os.path.exists(video_path):
        print(f"Downloading video to {video_path} ...")
        r = requests.get(video_url, stream=True)
        with open(video_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"⚠️ {video_id}: 영상 프레임 수가 0입니다. 건너뜀.")
        cap.release()
        continue

    middle_frame_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

    ret, frame = cap.read()
    if not ret:
        print(f"❌ {video_id}: 프레임 읽기 실패.")
        cap.release()
        continue

    # 이미지 저장 (BGR 그대로)
    output_path = os.path.join(save_dir, f"{video_id}.png")
    cv2.imwrite(output_path, frame)

    print(f"✅ 저장 완료: {output_path}")
    cap.release()
