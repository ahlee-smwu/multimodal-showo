# coding=utf-8

# Copyright 2025 NUS Show Lab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import wandb
import torch
import numpy as np
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_config, flatten_omega_conf, denorm_vid, get_hyper_params, path_to_llm_name, load_state_dict
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import cv2

# seed_value = 42
# set_seed(seed_value)
if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")

if __name__ == '__main__':
    config = get_config()
    
    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name,
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.model.weight_type == "bfloat16":
        weight_type = torch.bfloat16
    elif config.model.weight_type == "float32":
        weight_type = torch.float32
    else:
        raise NotImplementedError

    # VQ model for processing video into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path,
                                                        add_showo_tokens=True,
                                                        return_showo_token_ids=True,
                                                        llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model.model_path)
        model.load_state_dict(state_dict)
    
    model.to(weight_type)
    model.eval()

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_t2i_image_tokens += 1  # 729+1
        config.dataset.preprocessing.num_mmu_image_tokens += 1  # 729+1
        config.dataset.preprocessing.num_video_tokens += 1  # 3645+1

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    # Get video-specific hyperparameters
    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
    = get_hyper_params(config, text_tokenizer, showo_token_ids, is_video=True)

    # load users passed arguments
    batch_size = config.batch_size
    guidance_scale = config.guidance_scale
    config.transport.num_inference_steps = config.num_inference_steps

    if config.get("validation_prompts_file", None) is not None:
        validation_prompts_file = config.validation_prompts_file
        # load from users passed arguments

    transport = create_transport(
        path_type=config.transport.path_type,   # linear
        prediction=config.transport.prediction, # velocity
        loss_weight=config.transport.loss_weight,   # null
        train_eps=config.transport.train_eps,   # null
        sample_eps=config.transport.sample_eps, # null
        snr_type=config.transport.snr_type, # lognorm
        do_shift=config.transport.do_shift, # True
        seq_len=num_video_tokens,   #
    )  # default: velocity;

    sampler = Sampler(transport)

    # Video generation parameters
    T = 5  # Time dimension for video

    for step in tqdm(range(0, len(validation_prompts), config.batch_size)):
        prompts = validation_prompts[step:step + config.batch_size] # 배치 단위로 프롬프트 처리
        
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
            prepare_gen_input(
                prompts, text_tokenizer, num_video_tokens, bos_id, eos_id, bov_id, eov_id, pad_id, vid_pad_id,
                max_text_len, device
            )

        # Initialize video latent tensor with time dimension 비디오 latent tensor 초기화
        # 5D 텐서 생성 (batch, channels, time, height, width)
        z = torch.randn((
            len(prompts),  # 배치 크기
            image_latent_dim,  # latent channel (16)
            T,  # 시간 차원 (5 프레임)
            latent_height * patch_size,  # 높이 (54)
            latent_width * patch_size  # 너비 (54)
        )).to(weight_type).to(device)

        # 이미지는 4D: (batch, channels, height, width)
        # 비디오는 5D: (batch, channels, time, height, width) ← 시간 축 추가

        video_seq_len = batch_text_tokens.size(1)

        if guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
            
            # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
            # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
            # omni_mask_fn = omni_attn_mask(modality_positions)
            # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
            #                              KV_LEN=max_seq_len, device=device)
            # or use naive omni attention mask, which is more stable

            # block_mask = omni_attn_mask_naive(text_tokens.size(0),
            #                                 max_seq_len,
            #                                 modality_positions,
            #                                 device).to(weight_type)

            video_seq_len = batch_text_tokens.size(1)   # 동적계산 # 고정된 max_seq_len일 때 Target: [2, 12, 3650, 3650] vs Actual: [2, 1, 1024, 1024] 이라고 에러남

            # 어텐션 마스크 생성
            block_mask = omni_attn_mask_naive(text_tokens.size(0),  # batch size
                                              video_seq_len,  # 동적인 실제 시퀀스 길이 사용
                                              modality_positions,
                                              device).to(weight_type)
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions
            # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
            # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)  
            # omni_mask_fn = omni_attn_mask(modality_positions)
            # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
            #                              KV_LEN=max_seq_len, device=device)
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                            max_seq_len,
                                            modality_positions,
                                            device).to(weight_type)

        # 모델 추론 인자 설정
        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=video_seq_len, # max_seq_len,
            guidance_scale=guidance_scale
        )

        # ODE 샘플링 함수 설정
        sample_fn = sampler.sample_ode(
            sampling_method=config.transport.sampling_method,   # "euler"
            num_steps=config.transport.num_inference_steps,
            atol=config.transport.atol, # 절대오차
            rtol=config.transport.rtol, # 상대오차
            reverse=config.transport.reverse,   # 역방향 여부
            time_shifting_factor=config.transport.time_shifting_factor
        )

        # 실제 생성; ODE solver
        # noise z에서 video latent vector로 변환
        print(f"z.shape: {z.shape}")    # torch.Size([2, 16, 5, 54, 54])
        samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
        print(f"samples.shape: {samples.shape}")    # torch.Size([2, 16, 5, 54, 54])

        if guidance_scale > 0:  # Classifier-Free Guidance가 사용된 경우 조건부 결과만 선택
            samples = torch.chunk(samples, 2)[0]

        if config.model.vae_model.type == 'wan21':  # VAE 디코딩 (잠재 공간 → 픽셀 공간)
            videos = vae_model.batch_decode(samples)
        else:
            raise NotImplementedError

        # Convert to numpy arrays for video output  # 정규화 해제 ([-1, 1] → [0, 255])
        videos = denorm_vid(videos)

        # 비디오 차원 및 데이터 검증
        print(f"비디오 shape: {videos.shape}, dtype: {videos.dtype}")  # 비디오 shape: (1, 17, 3, 432, 432), dtype: uint8
        print(f"비디오 값 범위: [{videos.min():.2f}, {videos.max():.2f}]")

        # 데이터 범위 확인
        if videos.max() > 1.0:
            print("✅ 비디오는 이미 0-255 범위입니다")
        else:
            print("⚠️ 비디오를 0-255 범위로 변환합니다")
            videos = (videos * 255).clip(0, 255)
        videos = videos.astype(np.uint8)

        # 저장 디렉토리
        import os, cv2
        from datetime import datetime
        from PIL import Image

        output_dir, frames_dir = "generated_videos", "generated_videos/generated_frames"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        all_frame_images = []

        # 각 비디오 처리
        for i, video in enumerate(videos):
            # (T, C, H, W) → (T, H, W, C) 변환
            video_frames = video
            if video.shape[1] == 3:
                video_frames = video.transpose(0, 2, 3, 1)
            else:
                video_frames = video

            timestamp = datetime.now().strftime("%m%d_%H%M")
            # [:20] prompts[i] 문자열에서 앞부분 20글자까지만 사용
            safe_prompt = "".join(c for c in prompts[i][:20] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(
                ' ', '_')
            frame_folder = f"{frames_dir}/video_{timestamp}_step{step:03d}_batch{i}_{safe_prompt}"
            os.makedirs(frame_folder, exist_ok=True)

            # 프레임 저장 + wandb 등록
            for frame_idx, frame in enumerate(video_frames[:60]):  # 비디오 프레임 중 앞에서부터 최대 60프레임만 저장
                pil_image = Image.fromarray(frame.astype(np.uint8), mode='RGB')
                pil_image.save(f"{frame_folder}/frame_{frame_idx:03d}.png")
                all_frame_images.append(wandb.Image(
                    pil_image,
                    caption=f"Step{step} Batch{i} Frame{frame_idx}: {prompts[i][:30]}..."
                ))
                print(f"프레임 {frame_idx} 저장 성공, {frame_folder}")

            # 비디오 저장 (첫 번째 성공 코덱 사용)
            video_bgr = video_frames[..., ::-1]     # RGB → BGR
            height, width = video_bgr.shape[1:3]
            for codec_name, ext in [('XVID', '.avi'), ('mp4v', '.mp4')]:
                filename = f"{output_dir}/video_{timestamp}_step{step:03d}_batch{i}_{safe_prompt}{ext}"
                out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec_name), config.fps, (width, height))   # fps: 8
                if not out.isOpened():
                    continue
                for f in video_bgr:
                    out.write(f.astype(np.uint8))
                out.release()
                if os.path.exists(filename) and os.path.getsize(filename) > 1024:
                    print(f"비디오 저장 성공: {filename}")
                    break

        # wandb 프레임 로깅
        for b in range(0, len(all_frame_images), 20):
            wandb.log({f"Video frames batch {b // 20 + 1}": all_frame_images[b:b + 20]}, step=step)
        print(f"📤 Wandb 프레임 로깅: {len(all_frame_images)}개")
