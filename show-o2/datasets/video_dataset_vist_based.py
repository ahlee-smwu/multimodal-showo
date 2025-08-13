import collections
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from datasets.utils import video_transform, format_interleaved_sequence_video


class VISTVideoDataset(Dataset):
    """Dataset for interleaved Video-Text pairs."""

    def __init__(
        self,
        root: str,
        anno_path: str,
        text_tokenizer: Any,
        max_seq_len: int = 3840,
        image_size: int = 384,
        num_frames: int = 8,
        latent_height: int = 24,
        latent_width: int = 24,
        num_image_tokens: int = 576,
        cond_dropout_prob: float = 0.1,
        max_num_videos: int = 4,
        loader: Optional[Callable] = None,
        showo_token_ids: Optional[Dict[str, int]] = None,
        system: Tuple[str, str, str] = ("", "", ""),
    ):
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id'] #!!!!
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.bov_id = showo_token_ids['bov_id']
        self.eov_id = showo_token_ids['eov_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.vid_pad_id = showo_token_ids['vid_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.num_frames = num_frames
        self.data_type = "interleaved_data"
        self.transform = video_transform
        self.max_num_videos = max_num_videos

        self.root = root
        self.anno_path = anno_path
        self.samples: List[Dict[str, Any]] = []

        with open(self.anno_path) as file:
            self.samples = json.load(file)

        print(f"Video dataset loaded. {len(self.samples)} samples!")

        self.flag_tokens = self.text_tokenizer(
            "Mixed-modality generation (VIST video).", add_special_tokens=False
        ).input_ids
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)

        if len(self.system_tokens[0]) == 0:
            self.max_text_len = (
                max_seq_len
                - len(self.flag_tokens)
                - (num_image_tokens + 2) * max_num_videos
                - 2
            ) // max_num_videos
        else:
            self.max_text_len = (
                max_seq_len
                - (num_image_tokens + 2) * max_num_videos
                - 2
                - self.system_token_len
                - 1
            ) // max_num_videos

        if loader:
            self.loader = loader
        else:
            from decord import VideoReader
            from decord.bridge import set_bridge
            set_bridge("torch")
            self.loader = lambda x: VideoReader(x)

    def _get_interleaved_video_data(
        self, anno: Dict[str, Any]
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[List[int]]], List[str]]:
        video_paths = anno['videos'][:self.max_num_videos]
        texts = anno['captions'][:self.max_num_videos]

        video_list: List[Optional[torch.Tensor]] = []
        text_token_list: List[Optional[List[int]]] = []

        for path, text in zip(video_paths, texts):
            full_path = os.path.join(self.root, path)
            if not full_path.endswith(".mp4"):
                full_path += ".mp4"
            vr = self.loader(full_path)
            total_frames = len(vr)
            frame_indices = torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()
            video = torch.stack([vr[i] for i in frame_indices])  # shape: (T, H, W, C)
            video = video.permute(3, 0, 1, 2)  # → (C, T, H, W)
            video = self.transform(video, resolution=self.image_size)
            video_list.append(video)

            try:
                # tokens = self.text_tokenizer.tokenize(text)
                # print(f"🧩 Raw tokens: {tokens}")
                #
                # for t in tokens:
                #     tid = self.text_tokenizer.convert_tokens_to_ids(t)
                #     print(f"  '{t}' → {tid}")
                #     if tid < 0:
                #         print("⚠️  Invalid token:", t)

                # 위 token들로 다시 id 추출
                encoding = self.text_tokenizer(
                    text, add_special_tokens=True,
                    truncation=True, max_length=512,#self.max_text_len,
                )
                # print(f"input_ids type: {type(encoding.input_ids)}")
                # print(f"input_ids sample: {encoding.input_ids[:10]}")
                text_tokens = encoding.input_ids
                # print(f"✅ Final input_ids: {text_tokens}")

            except Exception as e:
                print(f"❌ Tokenization Error!\nText: '{text}'\nException: {e}")
                text_tokens = None  # 실패시 None 처리하거나 빈 리스트

            text_token_list.append(text_tokens)

        # Add flag token
        text_token_list[0] = self.flag_tokens + text_token_list[0]

        # Padding
        if len(video_list) != self.max_num_videos:
            video_list += [None] * (self.max_num_videos - len(video_list))
            text_token_list += [None] * (self.max_num_videos - len(text_token_list))
            texts += [''] * (self.max_num_videos - len(texts))

        return video_list, text_token_list, texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:

        try:
            anno = self.samples[idx]
            if len(anno['videos']) == 0:
                return self.__getitem__((idx + 1) % len(self))

            video_list, text_token_list, texts = self._get_interleaved_video_data(anno)

            (
                text_tokens,
                text_labels,
                modality_positions,
                text_mask,
                image_mask,
            ) = format_interleaved_sequence_video(
                video_frame_lists = video_list,
                text_token_list = text_token_list,
                bos_id = self.bos_id,
                eos_id = self.eos_id,
                bov_id = self.bov_id,
                eov_id = self.eov_id,
                pad_id = self.pad_id,
                vid_pad_id = self.vid_pad_id,
                num_image_tokens_per_frame = self.num_image_tokens,
                num_frames = -(-self.num_frames // 4), # 4 frame is sum up to 1 frame in prepare_latents_and_labels //train_video.py
                max_seq_len = self.max_seq_len,
                max_num_videos = self.max_num_videos,
            )

            # 강제로 long 타입 지정
            text_tokens = text_tokens.long()
            text_labels = text_labels.clone().long()
            modality_positions = modality_positions.long()
            text_mask = text_mask.long()
            image_mask = image_mask.long()

            text_labels[1: len(self.flag_tokens) + 1] = -100

            # 값 및 타입 출력 (디버깅용)
            # print(f"[DEBUG] After format_interleaved_sequence_video call, idx={idx}")
            # print(f"text_tokens dtype: {text_tokens.dtype}, min: {text_tokens.min()}, max: {text_tokens.max()}")
            # print(f"text_labels dtype: {text_labels.dtype}, min: {text_labels.min()}, max: {text_labels.max()}")
            # print(f"modality_positions dtype: {modality_positions.dtype}, values: {modality_positions}")
            # print(f"text_mask dtype: {text_mask.dtype}, unique: {text_mask.unique()}")
            # print(f"image_mask dtype: {image_mask.dtype}, unique: {image_mask.unique()}")

            # Pad empty videos
            temp: List[torch.Tensor] = []
            for v in video_list:
                if v is not None:
                    temp.append(v)
                else:
                    temp.append(torch.zeros((3, self.num_frames, self.image_size, self.image_size)))

            video_tensor = torch.stack(temp, dim=0)  # (N, C, T, H, W)


            return {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'videos': video_tensor,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': texts,
                'data_type': self.data_type,
            }

        except Exception as e:
            print(f"[WARN] skipping idx={idx} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))


    def collate_fn(self, batch: List[Optional[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data safely (ignores None)."""
        batch = [b for b in batch if b is not None]  # None 제거
        if not batch:
            return {}  # 빈 배치 방지
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)

        for key, value in batched.items():
            if key not in ('texts', 'data_type'):  # 리스트 유지
                try:
                    batched[key] = torch.stack(value, dim=0)
                except Exception as e:
                    print(f"[collate_fn] Failed to stack key={key} with error: {e}")
                    raise
            # 출력: batched 딕셔너리 키와 각 키별 리스트 길이, 값 샘플 확인
        # print("===== Batched content preview =====")
        # for key, val_list in batched.items():
        #     print(f"Key: {key}, Number of items: {len(val_list)}")
        #     # print(f" Sample items: {val_list[:1]}")  # 각 키별 앞 1개 값만 출력 (필요시 조절)
        # print("==================================")

        return batched

    '''def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key in batched:
            if key == 'texts':
                print("#####################\n", batched[key])
            # shape 정보 출력
            shapes = [v.shape for v in batched[key]] # texts: AttributeError: 'list' object has no attribute 'shape'
            print(f"Key: {key}, Shapes in batch: {shapes}")
            if len(set(shapes)) != 1:
                print(f"[ERROR] Shapes for key '{key}' are not all the same!")
            if key not in ("texts", "data_type"):
                batched[key] = torch.stack(batched[key], dim=0)
        return batched'''
