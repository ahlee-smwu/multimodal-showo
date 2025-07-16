import os
import copy
import json
import collections
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets.utils import resize_and_pad_image, to_tensor_and_normalize

IGNORE_INDEX = -100

class MMUVideoDataset(Dataset):
    def __init__(
        self,
        root: str,
        annotation_path: str,
        text_tokenizer,
        showo_token_ids,
        image_size=384,
        num_frames=8,
        max_seq_len=1024,
        source_max_len=512,
        target_max_len=512,
        cond_dropout_prob=0.1,
        stage='pre-training',
        default_system_prompt="system\nYou are a helpful assistant.<|im_end|>",
    ):
        self.root = root
        self.num_frames = num_frames
        self.image_size = image_size
        self.samples = json.load(open(annotation_path, 'r'))
        self.text_tokenizer = text_tokenizer
        self.showo_token_ids = showo_token_ids
        self.stage = stage
        self.default_system_prompt = default_system_prompt
        self.cond_dropout_prob = cond_dropout_prob

        self.max_seq_len = max_seq_len
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.num_image_tokens = 576  # 필요에 따라 조절 가능

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def __len__(self):
        return len(self.samples)

    def load_video(self, video_id):
        frame_dir = os.path.join(self.root, video_id)
        frame_names = sorted(os.listdir(frame_dir))[:self.num_frames]
        frames = []
        for fname in frame_names:
            img_path = os.path.join(frame_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = resize_and_pad_image(img, (self.image_size, self.image_size))
            img = to_tensor_and_normalize(img, mean=self.mean, std=self.std)
            frames.append(img)
        # (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        return video_tensor

    def format_multi_sequence_und_qwen2_5(
        self,
        sources,
        targets,
        ignore_question=True
    ):
        bos_id = self.showo_token_ids['bos_id']
        eos_id = self.showo_token_ids['eos_id']
        boi_id = self.showo_token_ids['boi_id']
        eoi_id = self.showo_token_ids['eoi_id']
        img_pad_id = self.showo_token_ids['img_pad_id']
        img_id = self.showo_token_ids['img_id']

        text_tokens = []
        text_labels = []
        modality_positions = []

        if not self.stage.startswith('pre-training'):
            default_system_prompt = self.text_tokenizer(
                self.default_system_prompt,
                max_length=100,
                truncation=True,
                add_special_tokens=False,
            )['input_ids']
            role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
            role_b = self.text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

        cur_len = 1  # bos 토큰 자리

        for src, tgt in zip(sources, targets):
            if not self.stage.startswith('pre-training'):
                src = role_a + src + [eos_id] + role_b
            if cur_len == 1 and not self.stage.startswith('pre-training'):
                src = default_system_prompt + src

            if img_id in src:
                idx = src.index(img_id)
                src = src[:idx] + [boi_id] + [img_pad_id] * self.num_image_tokens + [eoi_id] + src[idx + 1:]
                modality_positions.append((cur_len + idx + 1, self.num_image_tokens))

            text_tokens.extend(src + tgt)
            if ignore_question:
                text_labels.extend([IGNORE_INDEX] * len(src) + tgt)
            else:
                text_labels.extend(src + tgt)
            cur_len = len(text_tokens) + 1

        text_tokens = [bos_id] + text_tokens
        text_labels = [IGNORE_INDEX] + text_labels

        pad_len = self.max_seq_len - len(text_tokens)
        if pad_len > 0:
            text_tokens.extend([self.text_tokenizer.pad_token_id] * pad_len)
            text_labels.extend([IGNORE_INDEX] * pad_len)
        else:
            text_tokens = text_tokens[:self.max_seq_len]
            text_labels = text_labels[:self.max_seq_len]

        if len(modality_positions) == 0:
            modality_positions = [(0, 0)]

        modality_positions = torch.tensor(modality_positions)
        text_tokens = torch.tensor(text_tokens)
        text_labels = torch.tensor(text_labels)

        text_mask = torch.where(
            (text_tokens != img_pad_id) & (text_tokens != self.text_tokenizer.pad_token_id),
            torch.ones_like(text_tokens),
            torch.zeros_like(text_tokens),
        )
        image_mask = torch.where(text_tokens == img_pad_id,
                                 torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            video_tensor = self.load_video(sample['video_id'])  # (C, T, H, W)

            conversation = []
            for conv in sample['conversations']:
                role = "user" if conv['from'] == 'human' else "assistant"
                text = conv['value']
                if role == "user" and text.endswith('\n<video>'):
                    text = '<video>\n' + text[:-len('\n<video>')]
                conversation.append({"role": role, "content": text})

            sources = [c['content'] for c in conversation if c['role'] == "user"]
            targets = [c['content'] + self.text_tokenizer.eos_token for c in conversation if c['role'] == "assistant"]

            sources = [
                self.text_tokenizer(s, max_length=self.source_max_len, truncation=True, add_special_tokens=False).input_ids
                for s in sources
            ]
            targets = [
                self.text_tokenizer(t, max_length=self.target_max_len, truncation=True, add_special_tokens=False).input_ids
                for t in targets
            ]

            text_tokens, text_labels, modality_positions, text_mask, image_mask = self.format_multi_sequence_und_qwen2_5(
                sources, targets
            )

            ret = {
                "text_tokens": text_tokens,
                "text_labels": text_labels,
                "videos": video_tensor,  # (C, T, H, W)
                "modality_positions": modality_positions,
                "text_masks": text_mask,
                "image_masks": image_mask,
                "data_type": "mmu_vid",
                "texts": self.text_tokenizer.batch_decode(text_tokens),
            }

            return ret

        except Exception as e:
            print(f"Error at idx={idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type'):
                batched[k] = torch.stack(v, dim=0)
        return batched
