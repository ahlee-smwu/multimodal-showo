import copy
import math
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode


def image_transform(image, resolution=256, normalize=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                    y0_centercrop=False):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    if y0_centercrop:
        width, height = image.size
        left = (width - resolution) / 2
        top = 0
        right = (width + resolution) / 2
        bottom = resolution
        image = image.crop((left, top, right, bottom))
    else:
        image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=mean, std=std, inplace=True)(image)
    return image


def to_tensor_and_normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=mean, std=std, inplace=True)(image)
    return image


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '').replace("This image showcases ", '').replace("This suggests ", '').replace(
        "In the photo, we see ", '').replace("This is ", '').replace("This image is ", '').replace(
        "In the photo, we have ", '').replace("The photo features ", '').replace("The photo depicts ", '').replace(
        "The photo appears to be ", '')

    return caption


# At this time, we do not model the text in image-text pairs for t2i
def format_sequence_gen_qwen2_5(text_tokens, system_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, system_token_len):
    if system_token_len == 0:
        modality_positions = torch.tensor([[len(text_tokens) + 1 + 1, num_image_tokens]])
        # text_labels = [bos_id] + [-100] * len(text_tokens) + [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        # text_labels = [bos_id] + text_tokens + [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        text_labels = [-100] + [-100] * len(text_tokens) + [-100] + [-100] * num_image_tokens + [-100] + [-100]
        text_tokens = [bos_id] + text_tokens + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]
    else:
        # TODO TO BE VERIFIED
        modality_positions = torch.tensor([[1 + system_token_len + len(text_tokens) + 1 + 1, num_image_tokens]])
        text_labels = [bos_id] + [-100] * len(system_tokens[0] + system_tokens[1] + text_tokens) + [eos_id] + \
                      [-100] * len(system_tokens[2]) + \
                      [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        text_tokens = [bos_id] + system_tokens[0] + system_tokens[1] + text_tokens + [eos_id] + system_tokens[2] + \
                      [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask

def format_sequence_und(text_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                        num_image_tokens, max_seq_len):
    modality_positions = torch.tensor([[1 + 1, num_image_tokens]])

    text_labels = [bos_id] + [boi_id] + [-100] * num_image_tokens + [eoi_id] + \
                  text_tokens + [eos_id]

    text_tokens = [bos_id] + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + \
                  text_tokens + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def format_interleaved_sequence(image_list, text_token_list, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, max_num_images, system_tokens=None, system_token_len=0):
    """
    # generation
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    # eg. 0        1-9           10          11-15        16         17
    # understanding
    # [bos_id, im_start, image_tokens, im_end, text_tokens, eos_id, pad_id]
    # eg. 0        1            2-6           7           8-16       17
    """

    text_tokens = []
    text_labels = []
    modality_positions = []

    cur_len = 1 + system_token_len # bos token
    for txt_token, image in zip(text_token_list, image_list):
        if txt_token is not None:
            text_tokens.extend(txt_token)
            text_labels.extend(copy.deepcopy(txt_token))
            cur_len += len(txt_token)

        if image is not None:
            text_tokens.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
            text_labels.extend([boi_id] + [img_pad_id] * num_image_tokens + [eoi_id])
            # +1 for one <|img_start|> token
            modality_positions.append((cur_len + 1, num_image_tokens))
            cur_len = cur_len + 1 + num_image_tokens + 1  # +2 to include <|img_start|> and <|img_end|>

    if system_token_len == 0:
        text_labels = [bos_id] + text_labels + [eos_id]
        text_tokens = [bos_id] + text_tokens + [eos_id]
    else:
        # TODO TO BE VERIFIED
        text_labels = [bos_id] + [-100] * system_token_len + text_labels + [eos_id]
        text_tokens = [bos_id] + system_tokens[0] + system_tokens[1] + system_tokens[2] + text_tokens + [eos_id]

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    if len(modality_positions) < max_num_images:
        modality_positions += [(0, 0) for _ in range(max_num_images - len(modality_positions))]

    modality_positions = torch.tensor(modality_positions)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def resize_crop(image, image_height, image_width):
    aspect_ratio = image_width / image_height
    if isinstance(image, torch.Tensor) and image.ndim == 4:
        frame_height, frame_width = image[0].size(1), image[0].size(2)
        original_size_as_tuple = torch.tensor([frame_height, frame_width])
        image_aspect_ratio = frame_width / frame_height
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (frame_width / frame_height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (frame_height / frame_width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                           interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                         image_width)
    else:
        frame_height, frame_width = image.size(1), image.size(2)
        image_aspect_ratio = frame_width / frame_height
        original_size_as_tuple = torch.tensor([frame_height, frame_width])
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (frame_width / frame_height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (frame_height / frame_width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                           interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                         image_width)
    crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
    return image, original_size_as_tuple, crop_coords_top_left


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

from torchvision import transforms
def video_transform(video_tensor, resolution=256):
    # video_tensor: (C, T, H, W)
    frames = []
    for i in range(video_tensor.shape[1]):
        frame = video_tensor[:, i, :, :]  # (C, H, W)
        # frame은 tensor 이므로 ToPILImage 후 Resize -> ToTensor 후 Normalize 등
        frame_pil = transforms.ToPILImage()(frame)
        frame_resized = transforms.Resize(resolution)(frame_pil)
        frame_tensor = transforms.ToTensor()(frame_resized)
        frames.append(frame_tensor)
    # (T, C, H, W) -> (C, T, H, W)
    video_transformed = torch.stack(frames, dim=1)
    return video_transformed

import torch
import copy

import torch
import copy

def format_interleaved_sequence_video(
    video_frame_lists, text_token_list,
    bos_id, eos_id, bov_id, eov_id, pad_id, vid_pad_id,
    num_image_tokens_per_frame, num_frames,
    max_seq_len, max_num_videos,
    system_tokens=None, system_token_len=0
):
    """
    Format interleaved text & video sequence for LLM

    Args:
        video_frame_lists: List[List[Tensor or None]]  # 각 video는 frame list
        text_token_list:   List[List[int] or None]
        num_image_tokens_per_frame: int, 각 frame 당 이미지 토큰 수
        num_frames: int, 각 비디오에서 사용되는 프레임 개수
        max_seq_len: int
        max_num_videos: int (e.g. 5 등등)
        system_tokens: Optional[List[List[int]]], 3개 리스트가 필요 (optional)
        system_token_len: int
    Returns:
        text_tokens: [max_seq_len] LongTensor
        text_labels: [max_seq_len] LongTensor
        modality_positions: [max_num_videos, 2] LongTensor (start, length)
        text_mask: [max_seq_len] LongTensor
        image_mask: [max_seq_len] LongTensor
    """
    assert len(video_frame_lists) == len(text_token_list), \
        "video_frame_lists and text_token_list length mismatch!"

    text_tokens = []
    text_labels = []
    modality_positions = []

    # system_tokens 실제 삽입은 나중에 하므로 임시 누적 길이 잡음
    cur_len = 1 + system_token_len  # <bos> (+system)

    for idx in range(len(text_token_list)):
        txt_token = text_token_list[idx]
        video_frames = video_frame_lists[idx]

        if txt_token is not None:
            text_tokens.extend(txt_token)
            text_labels.extend(txt_token)
            cur_len += len(txt_token)

        if video_frames is not None:
            # 총 image 토큰 개수: num_image_tokens_per_frame * num_frames
            total_img_tokens = num_image_tokens_per_frame * num_frames
            # image token block: <|img_start|> + [img_pad_id] * (프레임수*프레임당토큰수) + <|img_end|>
            text_tokens.extend([bov_id] + [vid_pad_id] * total_img_tokens + [eov_id])
            text_labels.extend([bov_id] + [vid_pad_id] * total_img_tokens + [eov_id])
            # modality_positions: (idx, (시작, length))
            # 이미지 토큰들은 <|img_start|> 바로 다음부터 시작, cur_len+1
            modality_positions.append((cur_len+1, total_img_tokens))    # start point, length
            cur_len += 1 + total_img_tokens + 1  # <img_start>, tokens, <img_end>

    # 시스템 토큰 처리
    if system_token_len == 0:
        text_tokens = [bos_id] + text_tokens + [eos_id]
        text_labels = [bos_id] + text_labels + [eos_id]
    else:
        assert (system_tokens is not None) and (sum(len(x) for x in system_tokens) == system_token_len), \
            f"system_token_len={system_token_len} but got {sum(len(x) for x in system_tokens)}"
        text_tokens = [bos_id] \
            + list(system_tokens[0]) + list(system_tokens[1]) + list(system_tokens[2]) \
            + text_tokens + [eos_id]
        text_labels = [bos_id] + [-100] * system_token_len + text_labels + [eos_id]

    # 패딩
    text_len = len(text_tokens)
    if text_len > max_seq_len:
        raise ValueError(f"Generated sequence ({text_len}) > max_seq_len ({max_seq_len})")
    pad_len = max_seq_len - text_len
    text_tokens += [pad_id] * pad_len
    text_labels += [-100] * pad_len

    text_tokens = torch.tensor(text_tokens, dtype=torch.long)
    text_labels = torch.tensor(text_labels, dtype=torch.long)

    # modality_positions 패딩 (길이 맞추기, 미채워진 곳은 (0, 0))
    if len(modality_positions) < max_num_videos:
        modality_positions += [(0, 0)] * (max_num_videos - len(modality_positions))
    modality_positions = torch.tensor(modality_positions, dtype=torch.long)

    # 마스크
    text_mask = ((text_tokens != pad_id) & (text_tokens != vid_pad_id)).long()
    image_mask = (text_tokens == vid_pad_id).long()

    return text_tokens, text_labels, modality_positions, text_mask, image_mask