
# coding=utf-8
"""
LoRA-enabled training script for Showo-2 mixed modality model.
This file is adapted from train_mixed_modality_simple.py with minimal
changes to enable parameter-efficient fine-tuning (LoRA) of the Qwen2.5
LLM backbone inside Showo-2.

Usage
-----
$ python train_mixed_modality_lora.py \
    --config_path path/to/showo2_1.5b_downstream_mixed_modality_simple.yaml

Make sure to install `peft` beforehand:
$ pip install -U peft
"""

import os, sys, logging, math, json, shutil, time, random
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.optim import AdamW
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ------------------- LoRA imports -------------------
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError as e:
    raise ImportError("peft is required: pip install -U peft") from e
# ----------------------------------------------------

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type

from datasets import MixedDataLoader, VISTDataset
from utils import get_config, flatten_omega_conf, AverageMeter, denorm, denorm_vid, get_hyper_params, \
    path_to_llm_name, _freeze_params
from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# -----------------------------------------------------------------------------
# Helper: wrap Showo-2's LLM with LoRA
# -----------------------------------------------------------------------------

def _apply_lora_to_showo(model: Showo2Qwen2_5, r: int = 64, alpha: int = 32, dropout: float = 0.1,
                          target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Attach a LoRA adapter to the Showo2Qwen2_5.showo (Qwen2.5) sub-module.
    Returns the patched model with showo wrapped by PEFT.
    """
    logger.info("Applying LoRA to Qwen2.5 sub-module …")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        inference_mode=False,
    )
    # If you plan to train in 8-bit / 4-bit, uncomment next line
    # model.showo = prepare_model_for_kbit_training(model.showo)
    model.showo = get_peft_model(model.showo, lora_cfg)
    logger.info(model.showo.print_trainable_parameters())
    return model

# -----------------------------------------------------------------------------
# Main training entrance (largely identical to original, but LoRA-aware)
# -----------------------------------------------------------------------------

def main():
    # -------------------- Accelerator & config --------------------
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    bs_mixed_modal = config.training.batch_size_mixed_modal
    total_batch_size_per_gpu = bs_mixed_modal * config.dataset.accumulation
    total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
    total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu

    # ------------- logging -------------
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    (set_verbosity_info if accelerator.is_local_main_process else set_verbosity_error)()

    # wandb
    if accelerator.is_main_process:
        import wandb
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None) or wandb.util.generate_id()
        config.wandb.run_id = run_id
        accelerator.init_trackers(
            config.experiment.project,
            config={k: v for k, v in flatten_omega_conf(config, resolve=True)},
            init_kwargs={"wandb": dict(name=config.experiment.name, id=run_id, resume=resume_wandb_run)},
        )
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        OmegaConf = __import__("omegaconf").omegaconf.OmegaConf
        OmegaConf.save(config, Path(config.experiment.output_dir) / "config_lora.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # ------------------ Models ------------------
    weight_type = get_weight_type(config)

    # VAE
    if config.model.vae_model.type != "wan21":
        raise NotImplementedError("Only WanVAE is supported right now.")
    from models import WanVAE
    vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=accelerator.device)

    # Tokenizer & Showo-2
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False)
    else:
        model = Showo2Qwen2_5(**config.model.showo)
    model = model.to(accelerator.device)

    # ---- Apply LoRA here ----
    model = _apply_lora_to_showo(model)

    # Freeze all except LoRA
    _freeze_params(model, config.model.showo.frozen_params)

    # Update preprocessing numbers if using time embeddings
    if config.model.showo.add_time_embeds:
        pp = config.dataset.preprocessing
        pp.num_mmu_image_tokens += 1
        pp.num_t2i_image_tokens += 1
        pp.num_hq_image_tokens += 1
        pp.num_video_tokens += 1
        pp.num_mixed_modal_tokens += 1

    # -------------- Optimizer (LoRA params only) --------------
    optimizer_cfg = config.optimizer.params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=optimizer_cfg.learning_rate, betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
                      weight_decay=optimizer_cfg.weight_decay, eps=optimizer_cfg.epsilon)

    # -------------- Datasets & loaders --------------
    pp_cfg = config.dataset.preprocessing
    ds_cfg = config.dataset.params

    def _make_loader(ds, bs):
        sampler = DistributedSampler(ds, num_replicas=accelerator.num_processes, rank=accelerator.process_index,
                                     shuffle=True, drop_last=True) if accelerator.num_processes > 2 else None
        return DataLoader(ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                          num_workers=ds_cfg.num_workers, drop_last=True, collate_fn=ds.collate_fn)

    dataset = VISTDataset(ds_cfg.train_mixed_modal_shards_path_or_url, anno_path=ds_cfg.annotation_path,
                          text_tokenizer=text_tokenizer, image_size=pp_cfg.mixed_modal_resolution,
                          max_seq_len=pp_cfg.max_mixed_modal_seq_length, num_image_tokens=pp_cfg.num_mixed_modal_tokens,
                          latent_width=pp_cfg.mixed_modal_latent_width, latent_height=pp_cfg.mixed_modal_latent_height,
                          cond_dropout_prob=config.training.cond_dropout_prob, min_res=pp_cfg.min_res,
                          showo_token_ids=showo_token_ids, system=("", "", ""),
                          max_num_images=pp_cfg.max_num_images)

    train_loader = _make_loader(dataset, config.training.batch_size_mixed_modal)
    mixed_loader = MixedDataLoader([train_loader], samp_probs=config.dataset.samp_probs,
                                   accumulation=config.dataset.accumulation,
                                   mode=config.dataset.mixed_loader_mode)

    # ---------- Scheduler ----------
    lr_scheduler = get_scheduler(config.lr_scheduler.scheduler, optimizer=optimizer,
                                 num_training_steps=config.training.max_train_steps,
                                 num_warmup_steps=config.lr_scheduler.params.warmup_steps,
                                 power=config.lr_scheduler.params.power)

    # ---------- Accelerator prep ----------
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ---------- Training loop (shortened for brevity) ----------
    logger.info("***** LoRA Training *****")
    global_step = 0
    sampler = Sampler(create_transport(path_type=config.transport.path_type, prediction=config.transport.prediction,
                                       loss_weight=config.transport.loss_weight, train_eps=config.transport.train_eps,
                                       sample_eps=config.transport.sample_eps, snr_type=config.transport.snr_type,
                                       do_shift=config.transport.do_shift, seq_len=pp_cfg.num_t2i_image_tokens))

    loss_meter = AverageMeter(); batch_meter = AverageMeter(); end = time.time()

    for step, batch in enumerate(mixed_loader):
        if global_step >= config.training.max_train_steps:
            break
        model.train()
        # ---- forward ---- (reuse original logic via model.forward) ----
        text_tokens = batch['text_tokens'].to(accelerator.device)
        text_labels = batch['text_labels'].to(accelerator.device)
        pixel_values = batch['images'].to(accelerator.device).to(weight_type)
        image_latents = pixel_values.unsqueeze(2) if len(pixel_values.shape) == 4 else pixel_values
        bsz = text_tokens.size(0)
        t = torch.rand(bsz, device=accelerator.device)
        mod_pos = batch['modality_positions'].to(accelerator.device)
        mask = omni_attn_mask_naive(bsz, text_tokens.size(1), mod_pos, accelerator.device).to(weight_type)

        logits, loss_ntp = model(text_tokens=text_tokens, image_latents=image_latents, t=t.to(weight_type),
                                 attention_mask=mask, text_labels=text_labels, modality_positions=mod_pos,
                                 output_hidden_states=False, max_seq_len=text_tokens.size(1), device=accelerator.device)
        loss = loss_ntp
        accelerator.backward(loss / config.training.gradient_accumulation_steps)

        if (step + 1) % config.training.gradient_accumulation_steps == 0:
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
            global_step += 1
            batch_meter.update(time.time() - end); end = time.time()
            if accelerator.is_main_process and global_step % config.experiment.log_every == 0:
                logger.info(f"step {global_step} loss {loss.item():.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training finished. Saving LoRA adapter …")
        model.showo.save_pretrained(Path(config.experiment.output_dir) / "lora_adapter")


if __name__ == "__main__":
    main()
