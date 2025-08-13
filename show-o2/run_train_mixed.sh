# run_train.sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

accelerate launch \
  --config_file ../accelerate_configs/8_gpus_deepspeed_zero2.yaml \
  --main_process_port=9999 \
  train_mixed_modality_simple.py config=configs/showo2_1.5b_downstream_mixed_modality_simple.yaml