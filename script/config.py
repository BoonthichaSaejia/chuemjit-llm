model_name = "scb10x/llama-3-typhoon-v1.5-8b-instruct"

################################################################################
# QLoRA parameters (for base model)
# Minimum required: 1 x RTX 3090
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
#use_8bit = True
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "script/results"
save_dir = "script/model_lora"

# Number of training epochs
num_train_epochs = 110

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 16

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.12

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-5

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.01

# Optimizer to use
optim = "adamw_torch_fused"
#['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit', 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit', 'galore_adamw', 'galore_adamw_8bit', 'galore_adafactor', 'galore_adamw_layerwise', 'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise']

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = 40_000

# evaluation strategy
evaluation_strategy = "steps"

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.001

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 10
save_total_limit = 2

# Log every X updates steps
logging_steps = 5

# save strategy
save_strategy = "steps"

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 512

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
neftune_noise_alpha = 3
seed = 42

################################################################################
# SFT report
################################################################################
report_to = "wandb"
load_best_model_at_end = True
early_stopping_patience = 3