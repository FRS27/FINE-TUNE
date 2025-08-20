#!/usr/bin/env python3
# Qwen2.5-VL-3B-Instruct — LoRA SFT with Velocity Binning
# OPTIMIZED VERSION - Uses categorical velocity bins instead of raw floats

import os, json, torch, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import time
import numpy as np

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.cuda.empty_cache()
import gc
gc.collect()

# ============= VELOCITY BINNING FUNCTIONS =============
def create_velocity_bins():
    """Create detailed velocity bins for precise navigation"""
    
    # Linear velocity bins (m/s) - 10 levels
    LINEAR_BINS = {
        "stop": (0.0, 0.1),           # Stopped/minimal drift
        "crawl": (0.1, 0.5),          # Very slow movement
        "slow": (0.5, 0.8),           # Slow cautious movement
        "slow_normal": (0.8, 1.2),    # Below normal speed
        "normal": (1.2, 1.6),         # Standard navigation speed
        "normal_fast": (1.6, 1.9),    # Slightly above normal
        "fast": (1.9, 2.2),           # Fast movement
        "very_fast": (2.2, 2.5),      # Very fast
        "max": (2.5, 3.0),            # Maximum safe speed
        "emergency": (3.0, float('inf')) # Emergency/unusual
    }
    
    # Angular velocity bins (rad/s) - 15 levels for fine turning control
    # Negative = RIGHT turn, Positive = LEFT turn
    ANGULAR_BINS = {
        # RIGHT turns (negative values)
        "sharp_right": (-float('inf'), -0.8),   # Sharp right turn
        "hard_right": (-0.8, -0.6),             # Hard right
        "medium_right": (-0.6, -0.45),          # Medium right
        "right": (-0.45, -0.3),                 # Standard right
        "gentle_right": (-0.3, -0.2),           # Gentle right
        "slight_right": (-0.2, -0.1),           # Slight right
        "micro_right": (-0.1, -0.05),           # Micro adjustment right
        
        # STRAIGHT
        "straight": (-0.05, 0.05),              # Essentially straight
        
        # LEFT turns (positive values)
        "micro_left": (0.05, 0.1),              # Micro adjustment left
        "slight_left": (0.1, 0.2),              # Slight left
        "gentle_left": (0.2, 0.3),              # Gentle left
        "left": (0.3, 0.45),                    # Standard left
        "medium_left": (0.45, 0.6),             # Medium left
        "hard_left": (0.6, 0.8),                # Hard left
        "sharp_left": (0.8, float('inf')),      # Sharp left turn
    }
    
    return LINEAR_BINS, ANGULAR_BINS

def discretize_velocity_detailed(linear_vel, angular_vel):
    """Convert exact velocities to detailed bins"""
    LINEAR_BINS, ANGULAR_BINS = create_velocity_bins()
    
    # Find linear bin
    linear_bin = "normal"  # default
    for bin_name, (min_val, max_val) in LINEAR_BINS.items():
        if min_val <= linear_vel < max_val:
            linear_bin = bin_name
            break
    
    # Find angular bin  
    angular_bin = "straight"  # default
    for bin_name, (min_val, max_val) in ANGULAR_BINS.items():
        if min_val <= angular_vel < max_val:
            angular_bin = bin_name
            break
    
    return linear_bin, angular_bin

def validate_velocity_consistency(command, linear_bin, angular_bin):
    """Ensure velocity bins match the high-level command"""
    
    COMMAND_CONSTRAINTS = {
        "go_forward": {
            "linear": ["normal", "normal_fast", "fast", "slow_normal"],
            "angular": ["straight", "micro_left", "micro_right"]
        },
        "go_left": {
            "linear": ["slow", "slow_normal", "normal", "crawl"],
            "angular": ["left", "gentle_left", "medium_left", "slight_left", "hard_left"]
        },
        "go_right": {
            "linear": ["slow", "slow_normal", "normal", "crawl"],
            "angular": ["right", "gentle_right", "medium_right", "slight_right", "hard_right"]
        },
        "stop": {
            "linear": ["stop", "crawl"],
            "angular": ["straight", "micro_left", "micro_right"]
        }
    }
    
    constraints = COMMAND_CONSTRAINTS.get(command, None)
    if not constraints:
        return linear_bin, angular_bin
    
    # Adjust if inconsistent
    if linear_bin not in constraints["linear"]:
        linear_bin = constraints["linear"][0]  # Use first valid option
    
    if angular_bin not in constraints["angular"]:
        angular_bin = constraints["angular"][0]
    
    return linear_bin, angular_bin

# ============= CONFIGURATION =============
MODEL_ID          = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR        = "./qwen2.5vl3b_nav_lora_velocity_bins"  # New dir for binned version
SEED              = 42

# Image processing
MIN_PIXELS = 128 * 128
MAX_PIXELS = 160 * 160

# Sequence policy
MAX_SEQ_LEN_BASE  = 1024
LABEL_MARGIN      = 64
MAX_SEQ_LEN_CAP   = 1152

# LoRA configuration
LORA_RANK         = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.1

# Training parameters
NUM_EPOCHS        = 10
GRAD_ACCUM_STEPS  = 8
LEARNING_RATE     = 5e-6
WARMUP_RATIO      = 0.3
WEIGHT_DECAY      = 0.01
LABEL_SMOOTHING   = 0.02
MAX_STEPS         = 1000

# Save/eval cadence
LOG_STEPS         = 10
SAVE_STEPS        = 250
EVAL_STEPS        = 250
SAVE_TOTAL_LIMIT  = 8

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

class StepTimer(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        if not hasattr(self, "_t0"): return
        dt = time.time() - self._t0
        if state.global_step % 10 == 0:
            print(f"[timing] step={state.global_step} {dt:.3f}s")

class JSONValidationCallback(TrainerCallback):
    """Custom callback to check if model can generate valid JSON during training"""
    def __init__(self, processor, sample_images, sample_prompt):
        self.processor = processor
        self.sample_images = sample_images
        self.sample_prompt = sample_prompt
        self.last_check_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Check JSON generation every 500 steps
        if state.global_step > 0 and (state.global_step - self.last_check_step >= 500):
            self.last_check_step = state.global_step
            model = kwargs.get("model")
            if model is not None:
                self._test_json_generation(model, state.global_step)

    @torch.inference_mode()
    def _test_json_generation(self, model, step):
        try:
            model.eval()
            user_blocks = ([{"type": "image", "image": img} for img in self.sample_images] +
                          [{"type": "text", "text": self.sample_prompt}])
            messages = [{"role": "user", "content": user_blocks}]

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[prompt], images=self.sample_images, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

            response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check for velocity bins in response
            if "[Speed:" in response and "Turn:" in response:
                print(f"[JSON CHECK] Step {step}: ✓ Found velocity bins format")
                
            # Try to extract JSON
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    json_str = response[start:end+1]
                    try:
                        parsed = json.loads(json_str)
                        # Check if velocity bins are present
                        if "Velocity command" in parsed:
                            print(f"[JSON CHECK] Step {step}: ✓ Valid JSON with Velocity command")
                        return
                    except:
                        pass

            print(f"[JSON CHECK] Step {step}: ✗ Invalid format - continuing training")

        except Exception as e:
            print(f"[JSON CHECK] Step {step}: Error during validation: {e}")
        finally:
            model.train()

class NavJsonlWithBins(Dataset):
    """Dataset that converts float velocities to categorical bins on the fly"""
    
    def __init__(self, path: str, processor, use_bins: bool = True):
        with open(path, "r") as f:
            self.samples = []
            for line in f:
                sample = json.loads(line)
                if use_bins:
                    # Convert velocities to bins during loading
                    sample = self._convert_sample_to_bins(sample)
                self.samples.append(sample)
                
        self.processor = processor
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.pad_id = tok.pad_token_id

    def _convert_sample_to_bins(self, sample):
        """Convert a sample's velocities to use bins"""
        text = sample["text"]
        if "ASSISTANT:" not in text:
            return sample
            
        user_part, assistant_part = text.split("ASSISTANT:", 1)
        
        try:
            # Parse the original JSON
            gt_json = json.loads(assistant_part.strip())
            
            # Get velocities and command
            vel_cmd = gt_json.get("Velocity command", {})
            if not vel_cmd:
                return sample
                
            linear_vel = float(vel_cmd.get("linear_velocity", 0))
            angular_vel = float(vel_cmd.get("angular_velocity", 0))
            command = gt_json.get("Next high-level command", "go_forward")
            
            # Discretize velocities
            linear_bin, angular_bin = discretize_velocity_detailed(linear_vel, angular_vel)
            
            # Validate consistency with command
            linear_bin, angular_bin = validate_velocity_consistency(command, linear_bin, angular_bin)
            
            # Create modified JSON with bins
            modified_json = gt_json.copy()
            modified_json["Velocity command"] = {
                "linear_velocity": linear_bin,
                "angular_velocity": angular_bin
            }
            
            # Add velocity hint before JSON (helps model understand the pattern)
            velocity_hint = f"[Speed: {linear_bin}, Turn: {angular_bin}] "
            
            # Reconstruct sample
            new_text = f"{user_part}ASSISTANT: {velocity_hint}{json.dumps(modified_json)}"
            
            return {
                "images": sample["images"],
                "text": new_text
            }
            
        except Exception as e:
            print(f"Warning: Could not convert sample to bins: {e}")
            return sample

    def __len__(self): 
        return len(self.samples)

    def _conv(self, image_paths, user_text, assistant_text):
        user_blocks = ([{"type": "image", "image": p} for p in image_paths]
                       + [{"type": "text", "text": user_text}])
        assistant_blocks = [{"type": "text", "text": assistant_text}]
        prompt_only = [{"role": "user", "content": user_blocks}]
        full_conv   = prompt_only + [{"role": "assistant", "content": assistant_blocks}]
        return prompt_only, full_conv

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.samples[i]
        txt = s["text"]
        if "ASSISTANT:" not in txt:
            raise ValueError("Sample missing 'ASSISTANT:' separator.")
        user_txt, assistant_txt = txt.split("ASSISTANT:", 1)
        user_txt = user_txt.strip()
        assistant_txt = assistant_txt.strip()

        # Build chat blocks
        prompt_only, full_conv = self._conv(s["images"], user_txt, assistant_txt)

        # Render chat to strings
        prompt_text = self.processor.apply_chat_template(
            prompt_only, add_generation_prompt=True, tokenize=False
        )
        full_text = self.processor.apply_chat_template(
            full_conv, add_generation_prompt=False, tokenize=False
        )

        # Load images
        imgs = [Image.open(p).convert("RGB") for p in s["images"]]

        # Tokenize text + process images
        inputs_full = self.processor(
            text=[full_text],
            images=[imgs],
            return_tensors="pt"
        )

        # Verify images are present
        pv = inputs_full.get("pixel_values", None)
        assert pv is not None and pv.numel() > 0, "pixel_values missing/empty"
        
        # Tokenize prompt-only for label masking
        tok_prompt = self.processor.tokenizer(
            [prompt_text], return_tensors="pt", add_special_tokens=False
        )
        prompt_len = tok_prompt["input_ids"].shape[-1]

        # Apply sequence length policy
        input_ids = inputs_full["input_ids"].squeeze(0)
        attn_mask = inputs_full["attention_mask"].squeeze(0)

        L = input_ids.shape[0]
        need_len = min(L, max(prompt_len + LABEL_MARGIN, MAX_SEQ_LEN_BASE))
        seq_len  = min(need_len, MAX_SEQ_LEN_CAP)

        if L > seq_len:
            input_ids = input_ids[:seq_len]
            attn_mask = attn_mask[:seq_len]
        elif L < seq_len:
            pad_n = seq_len - L
            pad_ids = torch.full((pad_n,), self.pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attn_mask = torch.cat([attn_mask, torch.zeros(pad_n, dtype=attn_mask.dtype)], dim=0)

        # Create labels with prompt masking
        labels = input_ids.clone()
        labels[:min(prompt_len, seq_len)] = -100
        if (labels != -100).sum().item() == 0:
            cut = max(0, seq_len - LABEL_MARGIN)
            labels[:cut] = -100

        # Build batch
        batch = {k: v for k, v in inputs_full.items()}
        batch["input_ids"] = input_ids.unsqueeze(0)
        batch["attention_mask"] = attn_mask.unsqueeze(0)
        batch["labels"] = labels.unsqueeze(0)
        return batch

@dataclass
class PassThroughCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError("per_device_train_batch_size must be 1.")
        return features[0]

def build_target_module_names(model) -> List[str]:
    return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def load_model_and_processor() -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Set attention implementation
    try:
        model.config.attn_implementation = "flash_attention_2"
        print("[info] Using Flash Attention 2")
    except Exception:
        try:
            model.config.attn_implementation = "sdpa"
            print("[info] Using SDPA attention")
        except Exception:
            print("[info] Using default attention")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Find and unfreeze projector
    projector = None
    projector_names = ["multi_modal_projector", "mm_projector", "visual_projection", "vision_projector"]

    for name in projector_names:
        if hasattr(model, name):
            projector = getattr(model, name)
            print(f"[info] Found projector: {name}")
            break

    if projector is None:
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ["visual", "vision", "image", "merger", "aligner"]):
                print(f"[debug] Found vision-related module: {name}")
        if "proj" in name.lower() or "merge" in name.lower():
            for p in module.parameters():
                p.requires_grad_(True)
            projector_found = True
            print(f"[info] Unfroze vision module: {name}")

    if projector is not None:
        for p in projector.parameters():
            p.requires_grad_(True)
        projector.to(torch.bfloat16 if USE_BF16 else torch.float16)
        print(f"[info] Unfroze projector with {sum(p.numel() for p in projector.parameters())} parameters")
    else:
        print("[warning] Could not find vision projector")

    # Freeze base model parameters
    for n, p in model.named_parameters():
        is_projector = any(proj_name in n.lower() for proj_name in ["projector", "projection"])
        if not is_projector:
            p.requires_grad_(False)

    # LoRA configuration
    target_modules = build_target_module_names(model)
    modules_to_save = []
    for name, _ in model.named_modules():
        if any(proj_name in name.lower() for proj_name in ["projector", "projection"]):
            top_level = name.split('.')[0] if '.' in name else name
            if top_level not in modules_to_save:
                modules_to_save.append(top_level)

    if not modules_to_save:
        modules_to_save = ["multi_modal_projector", "mm_projector", "visual_projection"]

    print(f"[info] LoRA modules_to_save: {modules_to_save}")

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, processor

def _make_training_args(eval_enabled: bool) -> TrainingArguments:
    common_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_steps=EVAL_STEPS if eval_enabled else None,
        load_best_model_at_end=eval_enabled,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        label_smoothing_factor=LABEL_SMOOTHING,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        max_steps=MAX_STEPS,
        remove_unused_columns=False,
    )

    try:
        return TrainingArguments(
            evaluation_strategy="steps" if eval_enabled else "no",
            **common_kwargs,
        )
    except TypeError as e:
        if "evaluation_strategy" in str(e):
            return TrainingArguments(
                eval_strategy="steps" if eval_enabled else "no",
                **common_kwargs,
            )
        raise

def train(train_jsonl: str, val_jsonl: Optional[str], resume: Optional[str], use_bins: bool = True):
    set_seed(SEED)

    model, processor = load_model_and_processor()

    # Load datasets with velocity binning
    train_ds = NavJsonlWithBins(train_jsonl, processor, use_bins=use_bins)
    eval_ds = NavJsonlWithBins(val_jsonl, processor, use_bins=use_bins) if val_jsonl else None

    # Prepare sample for JSON validation callback
    raw_sample_data = train_ds.samples[0]
    sample_images = [Image.open(p).convert("RGB") for p in raw_sample_data["images"]]
    sample_prompt = "Analyze the scene and provide navigation commands."

    theoretical_updates_per_epoch = max(1, len(train_ds) // GRAD_ACCUM_STEPS)
    print(f"[info] train_samples={len(train_ds)}, grad_accum={GRAD_ACCUM_STEPS}, "
          f"epochs={NUM_EPOCHS}, updates/epoch~={theoretical_updates_per_epoch}, "
          f"total_updates_est~={theoretical_updates_per_epoch*NUM_EPOCHS}, max_steps={MAX_STEPS}")
    print(f"[info] Using velocity binning: {use_bins}")

    args = _make_training_args(eval_enabled=eval_ds is not None)
    collator = PassThroughCollator()

    callbacks = [
        StepTimer(),
        JSONValidationCallback(processor, sample_images, sample_prompt)
    ]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    
    if resume:
        print(f"[info] Resuming training from checkpoint: {resume}")
    else:
        print(f"[info] Starting training from scratch - will run full {MAX_STEPS} steps")
    
    trainer.train(resume_from_checkpoint=resume)

def preprocess_dataset_file(input_jsonl: str, output_jsonl: str):
    """Preprocess an entire JSONL file to use velocity bins"""
    print(f"Preprocessing {input_jsonl} -> {output_jsonl}")
    
    processed = 0
    errors = 0
    
    with open(input_jsonl, 'r') as fin, open(output_jsonl, 'w') as fout:
        for line_num, line in enumerate(fin):
            try:
                sample = json.loads(line)
                
                # Convert to bins
                text = sample["text"]
                if "ASSISTANT:" not in text:
                    errors += 1
                    continue
                    
                user_part, assistant_part = text.split("ASSISTANT:", 1)
                gt_json = json.loads(assistant_part.strip())
                
                # Get velocities and command
                vel_cmd = gt_json.get("Velocity command", {})
                linear_vel = float(vel_cmd.get("linear_velocity", 0))
                angular_vel = float(vel_cmd.get("angular_velocity", 0))
                command = gt_json.get("Next high-level command", "go_forward")
                
                # Discretize velocities
                linear_bin, angular_bin = discretize_velocity_detailed(linear_vel, angular_vel)
                
                # Validate consistency with command
                linear_bin, angular_bin = validate_velocity_consistency(command, linear_bin, angular_bin)
                
                # Create modified JSON with bins
                modified_json = gt_json.copy()
                modified_json["Velocity command"] = {
                    "linear_velocity": linear_bin,
                    "angular_velocity": angular_bin
                }
                
                # Add velocity hint before JSON
                velocity_hint = f"[Speed: {linear_bin}, Turn: {angular_bin}] "
                
                # Reconstruct sample
                new_sample = {
                    "images": sample["images"],
                    "text": f"{user_part}ASSISTANT: {velocity_hint}{json.dumps(modified_json)}"
                }
                
                fout.write(json.dumps(new_sample) + "\n")
                processed += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                errors += 1
    
    print(f"Preprocessing complete: {processed} samples processed, {errors} errors")

def main(train_jsonl: str, val_jsonl: str = None, resume: str = None,
         predict_jsonl: str = None, predict_n: int = 0, preprocess_only: bool = False):

    if preprocess_only:
        # Just preprocess the datasets and exit
        train_binned = train_jsonl.replace(".jsonl", "_binned.jsonl")
        preprocess_dataset_file(train_jsonl, train_binned)
        
        if val_jsonl:
            val_binned = val_jsonl.replace(".jsonl", "_binned.jsonl")
            preprocess_dataset_file(val_jsonl, val_binned)
        
        print("Preprocessing complete. Use the _binned.jsonl files for training.")
        return

    # Training
    if train_jsonl:
        if resume is None and os.path.isdir(OUTPUT_DIR):
            ckpts = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if ckpts:
                resume = str(ckpts[-1])
        
        # Use binning by default
        train(train_jsonl, val_jsonl, resume, use_bins=True)

    # Prediction (handled in separate script)
    if predict_jsonl and predict_n > 0:
        print("Use the separate prediction script for inference with velocity bin conversion.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True, help="path to train.jsonl")
    ap.add_argument("--val_jsonl", default=None)
    ap.add_argument("--resume", default=None, help="path to checkpoint dir to resume from")
    ap.add_argument("--preprocess_only", action="store_true", 
                    help="Only preprocess datasets to use velocity bins, don't train")
    args = ap.parse_args()

    main(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        resume=args.resume,
        preprocess_only=args.preprocess_only
    )