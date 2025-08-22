#!/usr/bin/env python3
"""
Local GP Training Script for RTX 3070
Loads pretrained Qwen model and trains only GP components locally
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    set_seed,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gpytorch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle
import argparse
import time
from datetime import datetime
import gc

warnings.filterwarnings('ignore')

# ============= RTX 3070 OPTIMIZED CONFIGURATION =============
MODEL_PATH = "/home/afaris/final_project/final_model"  # Your downloaded model
TRAIN_JSONL = "/home/afaris/final_project/train_dataset6_final_relative.jsonl"
VAL_JSONL = "/home/afaris/final_project/validation_dataset6_final_relative.jsonl"
OUTPUT_DIR = "./local_gp_results"

# RTX 3070 8GB optimized settings
RTX3070_CONFIG = {
    'max_batch_size': 1,           # Conservative for 8GB
    'max_seq_length': 512,         # Shorter sequences
    'use_cpu_offload': True,       # Offload when possible
    'use_fp16': True,              # FP16 for memory efficiency
    'tiny_image_size': 224,        # Reasonable image size
    'gp_batch_size': 32,           # GP training batch size
    'bridge_hidden_dim': 256,      # Smaller bridge network
    'gp_input_dim': 64,            # Smaller GP input
}

# GP Training Configuration
GP_TRAINING_SAMPLES = 300      # Reduced for faster training
GP_KERNEL_TYPE = 'matern'
QWEN_FEATURE_DIM = 2048       # Qwen's feature dimension
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

print(f"🚀 Local GP Training for RTX 3070")
print(f"📁 Model path: {MODEL_PATH}")
print(f"🖥️  Device: {DEVICE}")
print(f"⚙️  RTX 3070 Config: {RTX3070_CONFIG}")

# ============= MEMORY MANAGEMENT =============
def clear_gpu_memory():
    """Aggressive GPU memory clearing for RTX 3070"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Multiple garbage collections
        for _ in range(3):
            gc.collect()

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated(0) / 1e9,
            'reserved': torch.cuda.memory_reserved(0) / 1e9,
            'free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
        }
    return {'allocated': 0, 'reserved': 0, 'free': 0}

# ============= GP COMPONENTS (Same as original) =============
class ExactGPModel(gpytorch.models.ExactGP):
    """GP model for velocity prediction"""
    def __init__(self, train_x, train_y, likelihood, kernel_type='matern'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        elif kernel_type == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPVelocityPredictor(nn.Module):
    """GP-based velocity predictor optimized for RTX 3070"""
    def __init__(self, input_dim=None, hidden_dim=None, use_full_gp=True):
        super().__init__()
        self.input_dim = input_dim or RTX3070_CONFIG['gp_input_dim']
        hidden_dim = hidden_dim or RTX3070_CONFIG['bridge_hidden_dim']
        self.use_full_gp = use_full_gp
        
        if use_full_gp:
            self.likelihood_linear = gpytorch.likelihoods.GaussianLikelihood()
            self.likelihood_angular = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_linear = None
            self.gp_angular = None
            
            # Feature extractor for GP
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.input_dim // 2)
            )
        else:
            # Neural network fallback
            self.nn_predictor = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 2)
            )

    def extract_features(self, x):
        if self.use_full_gp:
            return self.feature_extractor(x)
        return x

    def fit_gp(self, train_features, train_velocities):
        if not self.use_full_gp:
            print("⚠️ Not using full GP mode")
            return
            
        print(f"🎯 Fitting GP with {len(train_features)} samples...")
        
        # Move to CPU for GP training (more stable)
        train_features_cpu = train_features.cpu()
        train_velocities_cpu = train_velocities.cpu()
        
        with torch.no_grad():
            gp_features = self.extract_features(train_features_cpu)
        
        # Create GP models
        self.gp_linear = ExactGPModel(
            gp_features,
            train_velocities_cpu[:, 0],
            self.likelihood_linear,
            kernel_type=GP_KERNEL_TYPE
        )
        self.gp_angular = ExactGPModel(
            gp_features,
            train_velocities_cpu[:, 1], 
            self.likelihood_angular,
            kernel_type=GP_KERNEL_TYPE
        )
        
        # Training mode
        self.gp_linear.train()
        self.gp_angular.train()
        self.likelihood_linear.train()
        self.likelihood_angular.train()
        
        # Optimize hyperparameters
        print("🔧 Optimizing GP hyperparameters...")
        optimizer_linear = torch.optim.Adam(self.gp_linear.parameters(), lr=0.1)
        optimizer_angular = torch.optim.Adam(self.gp_angular.parameters(), lr=0.1)
        mll_linear = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_linear, self.gp_linear)
        mll_angular = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_angular, self.gp_angular)
        
        # Training loop
        for i in range(30):  # Reduced iterations for faster training
            # Linear velocity GP
            optimizer_linear.zero_grad()
            output_linear = self.gp_linear(gp_features)
            loss_linear = -mll_linear(output_linear, train_velocities_cpu[:, 0])
            loss_linear.backward()
            optimizer_linear.step()
            
            # Angular velocity GP
            optimizer_angular.zero_grad()
            output_angular = self.gp_angular(gp_features)
            loss_angular = -mll_angular(output_angular, train_velocities_cpu[:, 1])
            loss_angular.backward()
            optimizer_angular.step()
            
            if i % 10 == 0:
                print(f"  Iteration {i}: Linear loss = {loss_linear.item():.4f}, Angular loss = {loss_angular.item():.4f}")
        
        print("✅ GP hyperparameter optimization complete!")

    def forward(self, x):
        if self.use_full_gp and self.gp_linear is not None:
            features = self.extract_features(x.cpu())  # GP inference on CPU
            self.gp_linear.eval()
            self.gp_angular.eval()
            self.likelihood_linear.eval()
            self.likelihood_angular.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                linear_pred = self.likelihood_linear(self.gp_linear(features))
                angular_pred = self.likelihood_angular(self.gp_angular(features))
                linear_mean = linear_pred.mean
                angular_mean = angular_pred.mean
                velocities = torch.stack([linear_mean, angular_mean], dim=1)
                return velocities.to(x.device)  # Move back to original device
        else:
            return self.nn_predictor(x)

# ============= BRIDGE NETWORK (RTX 3070 Optimized) =============
class QwenToGPBridge(nn.Module):
    """Lightweight bridge network for RTX 3070"""
    def __init__(self, qwen_dim=QWEN_FEATURE_DIM, gp_dim=None, hidden_dim=None):
        super().__init__()
        gp_dim = gp_dim or RTX3070_CONFIG['gp_input_dim']
        hidden_dim = hidden_dim or RTX3070_CONFIG['bridge_hidden_dim']
        
        self.feature_projector = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, gp_dim),
            nn.Tanh()
        )
        
        # Lightweight attention
        self.attention = nn.MultiheadAttention(
            embed_dim=qwen_dim,
            num_heads=4,  # Reduced from 8
            dropout=0.1,
            batch_first=True
        )
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, qwen_features, return_attention=False):
        if qwen_features.dim() == 2:
            qwen_features = qwen_features.unsqueeze(1)
        
        attended_features, attention_weights = self.attention(
            qwen_features, qwen_features, qwen_features
        )
        pooled_features = attended_features.mean(dim=1)
        gp_features = self.feature_projector(pooled_features)
        
        if return_attention:
            return gp_features, attention_weights
        return gp_features

# ============= FIXED HYBRID MODEL =============
class LocalHybridQwenGP(nn.Module):
    """Local hybrid model optimized for RTX 3070"""
    def __init__(self, qwen_model, processor):
        super().__init__()
        self.qwen_model = qwen_model
        self.processor = processor
        self.bridge = QwenToGPBridge()
        self.gp_predictor = GPVelocityPredictor(use_full_gp=True)
        
        # Move components to GPU
        if torch.cuda.is_available():
            self.bridge = self.bridge.cuda()
            self.gp_predictor = self.gp_predictor.cuda()

    def extract_features_direct(self, **inputs):
        """Direct feature extraction (fixed from your test script)"""
        with torch.no_grad():
            # Ensure inputs are on the right device
            device_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    device_inputs[k] = v.to(self.qwen_model.device)
                else:
                    device_inputs[k] = v
            
            # Get outputs with hidden states
            outputs = self.qwen_model(**device_inputs, return_dict=True, output_hidden_states=True)
            
            # Direct feature extraction
            extracted_features = None
            
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
                if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
                    extracted_features = hidden_states[-1]  # Last layer
                else:
                    extracted_features = hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                extracted_features = outputs.last_hidden_state
            
            if extracted_features is not None:
                # Pool over sequence dimension
                if extracted_features.dim() == 3:  # [batch, seq, hidden]
                    extracted_features = extracted_features.mean(dim=1)  # [batch, hidden]
                return extracted_features
            else:
                # Fallback
                batch_size = device_inputs['input_ids'].shape[0]
                return torch.randn(batch_size, QWEN_FEATURE_DIM, device=self.qwen_model.device)

    def predict_velocities(self, **inputs):
        """Predict velocities using the hybrid pipeline"""
        try:
            # Extract features
            qwen_features = self.extract_features_direct(**inputs)
            
            if qwen_features is not None:
                # Bridge to GP features
                gp_features = self.bridge(qwen_features)
                # Predict velocities
                velocities = self.gp_predictor(gp_features)
                return velocities
            else:
                batch_size = inputs['input_ids'].shape[0]
                return torch.zeros(batch_size, 2, device=self.qwen_model.device)
                
        except Exception as e:
            print(f"⚠️ Velocity prediction failed: {e}")
            batch_size = inputs['input_ids'].shape[0]
            return torch.zeros(batch_size, 2, device=self.qwen_model.device)

    def generate_with_velocities(self, **inputs):
        """Generate text and predict velocities"""
        with torch.no_grad():
            # Generate text
            generated_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1,
            )
            
            # Predict velocities
            velocities = self.predict_velocities(**inputs)
            
            return generated_ids, velocities

    def save_components(self, save_directory):
        """Save hybrid components"""
        os.makedirs(save_directory, exist_ok=True)
        torch.save({
            'bridge_state_dict': self.bridge.state_dict(),
            'gp_predictor_state_dict': self.gp_predictor.state_dict(),
            'config': {
                'rtx3070_config': RTX3070_CONFIG,
                'gp_input_dim': self.gp_predictor.input_dim,
                'use_full_gp': self.gp_predictor.use_full_gp
            }
        }, os.path.join(save_directory, 'local_hybrid_components.pt'))
        print(f"✅ Hybrid components saved to {save_directory}")

# ============= DATASET (RTX 3070 Optimized) =============
class LocalNavigationDataset(Dataset):
    """RTX 3070 optimized dataset"""
    def __init__(self, jsonl_path: str, processor, velocity_scaler=None, max_samples=None):
        self.processor = processor
        self.velocity_scaler = velocity_scaler
        self.samples = []
        
        print(f"📚 Loading dataset from {jsonl_path}")
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))
        
        # Set padding token
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.pad_id = tok.pad_token_id
        
        print(f"✅ Loaded {len(self.samples)} samples")

    def extract_velocity_from_text(self, text: str) -> np.ndarray:
        """Extract velocity from JSON text"""
        try:
            if "ASSISTANT:" not in text:
                return np.array([0.0, 0.0])
            _, assistant_part = text.split("ASSISTANT:", 1)
            assistant_json = json.loads(assistant_part.strip())
            vel_cmd = assistant_json.get("Velocity command", {})
            linear_vel = float(vel_cmd.get("linear_velocity", 0.0))
            angular_vel = float(vel_cmd.get("angular_velocity", 0.0))
            return np.array([linear_vel, angular_vel])
        except Exception:
            return np.array([0.0, 0.0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        text = sample["text"]
        
        if "ASSISTANT:" not in text:
            raise ValueError(f"Sample {idx} missing 'ASSISTANT:' separator")
        
        user_text, assistant_text = text.split("ASSISTANT:", 1)
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        
        # Extract velocity target
        velocity_target = self.extract_velocity_from_text(text)
        
        # Prepare user content
        user_content = []
        for img_path in sample["images"]:
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize for RTX 3070
                img = img.resize((RTX3070_CONFIG['tiny_image_size'], RTX3070_CONFIG['tiny_image_size']))
                user_content.append({"type": "image", "image": img})
            except Exception as e:
                print(f"⚠️ Could not load image {img_path}: {e}")
                continue
        
        user_content.append({"type": "text", "text": user_text})
        
        messages = [{"role": "user", "content": user_content}]
        
        # Process with shorter sequences
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Get images for processing
        images = [content["image"] for content in user_content if content["type"] == "image"]
        
        inputs = self.processor(
            text=[prompt_text],
            images=[images] if images else None,
            return_tensors="pt",
            max_length=RTX3070_CONFIG['max_seq_length'],
            truncation=True,
            padding=True
        )
        
        # Scale velocity if scaler provided
        if self.velocity_scaler is not None:
            velocity_target = self.velocity_scaler.transform(velocity_target.reshape(1, -1))[0]
        
        # Create batch
        batch = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "velocity_targets": torch.tensor(velocity_target, dtype=torch.float32)
        }
        
        # Add other input keys
        for key in inputs:
            if key not in ["input_ids", "attention_mask"]:
                batch[key] = inputs[key]
        
        return batch

# ============= MAIN FUNCTIONS =============
def load_pretrained_model():
    """Load the pretrained model from your A6000 training"""
    print(f"🔄 Loading pretrained model from {MODEL_PATH}")
    
    mem_before = get_gpu_memory()
    print(f"GPU memory before loading: {mem_before['allocated']:.1f}GB")
    
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Processor loaded")
        
        # Load model with RTX 3070 optimization
        qwen_model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if RTX3070_CONFIG['use_fp16'] else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        qwen_model.eval()  # Set to eval mode
        print("✅ Qwen model loaded")
        
        # Create hybrid model
        hybrid_model = LocalHybridQwenGP(qwen_model, processor)
        print("✅ Hybrid model created")
        
        mem_after = get_gpu_memory()
        print(f"GPU memory after loading: {mem_after['allocated']:.1f}GB")
        
        return hybrid_model, processor
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        clear_gpu_memory()
        raise

def prepare_velocity_scaler(train_jsonl: str) -> StandardScaler:
    """Prepare velocity scaler from training data"""
    print("📊 Preparing velocity scaler...")
    velocities = []
    
    with open(train_jsonl, 'r') as f:
        for line in f:
            sample = json.loads(line)
            text = sample["text"]
            if "ASSISTANT:" in text:
                try:
                    _, assistant_part = text.split("ASSISTANT:", 1)
                    assistant_json = json.loads(assistant_part.strip())
                    vel_cmd = assistant_json.get("Velocity command", {})
                    linear_vel = float(vel_cmd.get("linear_velocity", 0.0))
                    angular_vel = float(vel_cmd.get("angular_velocity", 0.0))
                    velocities.append([linear_vel, angular_vel])
                except:
                    continue
    
    velocities = np.array(velocities)
    scaler = StandardScaler()
    scaler.fit(velocities)
    
    print(f"✅ Velocity scaler fitted on {len(velocities)} samples")
    print(f"Linear velocity: mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}")
    print(f"Angular velocity: mean={scaler.mean_[1]:.3f}, std={scaler.scale_[1]:.3f}")
    
    return scaler

def train_gp_components(hybrid_model, train_dataset, velocity_scaler):
    """Train GP components locally"""
    print("\n🎯 Training GP components on RTX 3070...")
    
    train_features = []
    train_velocities = []
    
    hybrid_model.eval()
    clear_gpu_memory()
    
    print(f"🔄 Extracting features from {min(GP_TRAINING_SAMPLES, len(train_dataset))} samples...")
    
    with torch.no_grad():
        for i in range(min(GP_TRAINING_SAMPLES, len(train_dataset))):
            try:
                batch = train_dataset[i]
                
                # Move to device
                device_batch = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        device_batch[key] = value.to(DEVICE)
                    else:
                        device_batch[key] = value
                
                # Extract features
                qwen_features = hybrid_model.extract_features_direct(**device_batch)
                
                if qwen_features is not None:
                    # Convert through bridge
                    gp_features = hybrid_model.bridge(qwen_features)
                    train_features.append(gp_features.cpu())
                    
                    # Get velocity target
                    vel_target = device_batch['velocity_targets'].cpu()
                    if velocity_scaler:
                        # Inverse transform to original scale
                        vel_target = torch.tensor(
                            velocity_scaler.inverse_transform(vel_target.numpy().reshape(1, -1))[0]
                        )
                    train_velocities.append(vel_target)
                
                if (i + 1) % 50 == 0:
                    print(f"  ✅ Processed {i+1}/{min(GP_TRAINING_SAMPLES, len(train_dataset))} samples")
                    mem_info = get_gpu_memory()
                    print(f"  GPU memory: {mem_info['allocated']:.1f}GB")
                    
                    # Clear memory periodically
                    if (i + 1) % 100 == 0:
                        clear_gpu_memory()
                        
            except Exception as e:
                print(f"  ⚠️ Sample {i} failed: {e}")
                continue
    
    if len(train_features) > 10:  # Need at least some samples
        print(f"\n✅ Successfully extracted {len(train_features)} feature samples!")
        
        # Stack tensors
        train_features_tensor = torch.cat(train_features, dim=0)
        train_velocities_tensor = torch.stack(train_velocities, dim=0)
        
        print(f"📊 Features shape: {train_features_tensor.shape}")
        print(f"📊 Velocities shape: {train_velocities_tensor.shape}")
        print(f"📊 Features mean: {train_features_tensor.mean().item():.6f}")
        print(f"📊 Features std: {train_features_tensor.std().item():.6f}")
        
        # Train GP models
        print("\n🚀 Training GP models...")
        hybrid_model.gp_predictor.fit_gp(train_features_tensor, train_velocities_tensor)
        
        return True
    else:
        print(f"❌ Only extracted {len(train_features)} samples - insufficient for GP training")
        return False

# ============= EVALUATION (Same as original) =============
class ComprehensiveEvaluator:
    """Comprehensive evaluation system"""
    def __init__(self):
        print("🔄 Loading sentence transformer for evaluation...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def evaluate_text_similarity(self, pred_text: str, gt_text: str) -> float:
        """Calculate semantic similarity"""
        try:
            pred_embedding = self.sentence_model.encode([pred_text])
            gt_embedding = self.sentence_model.encode([gt_text])
            similarity = cosine_similarity(pred_embedding, gt_embedding)[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def evaluate_json_fields(self, pred_json: dict, gt_json: dict) -> dict:
        """Evaluate individual JSON fields"""
        results = {}
        
        # Scene description similarity
        if "Scene description" in pred_json and "Scene description" in gt_json:
            results["scene_similarity"] = self.evaluate_text_similarity(
                pred_json["Scene description"], 
                gt_json["Scene description"]
            )
        else:
            results["scene_similarity"] = 0.0
        
        # Command accuracy
        pred_cmd = pred_json.get("Next high-level command", "")
        gt_cmd = gt_json.get("Next high-level command", "")
        results["command_accuracy"] = 1.0 if pred_cmd == gt_cmd else 0.0
        
        # Explanation similarity
        if "Explanation" in pred_json and "Explanation" in gt_json:
            results["explanation_similarity"] = self.evaluate_text_similarity(
                pred_json["Explanation"], 
                gt_json["Explanation"]
            )
        else:
            results["explanation_similarity"] = 0.0
        
        return results
    
    def evaluate_velocities(self, pred_vel: np.ndarray, gt_vel: np.ndarray) -> dict:
        """Evaluate velocity predictions"""
        linear_error = abs(pred_vel[0] - gt_vel[0])
        angular_error = abs(pred_vel[1] - gt_vel[1])
        total_error = np.sqrt(linear_error**2 + angular_error**2)
        
        return {
            "linear_error": linear_error,
            "angular_error": angular_error,
            "total_error": total_error,
            "linear_relative_error": linear_error / (abs(gt_vel[0]) + 1e-6),
            "angular_relative_error": angular_error / (abs(gt_vel[1]) + 1e-6)
        }

def predict_with_hybrid_local(hybrid_model, processor, velocity_scaler, images: List[str], prompt: str = None):
    """Local inference with the hybrid model"""
    if prompt is None:
        prompt = """You are an expert robot navigation assistant. The user will provide a sequence of 6 images, where the final image represents the present moment (Frame N-1).
Your task is to analyze this sequence and perform the following:

Describe the Present (Frame N-1): Based on the final image in the sequence, provide a description of the robot's immediate surroundings.
Predict the Future (Frame N): Based on the entire sequence, predict the robot's action for the immediate next moment (Frame N).
Your final output MUST be a single, valid JSON object using these exact keys:

"Scene description"
"Next high-level command" (must be one of ["go_forward", "go_left", "go_right", "stop"])
"Explanation"
"Velocity command"
The value for "Velocity command" must be a nested JSON object with "linear_velocity" and "angular_velocity" keys."""
    
    try:
        # Load and resize images for RTX 3070
        pil_images = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((RTX3070_CONFIG['tiny_image_size'], RTX3070_CONFIG['tiny_image_size']))
            pil_images.append(img)
        
        # Prepare messages
        user_content = []
        for img in pil_images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]
        
        # Process input
        prompt_text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        inputs = processor(
            text=[prompt_text],
            images=[pil_images],
            return_tensors="pt",
            max_length=RTX3070_CONFIG['max_seq_length'],
            truncation=True
        )
        
        # Move to device
        device_inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Generate text and predict velocities
        with torch.no_grad():
            generated_ids, predicted_velocities = hybrid_model.generate_with_velocities(**device_inputs)
            
            # Decode generated text
            generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
            if "ASSISTANT:" in generated_text:
                assistant_response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                assistant_response = generated_text
            
            # Unscale velocities
            if velocity_scaler:
                predicted_velocities_unscaled = velocity_scaler.inverse_transform(
                    predicted_velocities.cpu().numpy()
                )[0]
            else:
                predicted_velocities_unscaled = predicted_velocities.cpu().numpy()[0]
        
        # Parse JSON response
        try:
            json_start = assistant_response.find('{')
            json_end = assistant_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = assistant_response[json_start:json_end]
                response_json = json.loads(json_str)
                
                # Override velocity command with GP prediction
                response_json["Velocity command"] = {
                    "linear_velocity": float(predicted_velocities_unscaled[0]),
                    "angular_velocity": float(predicted_velocities_unscaled[1])
                }
                
                return {
                    "success": True,
                    "text_response": assistant_response,
                    "json_output": response_json,
                    "velocities": predicted_velocities_unscaled,
                    "formatted_json": json.dumps(response_json, indent=2)
                }
        except Exception as e:
            print(f"⚠️ JSON parsing failed: {e}")
        
        return {
            "success": False,
            "text_response": assistant_response,
            "json_output": None,
            "velocities": predicted_velocities_unscaled,
            "formatted_json": None
        }
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return {
            "success": False,
            "text_response": "",
            "json_output": None,
            "velocities": np.array([0.0, 0.0]),
            "formatted_json": None
        }

def comprehensive_evaluation_local(hybrid_model, processor, velocity_scaler, eval_jsonl: str):
    """Comprehensive evaluation optimized for RTX 3070"""
    print("\n" + "="*60)
    print("LOCAL HYBRID MODEL EVALUATION (RTX 3070)")
    print("="*60)

    evaluator = ComprehensiveEvaluator()
    eval_dataset = LocalNavigationDataset(eval_jsonl, processor, velocity_scaler)

    results = {
        'samples': [],
        'velocity_metrics': {
            'linear_errors': [],
            'angular_errors': [],
            'total_errors': [],
            'linear_relative_errors': [],
            'angular_relative_errors': []
        },
        'text_metrics': {
            'scene_similarities': [],
            'command_accuracies': [],
            'explanation_similarities': [],
            'json_success_rate': [],
            'valid_command_rate': []
        },
        'overall_scores': {}
    }

    valid_commands = ["go_forward", "go_left", "go_right", "stop"]
    hybrid_model.eval()

    all_pred_velocities = []
    all_gt_velocities = []

    # Evaluate subset for RTX 3070 (to avoid memory issues)
    max_eval_samples = min(100, len(eval_dataset))  # Limit for RTX 3070
    print(f"🔄 Evaluating {max_eval_samples} samples...")

    for i in range(max_eval_samples):
        try:
            sample = eval_dataset.samples[i]
            sample_result = {
                'sample_id': i,
                'images': sample["images"],
                'timestamp': datetime.now().isoformat()
            }

            # Extract ground truth
            text = sample["text"]
            gt_velocities = eval_dataset.extract_velocity_from_text(text)
            all_gt_velocities.append(gt_velocities)

            # Parse ground truth JSON
            try:
                _, gt_assistant_text = text.split("ASSISTANT:", 1)
                gt_json = json.loads(gt_assistant_text.strip())
                sample_result['ground_truth'] = gt_json
            except:
                sample_result['ground_truth'] = None
                continue

            # Get user prompt
            user_text = text.split("ASSISTANT:")[0].strip()

            # Run inference
            prediction_result = predict_with_hybrid_local(
                hybrid_model, processor, velocity_scaler,
                sample["images"], user_text
            )

            sample_result['prediction'] = {
                'success': prediction_result["success"],
                'text_response': prediction_result["text_response"],
                'json_output': prediction_result["json_output"],
                'velocities': prediction_result["velocities"].tolist()
            }

            # Velocity evaluation
            pred_velocities = prediction_result["velocities"]
            all_pred_velocities.append(pred_velocities)

            velocity_eval = evaluator.evaluate_velocities(pred_velocities, gt_velocities)
            sample_result['velocity_evaluation'] = velocity_eval

            # Store velocity metrics
            results['velocity_metrics']['linear_errors'].append(velocity_eval['linear_error'])
            results['velocity_metrics']['angular_errors'].append(velocity_eval['angular_error'])
            results['velocity_metrics']['total_errors'].append(velocity_eval['total_error'])
            results['velocity_metrics']['linear_relative_errors'].append(velocity_eval['linear_relative_error'])
            results['velocity_metrics']['angular_relative_errors'].append(velocity_eval['angular_relative_error'])

            # Text evaluation
            if prediction_result["success"] and prediction_result["json_output"] and sample_result['ground_truth']:
                text_eval = evaluator.evaluate_json_fields(
                    prediction_result["json_output"],
                    sample_result['ground_truth']
                )
                sample_result['text_evaluation'] = text_eval

                # Store text metrics
                results['text_metrics']['scene_similarities'].append(text_eval['scene_similarity'])
                results['text_metrics']['command_accuracies'].append(text_eval['command_accuracy'])
                results['text_metrics']['explanation_similarities'].append(text_eval['explanation_similarity'])
                results['text_metrics']['json_success_rate'].append(1)

                # Check valid command
                pred_cmd = prediction_result["json_output"].get("Next high-level command", "")
                results['text_metrics']['valid_command_rate'].append(1 if pred_cmd in valid_commands else 0)
            else:
                sample_result['text_evaluation'] = {
                    'scene_similarity': 0.0,
                    'command_accuracy': 0.0,
                    'explanation_similarity': 0.0
                }
                results['text_metrics']['scene_similarities'].append(0.0)
                results['text_metrics']['command_accuracies'].append(0.0)
                results['text_metrics']['explanation_similarities'].append(0.0)
                results['text_metrics']['json_success_rate'].append(0)
                results['text_metrics']['valid_command_rate'].append(0)

            results['samples'].append(sample_result)

            # Progress and memory management
            if (i + 1) % 10 == 0:
                print(f"  ✅ Evaluated {i+1}/{max_eval_samples} samples")
                mem_info = get_gpu_memory()
                print(f"  GPU memory: {mem_info['allocated']:.1f}GB")
                clear_gpu_memory()
                
        except Exception as e:
            print(f"  ⚠️ Sample {i} evaluation failed: {e}")
            continue

    # Calculate overall metrics
    print("\n📊 Calculating overall metrics...")

    # Velocity metrics
    velocity_metrics = results['velocity_metrics']
    results['overall_scores']['velocity'] = {
        'mean_linear_error': np.mean(velocity_metrics['linear_errors']),
        'std_linear_error': np.std(velocity_metrics['linear_errors']),
        'mean_angular_error': np.mean(velocity_metrics['angular_errors']),
        'std_angular_error': np.std(velocity_metrics['angular_errors']),
        'mean_total_error': np.mean(velocity_metrics['total_errors']),
        'std_total_error': np.std(velocity_metrics['total_errors']),
        'mean_linear_relative_error': np.mean(velocity_metrics['linear_relative_errors']),
        'mean_angular_relative_error': np.mean(velocity_metrics['angular_relative_errors']),
        'rmse': np.sqrt(np.mean(np.array(velocity_metrics['total_errors'])**2)),
        'mae': np.mean(velocity_metrics['total_errors']),
        'median_error': np.median(velocity_metrics['total_errors']),
        'percentile_95_error': np.percentile(velocity_metrics['total_errors'], 95)
    }

    # R² scores
    if len(all_gt_velocities) > 1:
        all_gt = np.array(all_gt_velocities)
        all_pred = np.array(all_pred_velocities)
        velocity_r2 = r2_score(all_gt, all_pred)
        linear_r2 = r2_score(all_gt[:, 0], all_pred[:, 0])
        angular_r2 = r2_score(all_gt[:, 1], all_pred[:, 1])

        linear_corr = np.corrcoef(all_gt[:, 0], all_pred[:, 0])[0, 1]
        angular_corr = np.corrcoef(all_gt[:, 1], all_pred[:, 1])[0, 1]

        results['overall_scores']['velocity'].update({
            'r2_score': velocity_r2,
            'linear_r2': linear_r2,
            'angular_r2': angular_r2,
            'linear_correlation': linear_corr,
            'angular_correlation': angular_corr,
            'velocity_mse': mean_squared_error(all_gt, all_pred),
            'velocity_mae': mean_absolute_error(all_gt, all_pred)
        })

    # Text metrics
    text_metrics = results['text_metrics']
    results['overall_scores']['text'] = {
        'mean_scene_similarity': np.mean(text_metrics['scene_similarities']),
        'std_scene_similarity': np.std(text_metrics['scene_similarities']),
        'mean_command_accuracy': np.mean(text_metrics['command_accuracies']),
        'mean_explanation_similarity': np.mean(text_metrics['explanation_similarities']),
        'std_explanation_similarity': np.std(text_metrics['explanation_similarities']),
        'json_success_rate': np.mean(text_metrics['json_success_rate']),
        'valid_command_rate': np.mean(text_metrics['valid_command_rate']),
        'overall_text_score': (np.mean(text_metrics['scene_similarities']) +
                               np.mean(text_metrics['command_accuracies']) +
                               np.mean(text_metrics['explanation_similarities'])) / 3
    }

    # Hybrid score
    velocity_score = max(0, results['overall_scores']['velocity'].get('r2_score', 0))
    text_score = results['overall_scores']['text']['overall_text_score']
    hybrid_accuracy = (0.6 * velocity_score + 0.4 * text_score) * 100
    results['overall_scores']['hybrid_accuracy'] = min(100, hybrid_accuracy)

    # Add metadata
    results['evaluation_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': max_eval_samples,
        'total_available_samples': len(eval_dataset),
        'hardware': 'RTX 3070 8GB',
        'model_config': RTX3070_CONFIG,
        'gp_training_samples': GP_TRAINING_SAMPLES
    }

    return results

def print_evaluation_summary_local(results: dict):
    """Print evaluation summary for local results"""
    print("\n" + "="*60)
    print("LOCAL EVALUATION SUMMARY (RTX 3070)")
    print("="*60)
    
    # Velocity performance
    vel_scores = results['overall_scores']['velocity']
    print(f"\n🎯 VELOCITY PREDICTION PERFORMANCE:")
    print(f"  R² Score (Overall): {vel_scores.get('r2_score', 0):.4f}")
    print(f"  Linear Velocity R²: {vel_scores.get('linear_r2', 0):.4f}")
    print(f"  Angular Velocity R²: {vel_scores.get('angular_r2', 0):.4f}")
    print(f"  Mean Linear Error: {vel_scores['mean_linear_error']:.4f} ± {vel_scores.get('std_linear_error', 0):.4f} m/s")
    print(f"  Mean Angular Error: {vel_scores['mean_angular_error']:.4f} ± {vel_scores.get('std_angular_error', 0):.4f} rad/s")
    print(f"  RMSE: {vel_scores['rmse']:.4f}")
    print(f"  MAE: {vel_scores.get('mae', 0):.4f}")
    
    # Text performance
    text_scores = results['overall_scores']['text']
    print(f"\n📝 TEXT GENERATION PERFORMANCE:")
    print(f"  Scene Description Similarity: {text_scores['mean_scene_similarity']:.3f}")
    print(f"  Command Accuracy: {text_scores['mean_command_accuracy']:.3f}")
    print(f"  Explanation Similarity: {text_scores['mean_explanation_similarity']:.3f}")
    print(f"  JSON Success Rate: {text_scores['json_success_rate']:.3f}")
    print(f"  Valid Command Rate: {text_scores['valid_command_rate']:.3f}")
    
    # Overall performance
    hybrid_accuracy = results['overall_scores']['hybrid_accuracy']
    print(f"\n🏆 LOCAL HYBRID MODEL ACCURACY: {hybrid_accuracy:.2f}%")
    
    # Target comparison
    target_r2 = 0.99
    current_r2 = vel_scores.get('r2_score', 0)
    achievement = (current_r2 / target_r2) * 100 if target_r2 > 0 else 0
    print(f"\n📈 PERFORMANCE vs GP TARGET:")
    print(f"  Target GP R²: {target_r2:.4f}")
    print(f"  Current Hybrid R²: {current_r2:.4f}")
    print(f"  Achievement: {achievement:.1f}%")
    
    metadata = results.get('evaluation_metadata', {})
    print(f"\n⚙️  CONFIGURATION:")
    print(f"  Hardware: {metadata.get('hardware', 'Unknown')}")
    print(f"  Evaluated Samples: {metadata.get('num_samples', 0)}")
    print(f"  GP Training Samples: {metadata.get('gp_training_samples', 0)}")
    
    print("="*60)

# ============= MAIN EXECUTION =============
def main():
    """Main execution function"""
    print("🚀 Starting Local GP Training Pipeline")
    set_seed(SEED)
    
    # Check prerequisites
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model path not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(TRAIN_JSONL):
        print(f"❌ Training data not found: {TRAIN_JSONL}")
        return
        
    if not os.path.exists(VAL_JSONL):
        print(f"❌ Validation data not found: {VAL_JSONL}")
        return
    
    try:
        # Step 1: Load pretrained model
        print("\n🔄 STEP 1: Loading pretrained model...")
        hybrid_model, processor = load_pretrained_model()
        
        # Step 2: Prepare data
        print("\n📊 STEP 2: Preparing velocity scaler...")
        velocity_scaler = prepare_velocity_scaler(TRAIN_JSONL)
        
        print("\n📚 Loading training dataset...")
        train_dataset = LocalNavigationDataset(TRAIN_JSONL, processor, velocity_scaler)
        
        # Step 3: Train GP components
        print("\n🎯 STEP 3: Training GP components...")
        gp_success = train_gp_components(hybrid_model, train_dataset, velocity_scaler)
        
        if not gp_success:
            print("❌ GP training failed - cannot proceed")
            return
        
        # Step 4: Save trained components
        print("\n💾 STEP 4: Saving trained components...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        hybrid_model.save_components(OUTPUT_DIR)
        
        # Save velocity scaler
        with open(os.path.join(OUTPUT_DIR, "velocity_scaler.pkl"), 'wb') as f:
            pickle.dump(velocity_scaler, f)
        
        # Step 5: Comprehensive evaluation
        print("\n📊 STEP 5: Comprehensive evaluation...")
        evaluation_results = comprehensive_evaluation_local(hybrid_model, processor, velocity_scaler, VAL_JSONL)
        
        # Step 6: Print and save results
        print_evaluation_summary_local(evaluation_results)
        
        # Save results
        results_path = os.path.join(OUTPUT_DIR, "local_evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"\n✅ LOCAL GP TRAINING COMPLETE!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"🎯 Hybrid Accuracy: {evaluation_results['overall_scores']['hybrid_accuracy']:.2f}%")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local GP Training for RTX 3070")
    parser.add_argument("--model_path", help="Path to pretrained model")
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file")
    parser.add_argument("--val_jsonl", required=True, help="Validation JSONL file")
    parser.add_argument("--output_dir", default="./local_gp_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Update global paths if provided
    if args.model_path:
        MODEL_PATH = args.model_path
    TRAIN_JSONL = args.train_jsonl
    VAL_JSONL = args.val_jsonl
    OUTPUT_DIR = args.output_dir
    
    print(f"Using model path: {MODEL_PATH}")
    print(f"Using train data: {TRAIN_JSONL}")
    print(f"Using validation data: {VAL_JSONL}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    clear_gpu_memory()
    results = main()