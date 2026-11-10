"""
Vriksha AI — Oumi Configuration & WSL2 Hardware Optimizer
n2n-back/apis/utils/oumi_generator.py
Generates valid Oumi YAML configurations for SFT and GRPO RLHF pipelines
optimized for consumer GPUs (RTX 3050 / 4060 4GB–8GB VRAM).
"""

import os
import yaml
from typing import Dict, Any, List


def generate_oumi_yaml(pipeline_spec: Dict[str, Any], cluster_id: str) -> str:
    """
    Converts a Vriksha AI DAG pipeline specification into an executable Oumi YAML config.
    """
    model_node = pipeline_spec.get("baseModel", {})
    trainer_node = pipeline_spec.get("trainer", {})
    datasets: List[Dict[str, Any]] = pipeline_spec.get("datasets", [])
    hardware_node = pipeline_spec.get("hardware", {})

    mode = trainer_node.get("mode", "SFT")
    trainer_type = "TRL_GRPO" if mode == "GRPO" else "TRL_SFT"

    config = {
        "model": {
            "model_name": model_node.get("modelId", "HuggingFaceTB/SmolLM2-135M-Instruct"),
            "torch_dtype_str": model_node.get("dtype", "bfloat16"),
            "trust_remote_code": True,
            "model_max_length": int(model_node.get("contextLength", 2048)),
            "attn_implementation": "sdpa",
        },
        "data": {
            "train": {
                "datasets": [
                    {
                        "dataset_name": ds.get("hfPath", "HuggingFaceH4/ultrachat_200k"),
                        "split": ds.get("split", "train_sft"),
                        "sample_count": int(ds.get("sampleCount", 2500)),
                        "mixture_proportion": float(ds.get("mixWeight", 1.0)),
                    }
                    for ds in datasets
                ],
                "collator_name": "text_completions_only_with_padding",
                "seed": 42,
            }
        },
        "peft": {
            "lora_r": int(trainer_node.get("loraRank", 16)),
            "lora_alpha": int(trainer_node.get("loraAlpha", 32)),
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "q_lora": model_node.get("quantization", "4bit-nf4") == "4bit-nf4",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "training": {
            "trainer_type": trainer_type,
            "output_dir": f"output/{cluster_id}",
            "num_train_epochs": int(trainer_node.get("epochs", 3)),
            "per_device_train_batch_size": int(hardware_node.get("perDeviceBatchSize", 2)),
            "gradient_accumulation_steps": int(hardware_node.get("gradAccumSteps", 8)),
            "learning_rate": float(trainer_node.get("learningRate", 2e-4)),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "enable_gradient_checkpointing": bool(hardware_node.get("gradientCheckpointing", True)),
            "optimizer": "paged_adamw_8bit",
            "logging_steps": 10,
            "save_steps": 100,
            "dataloader_num_workers": 2 if hardware_node.get("wsl2Mode", True) else 4,
        },
    }

    if mode == "GRPO":
        config["training"]["grpo"] = {
            "num_generations": int(trainer_node.get("grpoGroupSize", 4)),
            "kl_coef": float(trainer_node.get("klPenaltyBeta", 0.04)),
            "reward_functions": [trainer_node.get("rewardPreset", "math_verifiable_reward")],
            "temperature": 0.7,
        }

    return yaml.dump(config, sort_keys=False, default_flow_style=False)
