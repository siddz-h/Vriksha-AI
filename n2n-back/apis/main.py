"""
Vriksha AI — FastAPI Gateway & Oumi Job Orchestrator
n2n-back/apis/main.py
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from utils.oumi_generator import generate_oumi_yaml
from vllm_server import VLLMAdapterBridge

app = FastAPI(
    title="Vriksha AI Orchestrator API",
    description="Asynchronous FastAPI Gateway for Oumi SFT/GRPO Training & vLLM ClusterID Inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vllm_bridge = VLLMAdapterBridge()


class PipelineCompileRequest(BaseModel):
    pipeline_name: str
    spec: Dict[str, Any]


class ClusterInferenceRequest(BaseModel):
    cluster_id: str
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 512


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "framework": "Oumi + vLLM PagedAttention",
        "wsl2_optimized": True,
    }


@app.post("/api/pipeline/compile")
async def compile_pipeline(req: PipelineCompileRequest):
    cluster_id = f"cluster-{uuid.uuid4().hex[:6]}"
    yaml_config = generate_oumi_yaml(req.spec, cluster_id)
    os.makedirs(f"../../output/{cluster_id}", exist_ok=True)
    with open(f"../../output/{cluster_id}/oumi_config.yaml", "w") as f:
        f.write(yaml_config)
    return {
        "cluster_id": cluster_id,
        "oumi_yaml": yaml_config,
        "cli_command": f"oumi train -c output/{cluster_id}/oumi_config.yaml",
    }


@app.post("/api/inference/query")
async def query_cluster(req: ClusterInferenceRequest):
    await vllm_bridge.mount_cluster_lora(req.cluster_id)
    return await vllm_bridge.generate_with_cluster(
        cluster_id=req.cluster_id,
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        max_tokens=req.max_tokens,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
