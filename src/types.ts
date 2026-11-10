import type { Node, Edge } from "@xyflow/react";

export type VrikshaNodeCategory =
  | "dataset"
  | "baseModel"
  | "trainer"
  | "hardware"
  | "endpoint";

export type NodeRunStatus = "idle" | "running" | "completed" | "error";

export interface VrikshaNodeData extends Record<string, unknown> {
  label: string;
  category: VrikshaNodeCategory;
  status?: NodeRunStatus;
  beginnerHint?: string;
  // Dataset properties
  hfPath?: string;
  split?: string;
  sampleCount?: number;
  mixWeight?: number;
  formatType?: "chatml" | "alpaca" | "preference_pairs" | "verifiable_math";
  // Base Model properties
  modelId?: string;
  paramCount?: string;
  quantization?: "4bit-nf4" | "8bit" | "bfloat16";
  dtype?: "bfloat16" | "float16";
  contextLength?: number;
  baseVramGb?: number;
  // Trainer (SFT / GRPO) properties
  mode?: "SFT" | "GRPO";
  loraRank?: number;
  loraAlpha?: number;
  learningRate?: string;
  epochs?: number;
  grpoGroupSize?: number;
  klPenaltyBeta?: number;
  rewardPreset?: string;
  // Hardware & WSL2 Optimizer properties
  gpuProfile?: "RTX 3050 Laptop (4GB VRAM)" | "RTX 4060 Laptop (8GB VRAM)" | "RTX 4070 Desktop (12GB VRAM)";
  vramLimitGb?: number;
  perDeviceBatchSize?: number;
  gradAccumSteps?: number;
  gradientCheckpointing?: boolean;
  pagedAdamw8bit?: boolean;
  wsl2Mode?: boolean;
  // Endpoint / Evaluation Sink properties
  clusterId?: string;
  vllmPagedAttention?: boolean;
  evalBenchmarks?: string[];
  finalLoss?: number;
  perplexity?: number;
  tokenAccuracy?: number;
}

export type VrikshaFlowNode = Node<VrikshaNodeData>;
export type VrikshaFlowEdge = Edge;

export interface EvalMetricStep {
  step: number;
  loss: number;
  perplexity: number;
  accuracy: number;
  reward: number;
  kl: number;
}

export interface ClusterRecord {
  clusterId: string;
  pipelineName: string;
  createdAt: string;
  baseModel: string;
  mode: "SFT" | "GRPO";
  datasets: string[];
  loraRank: number;
  loraAlpha: number;
  quantization: string;
  hardwareProfile: string;
  vramPeakGb: number;
  finalLoss: number;
  perplexity: number;
  tokenAccuracy: number;
  grpoRewardMean: number;
  klDivergence: number;
  adapterPath: string;
  oumiYaml: string;
  systemPersona: string;
  mountedInVllm: boolean;
  evalHistory: EvalMetricStep[];
}

export interface PipelinePreset {
  id: string;
  name: string;
  beginnerGoal: string;
  summary: string;
  mode: "SFT" | "GRPO";
  targetHardware: string;
  nodes: VrikshaFlowNode[];
  edges: VrikshaFlowEdge[];
}
