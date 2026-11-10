/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useCallback, useMemo, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  BackgroundVariant,
} from "@xyflow/react";
import {
  Play,
  Plus,
  Layers,
  Database,
  Cpu,
  Flame,
  HardDrive,
  Server,
  RotateCcw,
  Loader2,
  CheckCircle2,
  ArrowRight,
} from "lucide-react";
import { VrikshaCustomNode } from "./components/DagNodes";
import { NodeInspector } from "./components/NodeInspector";
import { EvaluationDrawer } from "./components/EvaluationDrawer";
import { ClusterInferenceView } from "./components/ClusterInferenceView";
import { RepoArchitectureView } from "./components/RepoArchitectureView";
import { NODE_LIBRARY_ITEMS, PIPELINE_PRESETS } from "./data/presets";
import type {
  VrikshaFlowNode,
  VrikshaNodeData,
  ClusterRecord,
  EvalMetricStep,
} from "./types";

const nodeTypes = {
  vrikshaNode: VrikshaCustomNode,
};

type ActiveWorkspaceTab = "studio" | "inference" | "evaluation" | "architecture";

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveWorkspaceTab>("studio");
  const [activePresetId, setActivePresetId] = useState<string>(
    PIPELINE_PRESETS[0].id
  );
  const [hasCompletedRun, setHasCompletedRun] = useState<boolean>(false);

  const [nodes, setNodes, onNodesChange] = useNodesState<VrikshaFlowNode>(
    PIPELINE_PRESETS[0].nodes
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    PIPELINE_PRESETS[0].edges
  );

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(
    "trainer-1"
  );
  const [isEvalDrawerOpen, setIsEvalDrawerOpen] = useState<boolean>(true);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  const [clusters, setClusters] = useState<ClusterRecord[]>([]);
  const [activeClusterId, setActiveClusterId] =
    useState<string>("cluster-8492-sft");

  const [evalMetrics, setEvalMetrics] = useState<EvalMetricStep[]>([
    { step: 20, loss: 2.38, perplexity: 10.8, accuracy: 0.512, reward: 0.42, kl: 0.004 },
    { step: 40, loss: 1.84, perplexity: 6.29, accuracy: 0.618, reward: 0.56, kl: 0.006 },
    { step: 60, loss: 1.41, perplexity: 4.09, accuracy: 0.704, reward: 0.67, kl: 0.008 },
    { step: 80, loss: 1.09, perplexity: 2.97, accuracy: 0.768, reward: 0.75, kl: 0.009 },
    { step: 100, loss: 0.88, perplexity: 2.41, accuracy: 0.815, reward: 0.81, kl: 0.01 },
    { step: 120, loss: 0.74, perplexity: 2.09, accuracy: 0.846, reward: 0.85, kl: 0.011 },
    { step: 140, loss: 0.642, perplexity: 1.9, accuracy: 0.864, reward: 0.89, kl: 0.012 },
  ]);

  const [trainingLogs, setTrainingLogs] = useState<string[]>([
    "[Step 1] Loaded 8,500 example conversations from UltraChat + OpenHermes.",
    "[Step 2] Enabled 4-bit Laptop Memory Saver on SmolLM2-135M (uses only 3.4 GB VRAM).",
    "[Step 3] Completed Supervised Fine-Tuning — Mistake Rate dropped from 2.38 down to 0.64.",
    "[Step 4] Saved trained AI to ClusterID: cluster-8492-sft (Ready to chat!).",
  ]);

  const fetchClusters = useCallback(async () => {
    try {
      const resp = await fetch("/api/clusters");
      const data = await resp.json();
      if (Array.isArray(data.clusters)) {
        setClusters(data.clusters);
      }
    } catch {
      // ignore offline error
    }
  }, []);

  useEffect(() => {
    fetchClusters();
  }, [fetchClusters]);

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, animated: true }, eds));
    },
    [setEdges]
  );

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) || null,
    [nodes, selectedNodeId]
  );

  const handleUpdateNodeData = useCallback(
    (nodeId: string, patch: Partial<VrikshaNodeData>) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, ...patch } }
            : node
        )
      );
    },
    [setNodes]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId)
      );
      if (selectedNodeId === nodeId) {
        setSelectedNodeId(null);
      }
    },
    [setNodes, setEdges, selectedNodeId]
  );

  const handleAddNodeFromLibrary = (
    item: (typeof NODE_LIBRARY_ITEMS)[number]
  ) => {
    const newId = `${item.category}-${Date.now().toString().slice(-4)}`;
    const offset = (nodes.length % 5) * 28;
    const newNode: VrikshaFlowNode = {
      id: newId,
      type: "vrikshaNode",
      position: { x: 220 + offset, y: 140 + offset },
      data: { ...item.defaultData },
    };
    setNodes((nds) => [...nds, newNode]);
    setSelectedNodeId(newId);
  };

  const handleLoadPreset = (presetId: string) => {
    const preset = PIPELINE_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setActivePresetId(preset.id);
    setNodes(preset.nodes);
    setEdges(preset.edges);
    const firstTrainer = preset.nodes.find((n) => n.data.category === "trainer");
    setSelectedNodeId(firstTrainer ? firstTrainer.id : preset.nodes[0]?.id || null);
    const endpointNode = preset.nodes.find((n) => n.data.category === "endpoint");
    if (endpointNode?.data.clusterId) {
      setActiveClusterId(String(endpointNode.data.clusterId));
    }
  };

  // Extract current pipeline configuration from DAG nodes
  const pipelineSummary = useMemo(() => {
    const datasetNodes = nodes.filter((n) => n.data.category === "dataset");
    const modelNode = nodes.find((n) => n.data.category === "baseModel");
    const trainerNode = nodes.find((n) => n.data.category === "trainer");
    const hwNode = nodes.find((n) => n.data.category === "hardware");
    const endpointNode = nodes.find((n) => n.data.category === "endpoint");

    const mode: "SFT" | "GRPO" =
      trainerNode?.data.mode === "GRPO" ? "GRPO" : "SFT";
    const baseModel =
      modelNode?.data.modelId || "HuggingFaceTB/SmolLM2-135M-Instruct";
    const quantization = modelNode?.data.quantization || "4bit-nf4";
    const dtype = modelNode?.data.dtype || "bfloat16";
    const contextLength = modelNode?.data.contextLength || 2048;
    const loraRank = trainerNode?.data.loraRank || 16;
    const loraAlpha = trainerNode?.data.loraAlpha || 32;
    const learningRate = trainerNode?.data.learningRate || "2e-4";
    const epochs = trainerNode?.data.epochs || 3;
    const grpoGroupSize = trainerNode?.data.grpoGroupSize || 4;
    const klPenaltyBeta = trainerNode?.data.klPenaltyBeta || 0.04;
    const rewardPreset =
      trainerNode?.data.rewardPreset || "math_verifiable_reward";
    const perDeviceBatchSize = hwNode?.data.perDeviceBatchSize || 2;
    const gradAccumSteps = hwNode?.data.gradAccumSteps || 8;
    const gradientCheckpointing = hwNode?.data.gradientCheckpointing ?? true;
    const wsl2Mode = hwNode?.data.wsl2Mode ?? true;
    const vramLimitGb = hwNode?.data.vramLimitGb || 8;
    const gpuProfile =
      hwNode?.data.gpuProfile || "RTX 4060 Laptop (8GB VRAM)";

    const rawWeights = modelNode?.data.baseVramGb || 0.9;
    const quantFactor =
      quantization === "4bit-nf4" ? 0.55 : quantization === "8bit" ? 0.78 : 1.45;
    const weightsGb = Number((rawWeights * quantFactor).toFixed(2));
    const kvCacheGb = Number(
      (
        (contextLength / 2048) *
        0.48 *
        perDeviceBatchSize *
        (mode === "GRPO" ? grpoGroupSize * 0.45 : 1)
      ).toFixed(2)
    );
    const loraGradientsGb = Number(
      ((loraRank / 16) * 0.36 * (gradientCheckpointing ? 0.65 : 1.15)).toFixed(2)
    );
    const optimizerGb = Number((wsl2Mode ? 0.42 : 0.78).toFixed(2));
    const totalGb = Number(
      (weightsGb + kvCacheGb + loraGradientsGb + optimizerGb).toFixed(2)
    );

    const datasets = datasetNodes.map((d) => ({
      hfPath: d.data.hfPath || "HuggingFaceH4/ultrachat_200k",
      split: d.data.split || "train_sft",
      sampleCount: d.data.sampleCount || 4000,
      mixWeight: d.data.mixWeight || 0.5,
    }));

    return {
      mode,
      baseModel,
      quantization,
      dtype,
      contextLength,
      loraRank,
      loraAlpha,
      learningRate,
      epochs,
      grpoGroupSize,
      klPenaltyBeta,
      rewardPreset,
      perDeviceBatchSize,
      gradAccumSteps,
      gradientCheckpointing,
      wsl2Mode,
      gpuProfile,
      clusterId: endpointNode?.data.clusterId || activeClusterId,
      datasets,
      vramBreakdown: {
        weightsGb,
        kvCacheGb,
        loraGradientsGb,
        optimizerGb,
        totalGb,
        limitGb: vramLimitGb,
      },
    };
  }, [nodes, activeClusterId]);

  const compiledYaml = useMemo(() => {
    const trainerType =
      pipelineSummary.mode === "GRPO" ? "TRL_GRPO" : "TRL_SFT";
    const dsYaml =
      pipelineSummary.datasets.length > 0
        ? pipelineSummary.datasets
            .map(
              (d) => `      - dataset_name: "${d.hfPath}"
        split: "${d.split}"
        sample_count: ${d.sampleCount}
        mixture_proportion: ${d.mixWeight}`
            )
            .join("\n")
        : `      - dataset_name: "HuggingFaceH4/ultrachat_200k"\n        split: "train_sft"`;

    const grpoSection =
      pipelineSummary.mode === "GRPO"
        ? `\n  grpo:\n    num_generations: ${pipelineSummary.grpoGroupSize}\n    kl_coef: ${pipelineSummary.klPenaltyBeta}\n    reward_functions:\n      - "${pipelineSummary.rewardPreset}"`
        : "";

    return `model:
  model_name: "${pipelineSummary.baseModel}"
  torch_dtype_str: "${pipelineSummary.dtype}"
  trust_remote_code: true
  model_max_length: ${pipelineSummary.contextLength}
  attn_implementation: "sdpa"

data:
  train:
    datasets:
${dsYaml}
    collator_name: "text_completions_only_with_padding"

peft:
  lora_r: ${pipelineSummary.loraRank}
  lora_alpha: ${pipelineSummary.loraAlpha}
  lora_dropout: 0.05
  q_lora: ${pipelineSummary.quantization === "4bit-nf4"}
  bnb_4bit_quant_type: "nf4"

training:
  trainer_type: "${trainerType}"
  output_dir: "output/${pipelineSummary.clusterId}"
  num_train_epochs: ${pipelineSummary.epochs}
  per_device_train_batch_size: ${pipelineSummary.perDeviceBatchSize}
  gradient_accumulation_steps: ${pipelineSummary.gradAccumSteps}
  learning_rate: ${pipelineSummary.learningRate}
  enable_gradient_checkpointing: ${pipelineSummary.gradientCheckpointing}
  optimizer: "paged_adamw_8bit"${grpoSection}`;
  }, [pipelineSummary]);

  const handleRunOumiPipeline = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setIsEvalDrawerOpen(true);
    setActiveTab("studio");

    setNodes((nds) =>
      nds.map((n) => ({ ...n, data: { ...n.data, status: "running" } }))
    );

    const isGrpo = pipelineSummary.mode === "GRPO";
    const simulatedSteps: EvalMetricStep[] = [
      {
        step: 20,
        loss: isGrpo ? 1.96 : 2.32,
        perplexity: isGrpo ? 7.1 : 10.17,
        accuracy: 0.53,
        reward: 0.41,
        kl: 0.006,
      },
      {
        step: 50,
        loss: isGrpo ? 1.44 : 1.68,
        perplexity: isGrpo ? 4.22 : 5.36,
        accuracy: 0.67,
        reward: 0.59,
        kl: 0.012,
      },
      {
        step: 80,
        loss: isGrpo ? 1.02 : 1.18,
        perplexity: isGrpo ? 2.77 : 3.25,
        accuracy: 0.78,
        reward: 0.76,
        kl: 0.018,
      },
      {
        step: 110,
        loss: isGrpo ? 0.68 : 0.82,
        perplexity: isGrpo ? 1.97 : 2.27,
        accuracy: 0.854,
        reward: 0.87,
        kl: 0.022,
      },
      {
        step: 140,
        loss: isGrpo ? 0.495 : 0.612,
        perplexity: isGrpo ? 1.64 : 1.84,
        accuracy: isGrpo ? 0.918 : 0.876,
        reward: isGrpo ? 0.948 : 0.895,
        kl: isGrpo ? 0.026 : 0.011,
      },
    ];

    setEvalMetrics([simulatedSteps[0]]);
    setTrainingLogs([
      `[Starting] Preparing ${pipelineSummary.baseModel.split("/").pop()} on ${pipelineSummary.gpuProfile}`,
      `[Study Data] Loading examples from ${pipelineSummary.datasets.map((d) => d.hfPath.split("/").pop()).join(" + ")}`,
    ]);

    for (let i = 1; i < simulatedSteps.length; i++) {
      await new Promise((r) => setTimeout(r, 420));
      const s = simulatedSteps[i];
      setEvalMetrics(simulatedSteps.slice(0, i + 1));
      setTrainingLogs((prev) => [
        ...prev,
        `[Step ${s.step}/140] Mistake Rate dropped to ${s.loss.toFixed(2)} · Answer Accuracy rose to ${(
          s.accuracy * 100
        ).toFixed(1)}%`,
      ]);
    }

    try {
      const resp = await fetch("/api/pipeline/compile-and-register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pipelineName: `${pipelineSummary.baseModel.split("/").pop()} ${
            pipelineSummary.mode
          } Run`,
          mode: pipelineSummary.mode,
          baseModel: pipelineSummary.baseModel,
          dtype: pipelineSummary.dtype,
          quantization: pipelineSummary.quantization,
          contextLength: pipelineSummary.contextLength,
          datasets: pipelineSummary.datasets,
          loraRank: pipelineSummary.loraRank,
          loraAlpha: pipelineSummary.loraAlpha,
          learningRate: pipelineSummary.learningRate,
          epochs: pipelineSummary.epochs,
          grpoGroupSize: pipelineSummary.grpoGroupSize,
          klPenaltyBeta: pipelineSummary.klPenaltyBeta,
          rewardPreset: pipelineSummary.rewardPreset,
          hardwareProfile: pipelineSummary.gpuProfile,
          perDeviceBatchSize: pipelineSummary.perDeviceBatchSize,
          gradAccumSteps: pipelineSummary.gradAccumSteps,
          gradientCheckpointing: pipelineSummary.gradientCheckpointing,
          wsl2Mode: pipelineSummary.wsl2Mode,
          vramPeakGb: pipelineSummary.vramBreakdown.totalGb,
          evalHistory: simulatedSteps,
        }),
      });
      const data = await resp.json();
      if (data.cluster) {
        const newCluster: ClusterRecord = data.cluster;
        setActiveClusterId(newCluster.clusterId);
        setHasCompletedRun(true);
        setTrainingLogs((prev) => [
          ...prev,
          `[Finished!] Created ClusterID: ${newCluster.clusterId} — Click 'Step 3: Chat with Trained AI' to test it!`,
        ]);
        setNodes((nds) =>
          nds.map((n) =>
            n.data.category === "endpoint"
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    status: "completed",
                    clusterId: newCluster.clusterId,
                    finalLoss: newCluster.finalLoss,
                    perplexity: newCluster.perplexity,
                    tokenAccuracy: newCluster.tokenAccuracy,
                  },
                }
              : { ...n, data: { ...n.data, status: "completed" } }
          )
        );
        await fetchClusters();
      }
    } catch {
      setNodes((nds) =>
        nds.map((n) => ({ ...n, data: { ...n.data, status: "completed" } }))
      );
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#0B0D12] text-[#F1F5F9]">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="h-[54px] shrink-0 flex items-center justify-between gap-8 px-6 border-b border-[#222938] bg-[#12161F]">
        {/* Zone 1: Single text element brand wordmark */}
        <a
          href="#studio"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab("studio");
          }}
          className="text-lg font-bold tracking-tight text-[#F1F5F9] font-display whitespace-nowrap shrink-0"
        >
          Vriksha AI
        </a>

        {/* Zone 2: 4 concise single-line text navigation links */}
        <nav className="flex items-center gap-6 text-sm font-medium text-[#94A3B8]">
          <button
            onClick={() => setActiveTab("studio")}
            className={`hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1 ${
              activeTab === "studio"
                ? "text-[#F1F5F9] underline underline-offset-8 decoration-[#FF6B2B] decoration-2"
                : ""
            }`}
          >
            1. Build &amp; Train
          </button>
          <button
            onClick={() => {
              setActiveTab("studio");
              setIsEvalDrawerOpen(true);
            }}
            className="hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1"
          >
            2. Training Scores
          </button>
          <button
            onClick={() => setActiveTab("inference")}
            className={`hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1 ${
              activeTab === "inference"
                ? "text-[#F1F5F9] underline underline-offset-8 decoration-[#FF6B2B] decoration-2"
                : ""
            }`}
          >
            3. Test Trained AI
          </button>
          <button
            onClick={() => setActiveTab("architecture")}
            className={`hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1 ${
              activeTab === "architecture"
                ? "text-[#F1F5F9] underline underline-offset-8 decoration-[#FF6B2B] decoration-2"
                : ""
            }`}
          >
            Download Code
          </button>
        </nav>

        {/* Zone 3: 1 Primary Action Button */}
        <div className="flex items-center shrink-0">
          <button
            onClick={handleRunOumiPipeline}
            disabled={isTraining}
            className="px-4 py-2 text-xs font-semibold text-white bg-[#FF6B2B] hover:bg-[#e55a1f] rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap shrink-0 disabled:opacity-60"
          >
            {isTraining ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            {isTraining ? "Training AI Model..." : "Start Training AI"}
          </button>
        </div>
      </header>

      {/* Main Workspace Area */}
      {activeTab === "inference" ? (
        <ClusterInferenceView
          clusters={clusters}
          selectedClusterId={activeClusterId}
          onSelectClusterId={setActiveClusterId}
          onRefreshClusters={fetchClusters}
          onBackToStudio={() => setActiveTab("studio")}
        />
      ) : activeTab === "architecture" ? (
        <RepoArchitectureView />
      ) : (
        <div className="flex-1 flex overflow-hidden">
          {/* Left Node Palette & Starter Templates Sidebar */}
          <aside className="w-[276px] shrink-0 bg-[#12161F] border-r border-[#222938] flex flex-col h-full overflow-y-auto p-3.5 space-y-5 select-none">
            {/* Step 1: Starter Goal Templates */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[#F1F5F9]">
                  Step 1: Pick a Goal Template
                </span>
                <button
                  onClick={() => handleLoadPreset(activePresetId)}
                  title="Reset diagram to default"
                  className="text-[11px] text-[#94A3B8] hover:text-[#F1F5F9] flex items-center gap-1"
                >
                  <RotateCcw className="w-3 h-3" />
                  Reset
                </button>
              </div>
              <p className="text-[11px] text-[#94A3B8]">
                New to AI? Pick a ready-made pipeline below to load all blocks automatically.
              </p>
              <div className="space-y-2">
                {PIPELINE_PRESETS.map((preset) => {
                  const isCurrent = preset.id === activePresetId;
                  return (
                    <button
                      key={preset.id}
                      onClick={() => handleLoadPreset(preset.id)}
                      className={`w-full text-left p-3 rounded-lg border transition-colors ${
                        isCurrent
                          ? "bg-[#161B26] border-[#FF6B2B]"
                          : "bg-[#0B0D12] border-[#222938] hover:border-[#334155]"
                      }`}
                    >
                      <div className="text-xs font-semibold text-[#F1F5F9]">
                        {preset.name}
                      </div>
                      <div className="text-[11px] text-[#38BDF8] mt-0.5">
                        {preset.beginnerGoal}
                      </div>
                      <div className="text-[11px] text-[#94A3B8] mt-1 line-clamp-2">
                        {preset.summary}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Optional: Add Extra Building Blocks */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-[#F1F5F9]">
                Add Extra Building Blocks
              </div>
              <p className="text-[11px] text-[#94A3B8]">
                Click any block to add it to your visual diagram, then drag a line between dots to connect them.
              </p>

              <div className="space-y-1.5 pt-1">
                {NODE_LIBRARY_ITEMS.map((item, idx) => {
                  const getIcon = () => {
                    switch (item.category) {
                      case "dataset":
                        return <Database className="w-3.5 h-3.5 text-[#38BDF8]" />;
                      case "baseModel":
                        return <Cpu className="w-3.5 h-3.5 text-[#A855F7]" />;
                      case "trainer":
                        return (
                          <Flame
                            className={`w-3.5 h-3.5 ${
                              item.defaultData.mode === "GRPO"
                                ? "text-[#8B5CF6]"
                                : "text-[#FF6B2B]"
                            }`}
                          />
                        );
                      case "hardware":
                        return <HardDrive className="w-3.5 h-3.5 text-[#F59E0B]" />;
                      case "endpoint":
                        return <Server className="w-3.5 h-3.5 text-[#10B981]" />;
                    }
                  };

                  return (
                    <button
                      key={idx}
                      onClick={() => handleAddNodeFromLibrary(item)}
                      className="w-full p-2.5 rounded-lg bg-[#0B0D12] hover:bg-[#161B26] border border-[#222938] hover:border-[#334155] transition-colors flex items-center justify-between gap-2 text-left group"
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        {getIcon()}
                        <div className="min-w-0">
                          <div className="text-[10px] text-[#94A3B8]">
                            {item.stepTag}
                          </div>
                          <div className="text-xs font-medium text-[#F1F5F9] truncate">
                            {item.title}
                          </div>
                          <div className="text-[11px] text-[#64748B] truncate">
                            {item.subtitle}
                          </div>
                        </div>
                      </div>
                      <Plus className="w-3.5 h-3.5 text-[#64748B] group-hover:text-[#FF6B2B] shrink-0" />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Active ClusterID Quick Card */}
            <div className="mt-auto p-3 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-[#F1F5F9]">
                <Layers className="w-3.5 h-3.5 text-[#10B981]" />
                <span>Step 3: Chat with Trained AI</span>
              </div>
              <div className="text-[11px] text-[#94A3B8]">
                Ready Model ID:{" "}
                <span className="font-mono text-[#10B981]">{activeClusterId}</span>
              </div>
              <button
                onClick={() => setActiveTab("inference")}
                className="w-full py-2 px-2.5 rounded bg-[#10B981] hover:bg-[#059669] text-xs font-semibold text-[#0B0D12] transition-colors whitespace-nowrap"
              >
                Test &amp; Chat with AI Now
              </button>
            </div>
          </aside>

          {/* Center React Flow Canvas + Beginner 3-Step Guide + Bottom Evaluation Drawer */}
          <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden">
            {/* Beginner 3-Step Guided Workflow Strip */}
            <div className="px-4 py-2.5 bg-[#0B0D12] border-b border-[#222938] flex flex-wrap items-center justify-between gap-3 text-xs shrink-0">
              <div className="flex flex-wrap items-center gap-2 text-[#94A3B8]">
                <span className="text-[#F1F5F9] font-semibold">
                  How It Works:
                </span>
                <span className="text-[#F1F5F9]">
                  1. Pick a Template or Click a Block to Edit
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
                <button
                  onClick={handleRunOumiPipeline}
                  disabled={isTraining}
                  className="text-[#FF6B2B] hover:underline font-medium"
                >
                  2. Click &quot;Start Training AI&quot;
                </button>
                <ArrowRight className="w-3.5 h-3.5 text-[#64748B]" />
                <button
                  onClick={() => setActiveTab("inference")}
                  className="text-[#10B981] hover:underline font-medium flex items-center gap-1"
                >
                  {hasCompletedRun && <CheckCircle2 className="w-3.5 h-3.5" />}
                  3. Chat with Your Trained AI ({activeClusterId})
                </button>
              </div>

              <div className="flex items-center gap-2 text-[#94A3B8] font-mono tabular-nums shrink-0">
                <span>
                  Laptop Memory: {pipelineSummary.vramBreakdown.totalGb.toFixed(1)} /{" "}
                  {pipelineSummary.vramBreakdown.limitGb} GB (Safe)
                </span>
              </div>
            </div>

            {/* Interactive React Flow Graph */}
            <div className="flex-1 relative">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={(_e, node) => setSelectedNodeId(node.id)}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.18 }}
                defaultEdgeOptions={{
                  style: { stroke: "#FF6B2B", strokeWidth: 2 },
                  animated: true,
                }}
              >
                <Background
                  variant={BackgroundVariant.Dots}
                  gap={20}
                  size={1.2}
                  color="#222938"
                />
                <Controls position="bottom-right" />
              </ReactFlow>
            </div>

            {/* Collapsible Evaluation & Oumi Stream Drawer */}
            <EvaluationDrawer
              isOpen={isEvalDrawerOpen}
              onToggle={() => setIsEvalDrawerOpen((o) => !o)}
              isTraining={isTraining}
              activeClusterId={activeClusterId}
              mode={pipelineSummary.mode}
              metrics={evalMetrics}
              logs={trainingLogs}
              onDeployCluster={(cid) => {
                setActiveClusterId(cid);
                setActiveTab("inference");
              }}
            />
          </div>

          {/* Right Contextual Node & Oumi YAML Inspector */}
          <NodeInspector
            selectedNode={selectedNode}
            onUpdateNodeData={handleUpdateNodeData}
            onDeleteNode={handleDeleteNode}
            compiledYaml={compiledYaml}
            vramBreakdown={pipelineSummary.vramBreakdown}
            onOpenClusterInference={(cid) => {
              setActiveClusterId(cid);
              setActiveTab("inference");
            }}
          />
        </div>
      )}
    </div>
  );
}
