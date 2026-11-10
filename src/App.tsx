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
  Database,
  Cpu,
  Flame,
  HardDrive,
  Loader2,
  CheckCircle2,
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

type ActiveWorkspaceTab = "studio" | "inference" | "architecture";

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

  // Start with no node selected so the right panel shows the simple 3-step welcome guide
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  // Keep bottom scoreboard closed by default so the screen is calm and uncluttered
  const [isEvalDrawerOpen, setIsEvalDrawerOpen] = useState<boolean>(false);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  const [clusters, setClusters] = useState<ClusterRecord[]>([]);
  const [activeClusterId, setActiveClusterId] =
    useState<string>("cluster-cet-2026");

  const [evalMetrics, setEvalMetrics] = useState<EvalMetricStep[]>([
    { step: 25, loss: 2.12, perplexity: 8.3, accuracy: 0.52, reward: 0.45, kl: 0.006 },
    { step: 50, loss: 1.34, perplexity: 3.8, accuracy: 0.71, reward: 0.68, kl: 0.012 },
    { step: 75, loss: 0.79, perplexity: 2.2, accuracy: 0.84, reward: 0.85, kl: 0.017 },
    { step: 100, loss: 0.48, perplexity: 1.62, accuracy: 0.92, reward: 0.94, kl: 0.021 },
  ]);

  const [trainingLogs, setTrainingLogs] = useState<string[]>([
    "MAH MCA CET Small-Scale Training complete! Accuracy jumped from 48% to 92%.",
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
    const offset = (nodes.length % 4) * 32;
    const newNode: VrikshaFlowNode = {
      id: newId,
      type: "vrikshaNode",
      position: { x: 260 + offset, y: 260 + offset },
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
    setSelectedNodeId(null);
    const endpointNode = preset.nodes.find((n) => n.data.category === "endpoint");
    if (endpointNode?.data.clusterId) {
      setActiveClusterId(String(endpointNode.data.clusterId));
    }
  };

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
    const kvCacheGb = 0.65;
    const loraGradientsGb = Number(((loraRank / 16) * 0.25).toFixed(2));
    const optimizerGb = 0.42;
    const totalGb = Number(
      (weightsGb + kvCacheGb + loraGradientsGb + optimizerGb).toFixed(2)
    );

    const datasets = datasetNodes.map((d) => ({
      hfPath: d.data.hfPath || "HuggingFaceH4/ultrachat_200k",
      split: d.data.split || "train_sft",
      sampleCount: d.data.sampleCount || 5000,
      mixWeight: d.data.mixWeight || 1.0,
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
        sample_count: ${d.sampleCount}`
            )
            .join("\n")
        : `      - dataset_name: "HuggingFaceH4/ultrachat_200k"`;

    return `model:
  model_name: "${pipelineSummary.baseModel}"
  torch_dtype_str: "${pipelineSummary.dtype}"

data:
  train:
    datasets:
${dsYaml}

peft:
  lora_r: ${pipelineSummary.loraRank}
  lora_alpha: ${pipelineSummary.loraAlpha}
  q_lora: ${pipelineSummary.quantization === "4bit-nf4"}

training:
  trainer_type: "${trainerType}"
  output_dir: "output/${pipelineSummary.clusterId}"
  num_train_epochs: ${pipelineSummary.epochs}
  learning_rate: ${pipelineSummary.learningRate}`;
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
        step: 35,
        loss: isGrpo ? 1.85 : 2.15,
        perplexity: 6.5,
        accuracy: 0.58,
        reward: 0.52,
        kl: 0.008,
      },
      {
        step: 70,
        loss: isGrpo ? 1.22 : 1.42,
        perplexity: 3.8,
        accuracy: 0.72,
        reward: 0.71,
        kl: 0.014,
      },
      {
        step: 105,
        loss: isGrpo ? 0.78 : 0.91,
        perplexity: 2.3,
        accuracy: 0.83,
        reward: 0.85,
        kl: 0.019,
      },
      {
        step: 140,
        loss: isGrpo ? 0.49 : 0.61,
        perplexity: 1.7,
        accuracy: isGrpo ? 0.92 : 0.88,
        reward: isGrpo ? 0.94 : 0.89,
        kl: 0.022,
      },
    ];

    setEvalMetrics([simulatedSteps[0]]);
    setTrainingLogs(["Step 1/4: Reading example questions and answers..."]);

    for (let i = 1; i < simulatedSteps.length; i++) {
      await new Promise((r) => setTimeout(r, 400));
      const s = simulatedSteps[i];
      setEvalMetrics(simulatedSteps.slice(0, i + 1));
      setTrainingLogs([
        `Training progress: Accuracy reached ${Math.round(
          s.accuracy * 100
        )}% (Mistake rate down to ${s.loss.toFixed(2)})`,
      ]);
    }

    try {
      const resp = await fetch("/api/pipeline/compile-and-register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pipelineName:
            pipelineSummary.mode === "GRPO"
              ? "Custom Math & Logic AI"
              : "Custom Helpful Chat AI",
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
        setTrainingLogs([
          `Finished! Your AI is ready (${Math.round(
            newCluster.tokenAccuracy * 100
          )}% accuracy). Click "Chat with Trained AI" to test it!`,
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
      <header className="h-[56px] shrink-0 flex items-center justify-between gap-8 px-6 border-b border-[#222938] bg-[#12161F]">
        {/* Zone 1: Brand Wordmark */}
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

        {/* Zone 2: Simple 4-Link Navigation */}
        <nav className="flex items-center gap-6 text-sm font-medium text-[#94A3B8]">
          <button
            onClick={() => setActiveTab("studio")}
            className={`hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1 ${
              activeTab === "studio"
                ? "text-[#F1F5F9] underline underline-offset-8 decoration-[#FF6B2B] decoration-2"
                : ""
            }`}
          >
            1. Build Pipeline
          </button>
          <button
            onClick={() => {
              setActiveTab("studio");
              setIsEvalDrawerOpen((o) => !o);
            }}
            className="hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1"
          >
            2. View Scores
          </button>
          <button
            onClick={() => setActiveTab("inference")}
            className={`hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1 ${
              activeTab === "inference"
                ? "text-[#F1F5F9] underline underline-offset-8 decoration-[#FF6B2B] decoration-2"
                : ""
            }`}
          >
            3. Chat with Trained AI
          </button>
          <button
            onClick={() => setActiveTab("architecture")}
            className={`hover:text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0 py-1 ${
              activeTab === "architecture"
                ? "text-[#F1F5F9] underline underline-offset-8 decoration-[#FF6B2B] decoration-2"
                : ""
            }`}
          >
            Python Code
          </button>
        </nav>

        {/* Zone 3: 1 Primary CTA Button */}
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
            {isTraining ? "Training AI..." : "Train AI Now"}
          </button>
        </div>
      </header>

      {/* Main Content Area */}
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
          {/* Calm, Minimal Left Sidebar (240px) */}
          <aside className="w-[240px] shrink-0 bg-[#12161F] border-r border-[#222938] flex flex-col h-full overflow-y-auto p-4 space-y-6 select-none">
            {/* 1. Choose a Goal */}
            <div className="space-y-2.5">
              <div className="text-xs font-bold uppercase tracking-wider text-[#94A3B8]">
                1. Choose AI Goal
              </div>
              <div className="space-y-2">
                {PIPELINE_PRESETS.map((preset) => {
                  const isCurrent = preset.id === activePresetId;
                  return (
                    <button
                      key={preset.id}
                      onClick={() => handleLoadPreset(preset.id)}
                      className={`w-full text-left p-3 rounded-xl border transition-colors ${
                        isCurrent
                          ? "bg-[#161B26] border-[#FF6B2B]"
                          : "bg-[#0B0D12] border-[#222938] hover:border-[#334155]"
                      }`}
                    >
                      <div className="text-xs font-bold text-[#F1F5F9]">
                        {preset.name}
                      </div>
                      <div className="text-[11px] text-[#94A3B8] mt-1 leading-snug">
                        {preset.beginnerGoal}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* 2. Add Extra Block */}
            <div className="space-y-2.5">
              <div className="text-xs font-bold uppercase tracking-wider text-[#94A3B8]">
                2. Add Extra Block
              </div>
              <div className="space-y-1.5">
                {NODE_LIBRARY_ITEMS.map((item, idx) => {
                  const getIcon = () => {
                    switch (item.category) {
                      case "dataset":
                        return <Database className="w-4 h-4 text-[#38BDF8]" />;
                      case "baseModel":
                        return <Cpu className="w-4 h-4 text-[#A855F7]" />;
                      case "trainer":
                        return <Flame className="w-4 h-4 text-[#FF6B2B]" />;
                      case "hardware":
                        return <HardDrive className="w-4 h-4 text-[#F59E0B]" />;
                      default:
                        return null;
                    }
                  };

                  return (
                    <button
                      key={idx}
                      onClick={() => handleAddNodeFromLibrary(item)}
                      className="w-full px-3 py-2.5 rounded-lg bg-[#0B0D12] hover:bg-[#161B26] border border-[#222938] hover:border-[#334155] transition-colors flex items-center justify-between gap-2 text-left group"
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        {getIcon()}
                        <span className="text-xs font-medium text-[#F1F5F9] truncate">
                          {item.title}
                        </span>
                      </div>
                      <Plus className="w-3.5 h-3.5 text-[#64748B] group-hover:text-[#FF6B2B] shrink-0" />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Bottom Step 3 CTA */}
            <div className="mt-auto pt-4 border-t border-[#222938] space-y-2">
              <button
                onClick={() => setActiveTab("inference")}
                className="w-full py-2.5 px-3 rounded-lg bg-[#10B981] hover:bg-[#059669] text-xs font-semibold text-[#0B0D12] transition-colors flex items-center justify-center gap-1.5"
              >
                {hasCompletedRun && <CheckCircle2 className="w-3.5 h-3.5" />}
                Chat with Trained AI →
              </button>
            </div>
          </aside>

          {/* Center Visual Flow Canvas */}
          <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden">
            {/* Simple 1-Line Instruction Banner */}
            <div className="h-[42px] px-5 bg-[#0B0D12] border-b border-[#222938] flex items-center justify-between text-xs shrink-0">
              <div className="text-[#94A3B8]">
                <strong className="text-[#F1F5F9]">Visual Pipeline:</strong> Follow the 4 blocks from left to right. Click any block to customize it, then click{" "}
                <button
                  onClick={handleRunOumiPipeline}
                  className="text-[#FF6B2B] font-semibold hover:underline"
                >
                  Train AI Now
                </button>
                .
              </div>
            </div>

            {/* Spacious React Flow Canvas */}
            <div className="flex-1 relative">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={(_e, node) => setSelectedNodeId(node.id)}
                onPaneClick={() => setSelectedNodeId(null)}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.28 }}
                defaultEdgeOptions={{
                  style: { stroke: "#FF6B2B", strokeWidth: 2.5 },
                  animated: true,
                }}
              >
                <Background
                  variant={BackgroundVariant.Dots}
                  gap={24}
                  size={1.2}
                  color="#222938"
                />
                <Controls position="bottom-right" />
              </ReactFlow>
            </div>

            {/* Collapsible Scoreboard Drawer (Closed by Default) */}
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

          {/* Simplified Right Settings Panel */}
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
            onRunTraining={handleRunOumiPipeline}
            isTraining={isTraining}
          />
        </div>
      )}
    </div>
  );
}
