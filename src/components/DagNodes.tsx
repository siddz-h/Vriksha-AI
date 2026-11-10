import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import {
  Database,
  Cpu,
  Flame,
  HardDrive,
  Server,
  CheckCircle2,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import type { VrikshaFlowNode } from "../types";

export const VrikshaCustomNode = memo(({ data, selected }: NodeProps<VrikshaFlowNode>) => {
  const category = data.category;
  const status = data.status || "idle";

  const getCategoryAccent = () => {
    switch (category) {
      case "dataset":
        return "border-t-[#38BDF8]";
      case "baseModel":
        return "border-t-[#A855F7]";
      case "trainer":
        return data.mode === "GRPO" ? "border-t-[#8B5CF6]" : "border-t-[#FF6B2B]";
      case "hardware":
        return "border-t-[#F59E0B]";
      case "endpoint":
        return "border-t-[#10B981]";
      default:
        return "border-t-[#64748B]";
    }
  };

  const renderIcon = () => {
    switch (category) {
      case "dataset":
        return <Database className="w-4 h-4 text-[#38BDF8] shrink-0" />;
      case "baseModel":
        return <Cpu className="w-4 h-4 text-[#A855F7] shrink-0" />;
      case "trainer":
        return (
          <Flame
            className={`w-4 h-4 shrink-0 ${
              data.mode === "GRPO" ? "text-[#8B5CF6]" : "text-[#FF6B2B]"
            }`}
          />
        );
      case "hardware":
        return <HardDrive className="w-4 h-4 text-[#F59E0B] shrink-0" />;
      case "endpoint":
        return <Server className="w-4 h-4 text-[#10B981] shrink-0" />;
    }
  };

  const getStepAndRoleLabel = () => {
    switch (category) {
      case "dataset":
        return "Step 1 · Training Data";
      case "baseModel":
        return "Step 1 · Starting AI Brain";
      case "hardware":
        return "Step 2 · Laptop Memory Saver";
      case "trainer":
        return data.mode === "GRPO"
          ? "Step 3 · Reward Trainer (GRPO)"
          : "Step 3 · Example Trainer (SFT)";
      case "endpoint":
        return "Step 4 · Trained AI Output";
    }
  };

  const getDefaultHint = () => {
    if (data.beginnerHint) return data.beginnerHint;
    switch (category) {
      case "dataset":
        return "Example Q&A conversations the AI studies.";
      case "baseModel":
        return "Compact starter AI model before training.";
      case "hardware":
        return "Keeps memory low so training runs on laptops.";
      case "trainer":
        return data.mode === "GRPO"
          ? "Rewards step-by-step logic and verified answers."
          : "Teaches the AI by copying helpful examples.";
      case "endpoint":
        return "Creates a ClusterID so you can chat with your AI.";
    }
  };

  return (
    <div
      className={`w-[288px] rounded-lg bg-[#161B26] border transition-all duration-150 ${getCategoryAccent()} border-t-2 ${
        selected
          ? "border-[#FF6B2B] ring-1 ring-[#FF6B2B]/40 shadow-lg shadow-black/50"
          : "border-[#222938] hover:border-[#334155]"
      }`}
    >
      {/* Input Handle for downstream nodes */}
      {(category === "trainer" || category === "endpoint") && (
        <Handle
          type="target"
          position={Position.Left}
          className="!w-2.5 !h-2.5 !bg-[#1A1F2B] !border-2 !border-[#94A3B8]"
        />
      )}

      {/* Node Header */}
      <div className="px-3.5 py-2.5 border-b border-[#222938] flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {renderIcon()}
          <div className="min-w-0">
            <div className="text-[11px] text-[#94A3B8] leading-none mb-1 truncate">
              {getStepAndRoleLabel()} ·{" "}
              {status === "running"
                ? "Training..."
                : status === "completed"
                ? "Done"
                : "Click to Edit"}
            </div>
            <div className="text-[14px] font-semibold text-[#F1F5F9] truncate font-display">
              {data.label}
            </div>
          </div>
        </div>

        <div className="shrink-0">
          {status === "running" && (
            <Loader2 className="w-4 h-4 text-[#FF6B2B] animate-spin" />
          )}
          {status === "completed" && (
            <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
          )}
          {status === "error" && (
            <AlertTriangle className="w-4 h-4 text-[#EF4444]" />
          )}
        </div>
      </div>

      {/* Plain-English Beginner Explanation */}
      <div className="px-3.5 py-2 bg-[#0B0D12]/60 border-b border-[#222938]/70 text-[11.5px] text-[#94A3B8] leading-snug">
        {getDefaultHint()}
      </div>

      {/* Node Body Parameters */}
      <div className="p-3.5 space-y-1.5 text-[12px]">
        {category === "dataset" && (
          <>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Dataset Source</span>
              <span className="font-mono text-[#F1F5F9] truncate max-w-[155px]">
                {data.hfPath?.split("/").pop() || data.hfPath}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Examples to Study</span>
              <span className="font-mono tabular-nums text-[#94A3B8]">
                {(data.sampleCount ?? 4000).toLocaleString()} pairs
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Share of Study Mix</span>
              <span className="font-mono tabular-nums text-[#38BDF8]">
                {Math.round((data.mixWeight ?? 0.5) * 100)}%
              </span>
            </div>
          </>
        )}

        {category === "baseModel" && (
          <>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">AI Brain Model</span>
              <span className="font-mono text-[#F1F5F9] truncate max-w-[155px]">
                {data.modelId?.split("/").pop() || data.modelId}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Memory Compression</span>
              <span className="font-mono tabular-nums text-[#A855F7]">
                {data.quantization === "4bit-nf4" ? "4-bit Saver (QLoRA)" : data.quantization}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Memory Needed</span>
              <span className="font-mono tabular-nums text-[#94A3B8]">
                ~{data.baseVramGb} GB VRAM
              </span>
            </div>
          </>
        )}

        {category === "trainer" && (
          <>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Learning Style</span>
              <span className="font-mono text-[#F1F5F9]">
                {data.mode === "GRPO" ? "Trial & Reward (GRPO)" : "Learn by Example (SFT)"}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Adapter Capacity</span>
              <span className="font-mono tabular-nums text-[#94A3B8]">
                Rank r={data.loraRank} ({data.loraRank === 8 ? "Light" : data.loraRank === 16 ? "Balanced" : "Deep"})
              </span>
            </div>
            {data.mode === "GRPO" ? (
              <div className="flex items-center justify-between gap-2">
                <span className="text-[#64748B]">Attempts per Question</span>
                <span className="font-mono tabular-nums text-[#8B5CF6]">
                  {data.grpoGroupSize} tries · Reward Verifier
                </span>
              </div>
            ) : (
              <div className="flex items-center justify-between gap-2">
                <span className="text-[#64748B]">Study Passes</span>
                <span className="font-mono tabular-nums text-[#FF6B2B]">
                  {data.epochs} Epochs · Speed {data.learningRate}
                </span>
              </div>
            )}
          </>
        )}

        {category === "hardware" && (
          <>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Laptop GPU Profile</span>
              <span className="font-mono text-[#F1F5F9] truncate max-w-[155px]">
                {data.gpuProfile?.replace(" Laptop", "")}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Examples at Once</span>
              <span className="font-mono tabular-nums text-[#94A3B8]">
                {data.perDeviceBatchSize} per step (Safe)
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Crash Protection</span>
              <span className="font-mono text-[#F59E0B]">
                {data.gradientCheckpointing ? "Memory Guard ON" : "Standard"}
              </span>
            </div>
          </>
        )}

        {category === "endpoint" && (
          <>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Your Trained AI ID</span>
              <span className="font-mono text-[#10B981] font-medium">
                {data.clusterId}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Answer Accuracy</span>
              <span className="font-mono tabular-nums text-[#F1F5F9]">
                {((data.tokenAccuracy ?? 0.864) * 100).toFixed(1)}% · Mistake Rate {data.finalLoss?.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[#64748B]">Status</span>
              <span className="font-mono text-[#10B981]">
                Ready to Chat in vLLM
              </span>
            </div>
          </>
        )}
      </div>

      {/* Output Handle for upstream nodes */}
      {category !== "endpoint" && (
        <Handle
          type="source"
          position={Position.Right}
          className="!w-2.5 !h-2.5 !bg-[#1A1F2B] !border-2 !border-[#94A3B8]"
        />
      )}
    </div>
  );
});

VrikshaCustomNode.displayName = "VrikshaCustomNode";
