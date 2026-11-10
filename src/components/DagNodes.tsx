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
        return <Database className="w-5 h-5 text-[#38BDF8] shrink-0" />;
      case "baseModel":
        return <Cpu className="w-5 h-5 text-[#A855F7] shrink-0" />;
      case "trainer":
        return (
          <Flame
            className={`w-5 h-5 shrink-0 ${
              data.mode === "GRPO" ? "text-[#8B5CF6]" : "text-[#FF6B2B]"
            }`}
          />
        );
      case "hardware":
        return <HardDrive className="w-5 h-5 text-[#F59E0B] shrink-0" />;
      case "endpoint":
        return <Server className="w-5 h-5 text-[#10B981] shrink-0" />;
    }
  };

  const getStepTitle = () => {
    switch (category) {
      case "dataset":
        return "STEP 1 · STUDY DATA";
      case "baseModel":
        return "STEP 2 · STARTER AI";
      case "trainer":
        return "STEP 3 · TRAINING";
      case "hardware":
        return "MEMORY SAVER";
      case "endpoint":
        return "STEP 4 · FINISHED AI";
    }
  };

  const getMainSummary = () => {
    switch (category) {
      case "dataset":
        return `${(data.sampleCount ?? 5000).toLocaleString()} example Q&As`;
      case "baseModel":
        return `${data.modelId?.split("/").pop()?.replace("-Instruct", "") || "SmolLM2"} (Laptop Friendly)`;
      case "trainer":
        return data.mode === "GRPO"
          ? "Learn by Reward (Math & Logic)"
          : "Learn by Example (Standard)";
      case "hardware":
        return data.gpuProfile || "4-bit Laptop Saver";
      case "endpoint":
        return `ID: ${data.clusterId || "cluster-8492-sft"}`;
    }
  };

  return (
    <div
      className={`w-[228px] rounded-xl bg-[#161B26] border transition-all duration-150 ${getCategoryAccent()} border-t-4 ${
        selected
          ? "border-[#FF6B2B] ring-2 ring-[#FF6B2B]/30 shadow-xl shadow-black/60"
          : "border-[#222938] hover:border-[#334155]"
      }`}
    >
      {category !== "dataset" && (
        <Handle
          type="target"
          position={Position.Left}
          className="!w-3 !h-3 !bg-[#1A1F2B] !border-2 !border-[#94A3B8]"
        />
      )}

      <div className="p-4 space-y-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold tracking-wider text-[#94A3B8]">
            {getStepTitle()}
          </span>
          {status === "running" && (
            <Loader2 className="w-4 h-4 text-[#FF6B2B] animate-spin shrink-0" />
          )}
          {status === "completed" && (
            <CheckCircle2 className="w-4 h-4 text-[#10B981] shrink-0" />
          )}
        </div>

        <div className="flex items-center gap-2.5">
          {renderIcon()}
          <div className="min-w-0">
            <div className="text-[15px] font-bold text-[#F1F5F9] truncate font-display">
              {data.label}
            </div>
            <div className="text-xs text-[#94A3B8] truncate mt-0.5">
              {getMainSummary()}
            </div>
          </div>
        </div>

        <div className="pt-2 border-t border-[#222938]/80 flex items-center justify-between text-[11px] text-[#64748B]">
          <span>{selected ? "Editing on right →" : "Click to customize"}</span>
          {category === "endpoint" && (
            <span className="text-[#10B981] font-medium">
              {Math.round((data.tokenAccuracy ?? 0.86) * 100)}% score
            </span>
          )}
        </div>
      </div>

      {category !== "endpoint" && (
        <Handle
          type="source"
          position={Position.Right}
          className="!w-3 !h-3 !bg-[#1A1F2B] !border-2 !border-[#94A3B8]"
        />
      )}
    </div>
  );
});

VrikshaCustomNode.displayName = "VrikshaCustomNode";
