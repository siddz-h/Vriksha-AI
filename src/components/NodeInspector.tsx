import { useState } from "react";
import {
  Sliders,
  FileCode2,
  Download,
  Copy,
  Check,
  Sparkles,
  Trash2,
  Play,
  HelpCircle,
} from "lucide-react";
import type { VrikshaFlowNode, VrikshaNodeData } from "../types";

interface NodeInspectorProps {
  selectedNode: VrikshaFlowNode | null;
  onUpdateNodeData: (nodeId: string, patch: Partial<VrikshaNodeData>) => void;
  onDeleteNode: (nodeId: string) => void;
  compiledYaml: string;
  vramBreakdown: {
    weightsGb: number;
    kvCacheGb: number;
    loraGradientsGb: number;
    optimizerGb: number;
    totalGb: number;
    limitGb: number;
  };
  onOpenClusterInference: (clusterId: string) => void;
}

export function NodeInspector({
  selectedNode,
  onUpdateNodeData,
  onDeleteNode,
  compiledYaml,
  vramBreakdown,
  onOpenClusterInference,
}: NodeInspectorProps) {
  const [activeTab, setActiveTab] = useState<"inspector" | "yaml">("inspector");
  const [copied, setCopied] = useState(false);
  const [rolloutLoading, setRolloutLoading] = useState(false);
  const [rolloutSamples, setRolloutSamples] = useState<
    Array<{ prompt: string; completion: string; rewardScore: number; tokenCount: number }>
  >([]);

  const handleCopyYaml = () => {
    navigator.clipboard.writeText(compiledYaml);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownloadYaml = () => {
    const blob = new Blob([compiledYaml], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "oumi_config.yaml";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePreviewRollout = async () => {
    if (!selectedNode) return;
    setRolloutLoading(true);
    try {
      const resp = await fetch("/api/oumi/preview-rollout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasetName: selectedNode.data.hfPath || "HuggingFaceH4/ultrachat_200k",
          mode: selectedNode.data.mode || "SFT",
          rewardPreset: selectedNode.data.rewardPreset || "math_verifiable_reward",
          baseModel: selectedNode.data.modelId || "HuggingFaceTB/SmolLM2-135M-Instruct",
        }),
      });
      const json = await resp.json();
      if (Array.isArray(json.samples)) {
        setRolloutSamples(json.samples);
      }
    } catch {
      // fallback handled silently
    } finally {
      setRolloutLoading(false);
    }
  };

  const vramRatio = Math.min(
    100,
    Math.round((vramBreakdown.totalGb / vramBreakdown.limitGb) * 100)
  );
  const vramStatusText =
    vramBreakdown.totalGb > vramBreakdown.limitGb
      ? "Too much memory for this laptop! Switch Memory Compression to '4-bit Saver'."
      : vramRatio > 82
      ? "Fits on your laptop, with Memory Guard keeping it stable."
      : `Safe to run! Uses less than half of your ${vramBreakdown.limitGb} GB laptop GPU memory.`;

  const getPlainEnglishExplanation = (category: VrikshaNodeData["category"]) => {
    switch (category) {
      case "dataset":
        return "Think of a Dataset as a textbook. Your AI reads these example questions and answers to learn how to respond accurately.";
      case "baseModel":
        return "This is the starting AI brain before training. Smaller brains (like SmolLM2-135M) train in minutes on a normal laptop.";
      case "trainer":
        return "This block controls HOW the AI learns: either by copying good examples (SFT) or by practicing math/logic problems and earning rewards (GRPO).";
      case "hardware":
        return "Consumer laptops have limited graphics memory (4GB–8GB VRAM). These settings compress training so your computer never runs out of memory.";
      case "endpoint":
        return "Once training finishes, Vriksha saves your custom AI under a unique ClusterID so you can immediately chat with it.";
    }
  };

  return (
    <aside className="w-[364px] shrink-0 bg-[#12161F] border-l border-[#222938] flex flex-col h-full select-none">
      {/* Segmented Control Header */}
      <div className="p-3 border-b border-[#222938] flex items-center gap-1 bg-[#0B0D12]/60">
        <button
          onClick={() => setActiveTab("inspector")}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 rounded text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
            activeTab === "inspector"
              ? "bg-[#1A1F2B] text-[#F1F5F9] border border-[#222938]"
              : "text-[#94A3B8] hover:text-[#F1F5F9]"
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          Easy Settings
        </button>
        <button
          onClick={() => setActiveTab("yaml")}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 rounded text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
            activeTab === "yaml"
              ? "bg-[#1A1F2B] text-[#F1F5F9] border border-[#222938]"
              : "text-[#94A3B8] hover:text-[#F1F5F9]"
          }`}
        >
          <FileCode2 className="w-3.5 h-3.5" />
          Auto-Generated Code
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {activeTab === "yaml" ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-[#F1F5F9]">
                  Oumi Training Recipe (YAML)
                </h3>
                <p className="text-xs text-[#94A3B8]">
                  Vriksha writes this configuration file for you automatically as you edit blocks.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopyYaml}
                className="flex-1 py-1.5 px-2.5 rounded bg-[#1A1F2B] hover:bg-[#222938] border border-[#222938] text-xs text-[#F1F5F9] flex items-center justify-center gap-1 whitespace-nowrap shrink-0"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-[#10B981]" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                {copied ? "Copied Recipe" : "Copy Recipe"}
              </button>
              <button
                onClick={handleDownloadYaml}
                className="flex-1 py-1.5 px-2.5 rounded bg-[#FF6B2B] hover:bg-[#e55a1f] text-xs font-medium text-white flex items-center justify-center gap-1 whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                Download YAML
              </button>
            </div>

            <pre className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] text-[11.5px] leading-relaxed font-mono text-[#F1F5F9] overflow-x-auto select-text">
              {compiledYaml}
            </pre>
          </div>
        ) : !selectedNode ? (
          <div className="py-10 px-3 text-center space-y-2 rounded-lg bg-[#0B0D12] border border-[#222938]">
            <HelpCircle className="w-6 h-6 text-[#FF6B2B] mx-auto" />
            <p className="text-sm font-semibold text-[#F1F5F9]">
              Click Any Block on the Canvas
            </p>
            <p className="text-xs text-[#94A3B8] leading-relaxed">
              Click any block in the center diagram to see what it does in plain English and customize how your AI is trained.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-2 pb-3 border-b border-[#222938]">
              <div>
                <div className="text-xs text-[#94A3B8]">
                  Selected Block ·{" "}
                  <span className="capitalize">{selectedNode.data.category}</span>
                </div>
                <h3 className="text-base font-semibold text-[#F1F5F9] mt-0.5">
                  {selectedNode.data.label}
                </h3>
              </div>
              <button
                onClick={() => onDeleteNode(selectedNode.id)}
                title="Remove this block"
                className="p-1.5 rounded text-[#94A3B8] hover:text-[#EF4444] hover:bg-[#1A1F2B] transition-colors"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {/* Plain-English Beginner Callout */}
            <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-1">
              <div className="text-xs font-semibold text-[#F1F5F9] flex items-center gap-1.5">
                <HelpCircle className="w-3.5 h-3.5 text-[#38BDF8]" />
                <span>What does this block do?</span>
              </div>
              <p className="text-xs text-[#94A3B8] leading-relaxed">
                {getPlainEnglishExplanation(selectedNode.data.category)}
              </p>
            </div>

            {/* Common Node Display Label */}
            <div className="space-y-1">
              <label className="block text-xs font-medium text-[#F1F5F9]">
                Block Name
              </label>
              <input
                type="text"
                value={selectedNode.data.label}
                onChange={(e) =>
                  onUpdateNodeData(selectedNode.id, { label: e.target.value })
                }
                className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs text-[#F1F5F9] focus:outline-none focus:border-[#FF6B2B]"
              />
            </div>

            {/* Dataset Specific Fields */}
            {selectedNode.data.category === "dataset" && (
              <>
                <div className="space-y-1">
                  <label className="block text-xs font-medium text-[#F1F5F9]">
                    Choose Study Dataset
                  </label>
                  <p className="text-[11px] text-[#94A3B8]">
                    Pick what topic of examples your AI should study.
                  </p>
                  <select
                    value={selectedNode.data.hfPath}
                    onChange={(e) =>
                      onUpdateNodeData(selectedNode.id, {
                        hfPath: e.target.value,
                        label: e.target.value.split("/").pop() || e.target.value,
                      })
                    }
                    className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono text-[#F1F5F9] focus:outline-none focus:border-[#FF6B2B]"
                  >
                    <option value="HuggingFaceH4/ultrachat_200k">
                      UltraChat 200k (Everyday Helpful Conversations)
                    </option>
                    <option value="teknium/OpenHermes-2.5">
                      OpenHermes 2.5 (Following Clear Instructions)
                    </option>
                    <option value="openai/gsm8k">
                      GSM8K (Step-by-Step Math Word Problems)
                    </option>
                    <option value="HuggingFaceH4/ultrafeedback_binarized">
                      UltraFeedback (Polite &amp; Safe Responses)
                    </option>
                    <option value="ise-uiuc/Magicoder-OSS-Instruct-75K">
                      Magicoder 75K (Writing Python &amp; Web Code)
                    </option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="block text-xs font-medium text-[#F1F5F9]">
                      Number of Examples
                    </label>
                    <p className="text-[10.5px] text-[#64748B]">
                      More examples = smarter, but takes longer
                    </p>
                    <input
                      type="number"
                      step={500}
                      min={500}
                      max={50000}
                      value={selectedNode.data.sampleCount ?? 4000}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          sampleCount: Number(e.target.value),
                        })
                      }
                      className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs font-medium text-[#F1F5F9]">
                      Mix Weight (0.1–1.0)
                    </label>
                    <p className="text-[10.5px] text-[#64748B]">
                      How much to focus on this dataset
                    </p>
                    <input
                      type="number"
                      step={0.1}
                      min={0.1}
                      max={1.0}
                      value={selectedNode.data.mixWeight ?? 0.5}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          mixWeight: Number(e.target.value),
                        })
                      }
                      className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                    />
                  </div>
                </div>
              </>
            )}

            {/* Base Model Specific Fields */}
            {selectedNode.data.category === "baseModel" && (
              <>
                <div className="space-y-1">
                  <label className="block text-xs font-medium text-[#F1F5F9]">
                    Starting AI Brain (Base Model)
                  </label>
                  <p className="text-[11px] text-[#94A3B8]">
                    Smaller models (135M) finish training fastest on laptops.
                  </p>
                  <select
                    value={selectedNode.data.modelId}
                    onChange={(e) => {
                      const val = e.target.value;
                      const vramMap: Record<string, number> = {
                        "HuggingFaceTB/SmolLM2-135M-Instruct": 0.85,
                        "HuggingFaceTB/SmolLM2-360M-Instruct": 1.25,
                        "Qwen/Qwen2.5-0.5B-Instruct": 1.45,
                        "meta-llama/Llama-3.2-1B-Instruct": 2.1,
                      };
                      onUpdateNodeData(selectedNode.id, {
                        modelId: val,
                        label: val.split("/").pop() || val,
                        baseVramGb: vramMap[val] || 1.2,
                      });
                    }}
                    className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono text-[#F1F5F9]"
                  >
                    <option value="HuggingFaceTB/SmolLM2-135M-Instruct">
                      SmolLM2-135M (Fastest · Great for Beginners)
                    </option>
                    <option value="HuggingFaceTB/SmolLM2-360M-Instruct">
                      SmolLM2-360M (Medium Size · Better Chat)
                    </option>
                    <option value="Qwen/Qwen2.5-0.5B-Instruct">
                      Qwen2.5-0.5B (Best for Math &amp; Logic)
                    </option>
                    <option value="meta-llama/Llama-3.2-1B-Instruct">
                      Llama-3.2-1B (Largest · Needs 6GB+ GPU)
                    </option>
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="block text-xs font-medium text-[#F1F5F9]">
                      Memory Compression
                    </label>
                    <select
                      value={selectedNode.data.quantization}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          quantization: e.target.value as VrikshaNodeData["quantization"],
                        })
                      }
                      className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono text-[#F1F5F9]"
                    >
                      <option value="4bit-nf4">4-bit Saver (Best for Laptops)</option>
                      <option value="8bit">8-bit Medium Compression</option>
                      <option value="bfloat16">Uncompressed (High Memory)</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs font-medium text-[#F1F5F9]">
                      Max Answer Length
                    </label>
                    <select
                      value={selectedNode.data.contextLength}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          contextLength: Number(e.target.value),
                        })
                      }
                      className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                    >
                      <option value={1024}>Short (1024 words/tokens)</option>
                      <option value={2048}>Standard (2048 tokens)</option>
                      <option value={4096}>Long (4096 tokens)</option>
                    </select>
                  </div>
                </div>
              </>
            )}

            {/* Trainer Specific Fields (SFT & GRPO) */}
            {selectedNode.data.category === "trainer" && (
              <>
                <div className="space-y-1.5">
                  <label className="block text-xs font-medium text-[#F1F5F9]">
                    How Should the AI Learn?
                  </label>
                  <div className="grid grid-cols-2 gap-1.5 p-1 bg-[#0B0D12] rounded border border-[#222938]">
                    <button
                      onClick={() =>
                        onUpdateNodeData(selectedNode.id, {
                          mode: "SFT",
                          label: "Supervised Fine-Tuning (SFT)",
                          beginnerHint:
                            "Teaches the AI by example using Oumi's automated trainer.",
                        })
                      }
                      className={`py-1.5 px-2 rounded text-xs font-medium transition-colors whitespace-nowrap ${
                        selectedNode.data.mode === "SFT"
                          ? "bg-[#FF6B2B] text-white"
                          : "text-[#94A3B8] hover:text-[#F1F5F9]"
                      }`}
                    >
                      By Example (SFT)
                    </button>
                    <button
                      onClick={() =>
                        onUpdateNodeData(selectedNode.id, {
                          mode: "GRPO",
                          label: "Reward-Based Trainer (GRPO)",
                          beginnerHint:
                            "Generates multiple answers and rewards the AI for correct reasoning.",
                        })
                      }
                      className={`py-1.5 px-2 rounded text-xs font-medium transition-colors whitespace-nowrap ${
                        selectedNode.data.mode === "GRPO"
                          ? "bg-[#8B5CF6] text-white"
                          : "text-[#94A3B8] hover:text-[#F1F5F9]"
                      }`}
                    >
                      By Reward (GRPO)
                    </button>
                  </div>
                  <p className="text-[11px] text-[#94A3B8]">
                    {selectedNode.data.mode === "GRPO"
                      ? "By Reward (GRPO): The AI tries several answers per question and gets rewarded when its math or reasoning is right."
                      : "By Example (SFT): Best for beginners! The AI directly studies question-and-answer pairs from your datasets."}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="block text-xs font-medium text-[#F1F5F9]">
                      Learning Capacity (LoRA)
                    </label>
                    <select
                      value={selectedNode.data.loraRank}
                      onChange={(e) => {
                        const r = Number(e.target.value);
                        onUpdateNodeData(selectedNode.id, {
                          loraRank: r,
                          loraAlpha: r * 2,
                        });
                      }}
                      className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                    >
                      <option value={8}>Light (r=8 · Fastest)</option>
                      <option value={16}>Balanced (r=16 · Recommended)</option>
                      <option value={32}>Deep (r=32 · Best for Math)</option>
                      <option value={64}>Max Capacity (r=64)</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs font-medium text-[#F1F5F9]">
                      Learning Speed
                    </label>
                    <select
                      value={selectedNode.data.learningRate}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          learningRate: e.target.value,
                        })
                      }
                      className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                    >
                      <option value="2e-4">Fast (2e-4 · Default SFT)</option>
                      <option value="1e-4">Steady (1e-4)</option>
                      <option value="5e-5">Careful (5e-5 · Default GRPO)</option>
                      <option value="1e-5">Gentle (1e-5)</option>
                    </select>
                  </div>
                </div>

                {selectedNode.data.mode === "GRPO" && (
                  <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-3">
                    <div className="text-xs font-medium text-[#F1F5F9]">
                      Reward Training Options (GRPO)
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <label className="block text-[11px] text-[#94A3B8]">
                          Tries per Question (G)
                        </label>
                        <select
                          value={selectedNode.data.grpoGroupSize}
                          onChange={(e) =>
                            onUpdateNodeData(selectedNode.id, {
                              grpoGroupSize: Number(e.target.value),
                            })
                          }
                          className="w-full px-2.5 py-1 rounded bg-[#161B26] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                        >
                          <option value={2}>2 Tries (Saves Memory)</option>
                          <option value={4}>4 Tries (Recommended)</option>
                          <option value={8}>8 Tries (Deep Practice)</option>
                        </select>
                      </div>
                      <div className="space-y-1">
                        <label className="block text-[11px] text-[#94A3B8]">
                          Stay Close to Original (β)
                        </label>
                        <input
                          type="number"
                          step={0.01}
                          min={0.01}
                          max={0.2}
                          value={selectedNode.data.klPenaltyBeta}
                          onChange={(e) =>
                            onUpdateNodeData(selectedNode.id, {
                              klPenaltyBeta: Number(e.target.value),
                            })
                          }
                          className="w-full px-2.5 py-1 rounded bg-[#161B26] border border-[#222938] text-xs font-mono tabular-nums text-[#F1F5F9]"
                        />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <label className="block text-[11px] text-[#94A3B8]">
                        What Gets Rewarded?
                      </label>
                      <select
                        value={selectedNode.data.rewardPreset}
                        onChange={(e) =>
                          onUpdateNodeData(selectedNode.id, {
                            rewardPreset: e.target.value,
                          })
                        }
                        className="w-full px-2.5 py-1.5 rounded bg-[#161B26] border border-[#222938] text-xs font-mono text-[#F1F5F9]"
                      >
                        <option value="math_verifiable_reward">
                          Correct Math Answer (GSM8K Verifier)
                        </option>
                        <option value="reasoning_format_reward">
                          Showing Step-by-Step Work (&lt;think&gt; tags)
                        </option>
                        <option value="safety_constitutional_reward">
                          Safe &amp; Helpful Tone
                        </option>
                      </select>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Hardware Optimizer Fields */}
            {selectedNode.data.category === "hardware" && (
              <>
                <div className="space-y-1">
                  <label className="block text-xs font-medium text-[#F1F5F9]">
                    Your Computer&apos;s Graphics Card (GPU)
                  </label>
                  <select
                    value={selectedNode.data.gpuProfile}
                    onChange={(e) => {
                      const val = e.target.value as VrikshaNodeData["gpuProfile"];
                      const limit = val?.includes("4GB")
                        ? 4
                        : val?.includes("8GB")
                        ? 8
                        : 12;
                      onUpdateNodeData(selectedNode.id, {
                        gpuProfile: val,
                        vramLimitGb: limit,
                      });
                    }}
                    className="w-full px-3 py-1.5 rounded bg-[#0B0D12] border border-[#222938] text-xs font-mono text-[#F1F5F9]"
                  >
                    <option value="RTX 3050 Laptop (4GB VRAM)">
                      Entry Laptop GPU (4GB Memory · RTX 3050)
                    </option>
                    <option value="RTX 4060 Laptop (8GB VRAM)">
                      Standard Gaming Laptop (8GB Memory · RTX 4060)
                    </option>
                    <option value="RTX 4070 Desktop (12GB VRAM)">
                      Desktop GPU (12GB Memory · RTX 4070)
                    </option>
                  </select>
                </div>

                <div className="space-y-2 pt-1">
                  <label className="flex items-center justify-between text-xs text-[#F1F5F9] cursor-pointer">
                    <span>Anti-Crash Memory Saver (Gradient Checkpointing)</span>
                    <input
                      type="checkbox"
                      checked={Boolean(selectedNode.data.gradientCheckpointing)}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          gradientCheckpointing: e.target.checked,
                        })
                      }
                      className="accent-[#FF6B2B]"
                    />
                  </label>
                  <label className="flex items-center justify-between text-xs text-[#F1F5F9] cursor-pointer">
                    <span>Windows WSL2 Laptop Optimization</span>
                    <input
                      type="checkbox"
                      checked={Boolean(selectedNode.data.wsl2Mode)}
                      onChange={(e) =>
                        onUpdateNodeData(selectedNode.id, {
                          wsl2Mode: e.target.checked,
                        })
                      }
                      className="accent-[#FF6B2B]"
                    />
                  </label>
                </div>
              </>
            )}

            {/* Endpoint Node Quick Action */}
            {selectedNode.data.category === "endpoint" && (
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-2.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#94A3B8]">Your Trained AI ID</span>
                  <span className="font-mono text-[#10B981] font-semibold">
                    {selectedNode.data.clusterId}
                  </span>
                </div>
                <button
                  onClick={() =>
                    onOpenClusterInference(
                      String(selectedNode.data.clusterId || "cluster-8492-sft")
                    )
                  }
                  className="w-full py-2 px-3 rounded bg-[#10B981] hover:bg-[#059669] text-xs font-semibold text-[#0B0D12] flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Chat with {selectedNode.data.clusterId} Now
                </button>
              </div>
            )}

            {/* Live Oumi Dataset / GRPO Rollout Synthesizer */}
            {(selectedNode.data.category === "dataset" ||
              selectedNode.data.category === "trainer") && (
              <div className="pt-2 border-t border-[#222938] space-y-2.5">
                <button
                  onClick={handlePreviewRollout}
                  disabled={rolloutLoading}
                  className="w-full py-2 px-3 rounded bg-[#1A1F2B] hover:bg-[#222938] border border-[#222938] text-xs text-[#F1F5F9] flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#FF6B2B]" />
                  {rolloutLoading
                    ? "Generating Sample Q&A..."
                    : "See Sample Questions & Answers"}
                </button>

                {rolloutSamples.length > 0 && (
                  <div className="space-y-2">
                    {rolloutSamples.map((s, idx) => (
                      <div
                        key={idx}
                        className="p-2.5 rounded bg-[#0B0D12] border border-[#222938] text-[11px] space-y-1"
                      >
                        <div className="text-[#94A3B8] font-mono">
                          Example #{idx + 1} · Quality Score{" "}
                          <span className="text-[#10B981]">
                            {Math.round(s.rewardScore * 100)}%
                          </span>
                        </div>
                        <div className="text-[#F1F5F9] font-medium line-clamp-2">
                          Question: {s.prompt}
                        </div>
                        <div className="text-[#94A3B8] font-mono line-clamp-3">
                          AI Answer: {s.completion}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Consumer GPU VRAM Budget Estimator Card */}
        <div className="p-3.5 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-[#F1F5F9]">
              Will This Fit on My Laptop?
            </span>
            <span className="font-mono tabular-nums text-[#F1F5F9]">
              {vramBreakdown.totalGb.toFixed(1)} / {vramBreakdown.limitGb.toFixed(1)} GB
            </span>
          </div>

          <div className="w-full h-2 rounded-full bg-[#161B26] overflow-hidden">
            <div
              className={`h-full transition-all duration-200 ${
                vramBreakdown.totalGb > vramBreakdown.limitGb
                  ? "bg-[#EF4444]"
                  : vramRatio > 80
                  ? "bg-[#F59E0B]"
                  : "bg-[#10B981]"
              }`}
              style={{ width: `${vramRatio}%` }}
            />
          </div>

          <div className="text-[11.5px] text-[#94A3B8] leading-snug">
            {vramStatusText}
          </div>
        </div>
      </div>
    </aside>
  );
}
