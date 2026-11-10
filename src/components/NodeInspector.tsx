import { useState, useEffect } from "react";
import {
  Download,
  Copy,
  Check,
  Trash2,
  Play,
  FileCode2,
  Sliders,
  Plus,
} from "lucide-react";
import type { VrikshaFlowNode, VrikshaNodeData } from "../types";

interface CetSample {
  id: string;
  topic: string;
  question: string;
  think: string;
  answer: string;
}

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
  onRunTraining: () => void;
  isTraining: boolean;
}

export function NodeInspector({
  selectedNode,
  onUpdateNodeData,
  onDeleteNode,
  compiledYaml,
  vramBreakdown,
  onOpenClusterInference,
  onRunTraining,
  isTraining,
}: NodeInspectorProps) {
  const [showYaml, setShowYaml] = useState(false);
  const [copied, setCopied] = useState(false);
  const [cetSamples, setCetSamples] = useState<CetSample[]>([]);
  const [newQuestion, setNewQuestion] = useState("");
  const [newAnswer, setNewAnswer] = useState("");
  const [addedNotice, setAddedNotice] = useState(false);

  useEffect(() => {
    fetch("/api/cet-dataset")
      .then((r) => r.json())
      .then((d) => {
        if (Array.isArray(d.samples)) setCetSamples(d.samples);
      })
      .catch(() => {});
  }, []);

  const handleAddCustomCetQuestion = async () => {
    if (!newQuestion.trim() || !newAnswer.trim()) return;
    try {
      const resp = await fetch("/api/cet-dataset/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: "Custom MAH MCA CET",
          question: newQuestion.trim(),
          think: "User-added CET shortcut example",
          answer: newAnswer.trim(),
        }),
      });
      const data = await resp.json();
      if (Array.isArray(data.samples)) {
        setCetSamples(data.samples);
        setNewQuestion("");
        setNewAnswer("");
        setAddedNotice(true);
        setTimeout(() => setAddedNotice(false), 2500);
      }
    } catch {
      // ignore
    }
  };

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

  const getSimpleExplanation = (category: VrikshaNodeData["category"]) => {
    switch (category) {
      case "dataset":
        return "This is the study material (500 MAH MCA CET Math & Reasoning questions + shortcut tricks) your AI reads.";
      case "baseModel":
        return "Choose which starting AI model to train. Qwen-0.5B is fast and great at math.";
      case "trainer":
        return "Learn by Reward (GRPO) teaches the AI to write fast CET exam shortcuts and pick the right MCQ option.";
      case "hardware":
        return "Keeps memory usage around 2.1 GB so training runs smoothly on any laptop.";
      case "endpoint":
        return "Your finished MAH MCA CET AI! Click the green button below to solve CET questions.";
    }
  };

  return (
    <aside className="w-[330px] shrink-0 bg-[#12161F] border-l border-[#222938] flex flex-col h-full select-none">
      {/* Clean 2-Tab Switcher */}
      <div className="p-3 border-b border-[#222938] flex items-center gap-1 bg-[#0B0D12]/50">
        <button
          onClick={() => setShowYaml(false)}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 rounded text-xs font-medium transition-colors whitespace-nowrap ${
            !showYaml
              ? "bg-[#1A1F2B] text-[#F1F5F9] border border-[#222938]"
              : "text-[#94A3B8] hover:text-[#F1F5F9]"
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          Block Settings
        </button>
        <button
          onClick={() => setShowYaml(true)}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 rounded text-xs font-medium transition-colors whitespace-nowrap ${
            showYaml
              ? "bg-[#1A1F2B] text-[#F1F5F9] border border-[#222938]"
              : "text-[#94A3B8] hover:text-[#F1F5F9]"
          }`}
        >
          <FileCode2 className="w-3.5 h-3.5" />
          Oumi YAML
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        {showYaml ? (
          <div className="space-y-3">
            <p className="text-xs text-[#94A3B8]">
              Auto-generated Oumi config for your MAH MCA CET small-scale run:
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopyYaml}
                className="flex-1 py-1.5 px-2.5 rounded bg-[#1A1F2B] hover:bg-[#222938] border border-[#222938] text-xs text-[#F1F5F9] flex items-center justify-center gap-1"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-[#10B981]" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                {copied ? "Copied" : "Copy"}
              </button>
              <button
                onClick={handleDownloadYaml}
                className="flex-1 py-1.5 px-2.5 rounded bg-[#FF6B2B] hover:bg-[#e55a1f] text-xs font-medium text-white flex items-center justify-center gap-1"
              >
                <Download className="w-3.5 h-3.5" />
                Download
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] text-[11px] leading-relaxed font-mono text-[#F1F5F9] overflow-x-auto select-text">
              {compiledYaml}
            </pre>
          </div>
        ) : !selectedNode ? (
          <div className="space-y-4">
            <h3 className="text-base font-bold text-[#F1F5F9] font-display">
              MAH MCA CET Small-Scale Setup Ready
            </h3>
            <div className="space-y-2.5 text-xs text-[#94A3B8] leading-relaxed">
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938]">
                <span className="text-[#38BDF8] font-semibold block mb-0.5">
                  Step 1: Inspect CET Questions
                </span>
                Click the first block (<strong>MAH MCA CET Math &amp; LR</strong>) to view the CET questions or add your own.
              </div>
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938]">
                <span className="text-[#FF6B2B] font-semibold block mb-0.5">
                  Step 2: Run Small-Scale Training
                </span>
                Click the orange button below to train on the 500-question CET batch.
              </div>
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938]">
                <span className="text-[#10B981] font-semibold block mb-0.5">
                  Step 3: Solve CET Questions
                </span>
                Test your trained AI on Probability, P&amp;C, Series, and Coding-Decoding!
              </div>
            </div>
            <button
              onClick={onRunTraining}
              disabled={isTraining}
              className="w-full py-2.5 px-4 rounded-lg bg-[#FF6B2B] hover:bg-[#e55a1f] text-xs font-semibold text-white flex items-center justify-center gap-2 transition-colors"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {isTraining ? "Training CET Model..." : "Train MAH MCA CET AI Now"}
            </button>
          </div>
        ) : (
          <div className="space-y-5">
            {/* Block Header */}
            <div className="flex items-start justify-between gap-2 pb-3 border-b border-[#222938]">
              <div>
                <h3 className="text-base font-bold text-[#F1F5F9] font-display">
                  {selectedNode.data.label}
                </h3>
                <p className="text-xs text-[#94A3B8] mt-1 leading-relaxed">
                  {getSimpleExplanation(selectedNode.data.category)}
                </p>
              </div>
              <button
                onClick={() => onDeleteNode(selectedNode.id)}
                title="Delete block"
                className="p-1.5 rounded text-[#64748B] hover:text-[#EF4444] hover:bg-[#1A1F2B] transition-colors shrink-0"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {/* Dataset Settings + Live CET Question Viewer & Adder */}
            {selectedNode.data.category === "dataset" && (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-[#F1F5F9]">
                    1. Choose Dataset
                  </label>
                  <select
                    value={selectedNode.data.hfPath}
                    onChange={(e) => {
                      const nameMap: Record<string, string> = {
                        "n2n-back/data/mah_mca_cet_small.jsonl": "MAH MCA CET Math & LR",
                        "HuggingFaceH4/ultrachat_200k": "Conversation Examples",
                        "openai/gsm8k": "GSM8K Math Problems",
                      };
                      onUpdateNodeData(selectedNode.id, {
                        hfPath: e.target.value,
                        label: nameMap[e.target.value] || "Custom Dataset",
                      });
                    }}
                    className="w-full px-3 py-2 rounded-lg bg-[#0B0D12] border border-[#222938] text-xs text-[#F1F5F9]"
                  >
                    <option value="n2n-back/data/mah_mca_cet_small.jsonl">
                      MAH MCA CET Math &amp; Reasoning (Small Scale)
                    </option>
                    <option value="openai/gsm8k">
                      General Math Word Problems (GSM8K)
                    </option>
                    <option value="HuggingFaceH4/ultrachat_200k">
                      Everyday Helpful Conversations (UltraChat)
                    </option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-[#F1F5F9]">
                    2. Small-Scale Batch Size
                  </label>
                  <select
                    value={selectedNode.data.sampleCount ?? 500}
                    onChange={(e) =>
                      onUpdateNodeData(selectedNode.id, {
                        sampleCount: Number(e.target.value),
                      })
                    }
                    className="w-full px-3 py-2 rounded-lg bg-[#0B0D12] border border-[#222938] text-xs text-[#F1F5F9]"
                  >
                    <option value={100}>100 CET Questions (Ultra-Fast Test)</option>
                    <option value={500}>500 CET Questions (Recommended Small Scale)</option>
                    <option value={2000}>2,000 CET Questions (Full Topic Mix)</option>
                  </select>
                </div>

                {/* Live Preview of MAH MCA CET Questions in this Dataset */}
                <div className="pt-2 border-t border-[#222938] space-y-2">
                  <div className="text-xs font-semibold text-[#F1F5F9]">
                    Sample CET Questions Inside This Batch:
                  </div>
                  <div className="space-y-2 max-h-[190px] overflow-y-auto pr-1">
                    {cetSamples.slice(0, 4).map((s) => (
                      <div
                        key={s.id}
                        className="p-2.5 rounded-lg bg-[#0B0D12] border border-[#222938] text-[11px] space-y-1"
                      >
                        <div className="text-[#38BDF8] font-semibold">{s.topic}</div>
                        <div className="text-[#F1F5F9] line-clamp-2">{s.question}</div>
                        <div className="text-[#10B981] font-mono">{s.answer}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Add Your Own CET Question */}
                <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-2">
                  <div className="text-xs font-semibold text-[#F1F5F9]">
                    + Add Your Own CET Question to Train On
                  </div>
                  <input
                    type="text"
                    value={newQuestion}
                    onChange={(e) => setNewQuestion(e.target.value)}
                    placeholder="e.g. Find 10th term of AP: 2, 7, 12..."
                    className="w-full px-2.5 py-1.5 rounded bg-[#161B26] border border-[#222938] text-xs text-[#F1F5F9]"
                  />
                  <input
                    type="text"
                    value={newAnswer}
                    onChange={(e) => setNewAnswer(e.target.value)}
                    placeholder="Shortcut & Answer: a + 9d = 2 + 45 = 47"
                    className="w-full px-2.5 py-1.5 rounded bg-[#161B26] border border-[#222938] text-xs text-[#F1F5F9]"
                  />
                  <button
                    onClick={handleAddCustomCetQuestion}
                    className="w-full py-1.5 px-3 rounded bg-[#1A1F2B] hover:bg-[#222938] border border-[#222938] text-xs font-medium text-[#F1F5F9] flex items-center justify-center gap-1"
                  >
                    <Plus className="w-3.5 h-3.5 text-[#FF6B2B]" />
                    {addedNotice ? "Added to Training Batch!" : "Add Question to Batch"}
                  </button>
                </div>
              </div>
            )}

            {/* Base Model Simple Settings */}
            {selectedNode.data.category === "baseModel" && (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-[#F1F5F9]">
                    1. Choose Starting AI Brain
                  </label>
                  <select
                    value={selectedNode.data.modelId}
                    onChange={(e) => {
                      const val = e.target.value;
                      const labelMap: Record<string, string> = {
                        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen-0.5B Starter AI",
                        "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2 Starter AI",
                        "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2 Medium AI",
                      };
                      onUpdateNodeData(selectedNode.id, {
                        modelId: val,
                        label: labelMap[val] || "Starter AI",
                      });
                    }}
                    className="w-full px-3 py-2 rounded-lg bg-[#0B0D12] border border-[#222938] text-xs text-[#F1F5F9]"
                  >
                    <option value="Qwen/Qwen2.5-0.5B-Instruct">
                      Qwen-0.5B (Best for MAH MCA CET Math &amp; Logic)
                    </option>
                    <option value="HuggingFaceTB/SmolLM2-135M-Instruct">
                      SmolLM2-135M (Smallest &amp; Fastest)
                    </option>
                    <option value="HuggingFaceTB/SmolLM2-360M-Instruct">
                      SmolLM2-360M (Balanced)
                    </option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-[#F1F5F9]">
                    2. Laptop Memory Saver
                  </label>
                  <select
                    value={selectedNode.data.quantization}
                    onChange={(e) =>
                      onUpdateNodeData(selectedNode.id, {
                        quantization: e.target.value as VrikshaNodeData["quantization"],
                      })
                    }
                    className="w-full px-3 py-2 rounded-lg bg-[#0B0D12] border border-[#222938] text-xs text-[#F1F5F9]"
                  >
                    <option value="4bit-nf4">ON — 4-bit Memory Saver (Recommended)</option>
                    <option value="8bit">Medium — 8-bit Mode</option>
                  </select>
                </div>
              </div>
            )}

            {/* Trainer Simple Settings */}
            {selectedNode.data.category === "trainer" && (
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-[#F1F5F9]">
                    1. Learning Method
                  </label>
                  <div className="grid grid-cols-1 gap-2">
                    <button
                      onClick={() =>
                        onUpdateNodeData(selectedNode.id, {
                          mode: "GRPO",
                          label: "Learn by Reward (GRPO)",
                        })
                      }
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        selectedNode.data.mode === "GRPO"
                          ? "bg-[#161B26] border-[#8B5CF6]"
                          : "bg-[#0B0D12] border-[#222938]"
                      }`}
                    >
                      <div className="text-xs font-semibold text-[#F1F5F9]">
                        Learn by Reward (GRPO · Best for CET Math)
                      </div>
                      <div className="text-[11px] text-[#94A3B8] mt-0.5">
                        Rewards the AI when it uses fast shortcut formulas and gets the right MCQ answer.
                      </div>
                    </button>

                    <button
                      onClick={() =>
                        onUpdateNodeData(selectedNode.id, {
                          mode: "SFT",
                          label: "Learn by Example (SFT)",
                        })
                      }
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        selectedNode.data.mode === "SFT"
                          ? "bg-[#161B26] border-[#FF6B2B]"
                          : "bg-[#0B0D12] border-[#222938]"
                      }`}
                    >
                      <div className="text-xs font-semibold text-[#F1F5F9]">
                        Learn by Example (SFT · Standard)
                      </div>
                      <div className="text-[11px] text-[#94A3B8] mt-0.5">
                        Directly copies example answers without reward verification.
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Endpoint Simple Action */}
            {selectedNode.data.category === "endpoint" && (
              <div className="space-y-3">
                <div className="p-3.5 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-1">
                  <div className="text-xs text-[#94A3B8]">Trained CET Model ID</div>
                  <div className="text-sm font-mono font-bold text-[#10B981]">
                    {selectedNode.data.clusterId}
                  </div>
                </div>
                <button
                  onClick={() =>
                    onOpenClusterInference(
                      String(selectedNode.data.clusterId || "cluster-cet-2026")
                    )
                  }
                  className="w-full py-2.5 px-4 rounded-lg bg-[#10B981] hover:bg-[#059669] text-xs font-semibold text-[#0B0D12] flex items-center justify-center gap-1.5 transition-colors"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Test MAH MCA CET Questions Now
                </button>
              </div>
            )}
          </div>
        )}

        {/* Simple 1-Line Laptop Memory Check at Bottom */}
        <div className="p-3.5 rounded-lg bg-[#0B0D12] border border-[#222938] space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#94A3B8]">Small-Scale Memory Needed:</span>
            <span className="font-mono font-semibold text-[#10B981]">
              {vramBreakdown.totalGb.toFixed(1)} GB (Safe on Laptop)
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
