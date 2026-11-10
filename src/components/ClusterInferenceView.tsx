import { useState } from "react";
import {
  Server,
  Send,
  CheckCircle2,
  Cpu,
  Layers,
  RotateCcw,
  ArrowLeft,
} from "lucide-react";
import type { ClusterRecord } from "../types";

interface ClusterInferenceViewProps {
  clusters: ClusterRecord[];
  selectedClusterId: string;
  onSelectClusterId: (id: string) => void;
  onRefreshClusters: () => void;
  onBackToStudio?: () => void;
}

interface InferenceResult {
  clusterId: string;
  baseModel: string;
  mode: string;
  adapterPath: string;
  loraResponse: string;
  baseModelResponse: string | null;
  telemetry: {
    ttftMs: number;
    totalLatencyMs: number;
    tokensGenerated: number;
    tokensPerSec: number;
    pagedAttentionKvCachePct: number;
    rewardScore: number;
    activeAdapterWeights: string;
  };
}

const SAMPLE_PROMPTS = [
  {
    label: "Math Word Problem",
    text: "A train leaves Mumbai at 60 km/h and another leaves Pune (150 km away) towards Mumbai at 90 km/h. In how many minutes do they meet? Show your steps clearly.",
  },
  {
    label: "Python Coding Task",
    text: "Write a clean Python function to find the top 3 most frequent words in a sentence and explain how it works.",
  },
  {
    label: "Explain AI Concept Simply",
    text: "Explain how fine-tuning a small AI model on a laptop works to a high school student in 3 bullet points.",
  },
];

export function ClusterInferenceView({
  clusters,
  selectedClusterId,
  onSelectClusterId,
  onRefreshClusters,
  onBackToStudio,
}: ClusterInferenceViewProps) {
  const [customClusterInput, setCustomClusterInput] = useState(selectedClusterId);
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0].text);
  const [compareWithBase, setCompareWithBase] = useState(true);
  const [loading, setLoading] = useState(false);
  const [mountStatus, setMountStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);

  const activeCluster =
    clusters.find((c) => c.clusterId === selectedClusterId) || clusters[0];

  const handleMountAdapter = async (idToMount: string) => {
    setError(null);
    try {
      const resp = await fetch(`/api/clusters/${encodeURIComponent(idToMount)}/mount`, {
        method: "POST",
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || "Could not load that ClusterID");
        return;
      }
      onSelectClusterId(idToMount);
      setCustomClusterInput(idToMount);
      setMountStatus(
        `Loaded trained AI '${idToMount}' into the vLLM engine — ready to chat!`
      );
      onRefreshClusters();
    } catch {
      setError("Network error while loading trained model.");
    }
  };

  const handleRunInference = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const targetId = customClusterInput.trim() || selectedClusterId;
      const resp = await fetch("/api/inference/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clusterId: targetId,
          prompt,
          compareWithBase,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || "Inference request failed");
      } else {
        setResult(data);
      }
    } catch {
      setError("Failed to reach the ClusterID chat gateway.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#0B0D12] p-6">
      <div className="max-w-[1320px] mx-auto space-y-6">
        {/* Beginner-Friendly Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-[#222938]">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              {onBackToStudio && (
                <button
                  onClick={onBackToStudio}
                  className="px-2.5 py-1 rounded bg-[#161B26] hover:bg-[#1A1F2B] border border-[#222938] text-xs text-[#94A3B8] hover:text-[#F1F5F9] flex items-center gap-1 transition-colors"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  Back to Pipeline Canvas
                </button>
              )}
              <span className="text-xs text-[#10B981] font-medium">
                Step 3 of 3 · Test &amp; Compare Your Trained AI
              </span>
            </div>
            <h1 className="text-2xl font-bold text-[#F1F5F9] font-display">
              Chat with Your Trained AI (ClusterID Playground)
            </h1>
            <p className="text-xs text-[#94A3B8] max-w-2xl">
              Every training run gives your AI a unique <span className="text-[#F1F5F9] font-mono">ClusterID</span>. Pick one of your trained models on the left and ask it a question to see how much smarter it is compared to the untrained model!
            </p>
          </div>

          {/* Direct ClusterID Input & Mount Bar */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="flex items-center bg-[#12161F] border border-[#222938] rounded-lg px-3 py-1.5">
              <span className="text-xs text-[#64748B] mr-2 whitespace-nowrap">
                Trained AI ID:
              </span>
              <input
                type="text"
                value={customClusterInput}
                onChange={(e) => setCustomClusterInput(e.target.value)}
                placeholder="cluster-8492-sft"
                className="bg-transparent text-xs font-mono text-[#10B981] focus:outline-none w-[145px]"
              />
            </div>
            <button
              onClick={() => handleMountAdapter(customClusterInput)}
              className="px-3.5 py-2 rounded-lg bg-[#1A1F2B] hover:bg-[#222938] border border-[#222938] text-xs font-medium text-[#F1F5F9] transition-colors whitespace-nowrap shrink-0"
            >
              Load AI Model
            </button>
          </div>
        </div>

        {mountStatus && (
          <div className="p-3 rounded-lg bg-[#12161F] border border-[#10B981]/40 text-xs text-[#10B981] flex items-center justify-between">
            <span>{mountStatus}</span>
            <button
              onClick={() => setMountStatus(null)}
              className="text-[#94A3B8] hover:text-white text-xs"
            >
              Close
            </button>
          </div>
        )}

        {error && (
          <div className="p-3 rounded-lg bg-[#12161F] border border-[#EF4444]/50 text-xs text-[#EF4444]">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Trained AI Models List (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            <div className="p-4 rounded-lg bg-[#12161F] border border-[#222938] space-y-3">
              <div>
                <h2 className="text-sm font-semibold text-[#F1F5F9]">
                  1. Choose a Trained AI Model
                </h2>
                <p className="text-xs text-[#94A3B8] mt-0.5">
                  Click any model below to load its learned skills.
                </p>
              </div>

              <div className="space-y-2">
                {clusters.map((c) => {
                  const isSelected = c.clusterId === selectedClusterId;
                  return (
                    <button
                      key={c.clusterId}
                      onClick={() => {
                        onSelectClusterId(c.clusterId);
                        setCustomClusterInput(c.clusterId);
                      }}
                      className={`w-full text-left p-3 rounded-lg border transition-colors ${
                        isSelected
                          ? "bg-[#161B26] border-[#FF6B2B]"
                          : "bg-[#0B0D12] border-[#222938] hover:border-[#334155]"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-xs font-semibold text-[#10B981]">
                          {c.clusterId}
                        </span>
                        <span className="text-[11px] text-[#94A3B8]">
                          {c.mode === "GRPO"
                            ? "Trained with Rewards (GRPO)"
                            : "Trained by Example (SFT)"}
                        </span>
                      </div>
                      <div className="text-xs font-medium text-[#F1F5F9] mt-1 truncate">
                        {c.pipelineName}
                      </div>
                      <div className="text-[11px] text-[#64748B] font-mono tabular-nums mt-1">
                        Accuracy: {(c.tokenAccuracy * 100).toFixed(1)}% · Mistake Rate:{" "}
                        {c.finalLoss.toFixed(2)}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {activeCluster && (
              <div className="p-4 rounded-lg bg-[#12161F] border border-[#222938] space-y-3 text-xs">
                <div className="flex items-center gap-2 text-[#F1F5F9] font-semibold">
                  <Layers className="w-4 h-4 text-[#FF6B2B]" />
                  <span>About This Trained AI</span>
                </div>
                <div className="space-y-2 text-[11.5px]">
                  <div className="flex justify-between">
                    <span className="text-[#64748B]">Base Brain Used</span>
                    <span className="font-mono text-[#A855F7]">
                      {activeCluster.baseModel.split("/").pop()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#64748B]">Training Style</span>
                    <span className="text-[#F1F5F9]">
                      {activeCluster.mode === "GRPO"
                        ? "Step-by-Step Reward (GRPO)"
                        : "Supervised Examples (SFT)"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#64748B]">Laptop Memory Used</span>
                    <span className="font-mono text-[#10B981]">
                      {activeCluster.vramPeakGb.toFixed(1)} GB VRAM
                    </span>
                  </div>
                  <div className="pt-2 border-t border-[#222938] text-[#94A3B8]">
                    <span className="text-[#F1F5F9] font-medium">Studied Datasets:</span>{" "}
                    {activeCluster.datasets.map((d) => d.split("/").pop()).join(" + ")}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Prompt Console & Side-by-Side Comparison (8 cols) */}
          <div className="lg:col-span-8 space-y-4">
            <div className="p-4 rounded-lg bg-[#12161F] border border-[#222938] space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-sm font-semibold text-[#F1F5F9]">
                    2. Ask Your Trained AI a Question
                  </span>
                  <p className="text-xs text-[#94A3B8]">
                    Click a sample question below or type your own.
                  </p>
                </div>
                <label className="flex items-center gap-1.5 text-xs text-[#F1F5F9] cursor-pointer bg-[#0B0D12] px-3 py-1.5 rounded border border-[#222938]">
                  <input
                    type="checkbox"
                    checked={compareWithBase}
                    onChange={(e) => setCompareWithBase(e.target.checked)}
                    className="accent-[#FF6B2B]"
                  />
                  <span>Compare side-by-side with Untrained AI</span>
                </label>
              </div>

              {/* Preset Test Prompts */}
              <div className="flex flex-wrap gap-2">
                {SAMPLE_PROMPTS.map((sp, i) => (
                  <button
                    key={i}
                    onClick={() => setPrompt(sp.text)}
                    className="px-3 py-1.5 rounded bg-[#0B0D12] hover:bg-[#1A1F2B] border border-[#222938] text-xs text-[#F1F5F9] transition-colors"
                  >
                    Try: {sp.label}
                  </button>
                ))}
              </div>

              <textarea
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Type any question or math problem here..."
                className="w-full p-3 rounded-lg bg-[#0B0D12] border border-[#222938] text-sm text-[#F1F5F9] focus:outline-none focus:border-[#FF6B2B] resize-none"
              />

              <div className="flex items-center justify-between pt-1">
                <div className="text-xs text-[#94A3B8]">
                  Active AI: <span className="font-mono text-[#10B981]">{customClusterInput}</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setResult(null)}
                    className="px-3 py-2 rounded-lg bg-[#1A1F2B] hover:bg-[#222938] text-xs text-[#94A3B8] flex items-center gap-1.5 whitespace-nowrap"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Reset
                  </button>
                  <button
                    onClick={handleRunInference}
                    disabled={loading}
                    className="px-5 py-2 rounded-lg bg-[#FF6B2B] hover:bg-[#e55a1f] text-xs font-semibold text-white flex items-center gap-1.5 transition-colors whitespace-nowrap disabled:opacity-50"
                  >
                    <Send className="w-3.5 h-3.5" />
                    {loading
                      ? "Generating Answer..."
                      : "Ask Trained AI"}
                  </button>
                </div>
              </div>
            </div>

            {/* Output Arena */}
            {result ? (
              <div className="space-y-4">
                {/* Speed & Quality Strip */}
                <div className="p-3.5 rounded-lg bg-[#12161F] border border-[#222938] flex flex-wrap items-center justify-between gap-4 text-xs font-mono tabular-nums">
                  <div>
                    <span className="text-[#94A3B8]">First Word Delay:</span>{" "}
                    <span className="text-[#10B981]">{result.telemetry.ttftMs} ms</span>
                  </div>
                  <div>
                    <span className="text-[#94A3B8]">Reading Speed:</span>{" "}
                    <span className="text-[#F1F5F9]">
                      {result.telemetry.tokensPerSec} words/sec
                    </span>
                  </div>
                  <div>
                    <span className="text-[#94A3B8]">Answer Quality Score:</span>{" "}
                    <span className="text-[#10B981]">
                      {(result.telemetry.rewardScore * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Comparison Columns */}
                <div
                  className={`grid gap-4 ${
                    compareWithBase && result.baseModelResponse
                      ? "grid-cols-1 md:grid-cols-2"
                      : "grid-cols-1"
                  }`}
                >
                  {/* Fine-Tuned ClusterID LoRA Output */}
                  <div className="p-4 rounded-lg bg-[#12161F] border border-[#10B981]/50 space-y-3">
                    <div className="flex items-center justify-between pb-2.5 border-b border-[#222938]">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
                        <div>
                          <div className="text-xs font-semibold text-[#F1F5F9]">
                            Your Newly Trained AI ({result.clusterId})
                          </div>
                          <div className="text-[11px] text-[#10B981]">
                            After Oumi {result.mode} Training · Structured &amp; Accurate
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="text-xs leading-relaxed text-[#F1F5F9] whitespace-pre-wrap font-mono select-text">
                      {result.loraResponse}
                    </div>
                  </div>

                  {/* Base Unaligned Model Output */}
                  {compareWithBase && result.baseModelResponse && (
                    <div className="p-4 rounded-lg bg-[#12161F] border border-[#222938] space-y-3">
                      <div className="flex items-center justify-between pb-2.5 border-b border-[#222938]">
                        <div className="flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-[#94A3B8]" />
                          <div>
                            <div className="text-xs font-semibold text-[#94A3B8]">
                              Untrained Starter AI (Before Training)
                            </div>
                            <div className="text-[11px] text-[#64748B]">
                              Raw {result.baseModel.split("/").pop()} without your training data
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="text-xs leading-relaxed text-[#94A3B8] whitespace-pre-wrap font-mono select-text">
                        {result.baseModelResponse}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-10 rounded-lg bg-[#12161F] border border-[#222938] text-center space-y-2">
                <Server className="w-7 h-7 text-[#FF6B2B] mx-auto" />
                <div className="text-sm font-semibold text-[#F1F5F9]">
                  Ready to Test Your Trained AI!
                </div>
                <p className="text-xs text-[#94A3B8] max-w-md mx-auto leading-relaxed">
                  Click the orange <span className="text-[#F1F5F9] font-semibold">Ask Trained AI</span> button above to see how your fine-tuned model answers compared to the untrained starter model.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
