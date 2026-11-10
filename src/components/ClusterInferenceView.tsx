import { useState } from "react";
import {
  Send,
  CheckCircle2,
  Cpu,
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
}

const SAMPLE_PROMPTS = [
  {
    label: "1. CET Set Theory (Venn Diagram)",
    text: "In a group of 100 MAH MCA CET aspirants, 65 solved Mathematics, 55 solved Logical Reasoning, and 15 solved neither. How many aspirants solved ONLY Mathematics? Solve with a fast CET shortcut.",
  },
  {
    label: "2. CET Permutations & Probability",
    text: "How many 4-digit numbers divisible by 5 can be formed using the digits {0, 1, 2, 3, 4, 5} without repetition?",
  },
  {
    label: "3. CET Coding & Number Series",
    text: "In a certain code, COMPUTER is written as RFUVQNPC. How is MEDICINE written in that code? Also find the missing number: 3, 10, 29, 66, 127, ?",
  },
  {
    label: "4. CET Coordinate Geometry",
    text: "Find the distance between the parallel lines 3x - 4y + 9 = 0 and 6x - 8y - 15 = 0.",
  },
];

export function ClusterInferenceView({
  clusters,
  selectedClusterId,
  onSelectClusterId,
  onBackToStudio,
}: ClusterInferenceViewProps) {
  const [prompt, setPrompt] = useState(SAMPLE_PROMPTS[0].text);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);

  const handleRunInference = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/inference/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clusterId: selectedClusterId,
          prompt,
          compareWithBase: true,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || "Could not get an answer");
      } else {
        setResult(data);
      }
    } catch {
      setError("Failed to connect to the AI server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#0B0D12] p-6">
      <div className="max-w-[980px] mx-auto space-y-6">
        {/* Simple Header */}
        <div className="flex items-center justify-between gap-4 pb-4 border-b border-[#222938]">
          <div>
            <h1 className="text-xl font-bold text-[#F1F5F9] font-display">
              Test Your Trained MAH MCA CET AI
            </h1>
            <p className="text-xs text-[#94A3B8] mt-1">
              Pick a sample MAH MCA CET question below (or type any CET Math / Logical Reasoning problem) to compare your trained AI against the untrained model.
            </p>
          </div>
          {onBackToStudio && (
            <button
              onClick={onBackToStudio}
              className="px-3 py-2 rounded-lg bg-[#161B26] hover:bg-[#1A1F2B] border border-[#222938] text-xs font-medium text-[#F1F5F9] flex items-center gap-1.5 transition-colors shrink-0"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Back to Visual Blocks
            </button>
          )}
        </div>

        {/* Step 1: Pick Which Trained AI to Test */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-[#94A3B8]">
            1. Active Trained Model:
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {clusters.map((c) => {
              const isSelected = c.clusterId === selectedClusterId;
              return (
                <button
                  key={c.clusterId}
                  onClick={() => onSelectClusterId(c.clusterId)}
                  className={`p-3.5 rounded-xl border text-left transition-colors ${
                    isSelected
                      ? "bg-[#161B26] border-[#FF6B2B]"
                      : "bg-[#12161F] border-[#222938] hover:border-[#334155]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold text-[#F1F5F9] truncate">
                      {c.pipelineName}
                    </span>
                    <span className="text-xs font-mono text-[#10B981] shrink-0">
                      {Math.round(c.tokenAccuracy * 100)}% score
                    </span>
                  </div>
                  <div className="text-xs text-[#94A3B8] mt-1 font-mono">
                    ID: {c.clusterId} · {c.mode === "GRPO" ? "Reward Trained (GRPO)" : "Example Trained (SFT)"}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Step 2: Ask a MAH MCA CET Question */}
        <div className="p-5 rounded-xl bg-[#12161F] border border-[#222938] space-y-4">
          <div className="space-y-2">
            <span className="text-xs font-semibold text-[#F1F5F9] block">
              2. Click a Sample MAH MCA CET Question (or Type Your Own):
            </span>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_PROMPTS.map((sp, i) => (
                <button
                  key={i}
                  onClick={() => setPrompt(sp.text)}
                  className="px-3 py-1.5 rounded-lg bg-[#0B0D12] hover:bg-[#1A1F2B] border border-[#222938] text-xs text-[#F1F5F9] transition-colors"
                >
                  {sp.label}
                </button>
              ))}
            </div>
          </div>

          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Paste any MAH MCA CET Math or Logical Reasoning question here..."
            className="w-full p-3.5 rounded-lg bg-[#0B0D12] border border-[#222938] text-sm text-[#F1F5F9] focus:outline-none focus:border-[#FF6B2B] resize-none"
          />

          <div className="flex justify-end">
            <button
              onClick={handleRunInference}
              disabled={loading}
              className="px-6 py-2.5 rounded-lg bg-[#FF6B2B] hover:bg-[#e55a1f] text-xs font-semibold text-white flex items-center gap-2 transition-colors disabled:opacity-50"
            >
              <Send className="w-3.5 h-3.5" />
              {loading ? "Solving with CET Shortcuts..." : "Solve Question & Compare"}
            </button>
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-[#12161F] border border-[#EF4444]/50 text-xs text-[#EF4444]">
            {error}
          </div>
        )}

        {/* Step 3: Side-by-Side Answers */}
        {result && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Trained AI */}
            <div className="p-5 rounded-xl bg-[#12161F] border-2 border-[#10B981]/60 space-y-3">
              <div className="flex items-center gap-2 pb-3 border-b border-[#222938]">
                <CheckCircle2 className="w-5 h-5 text-[#10B981] shrink-0" />
                <div>
                  <div className="text-sm font-bold text-[#F1F5F9]">
                    After Training (MAH MCA CET AI · {result.clusterId})
                  </div>
                  <div className="text-xs text-[#10B981]">
                    Uses fast CET Exam Shortcut + Step-by-Step Verification
                  </div>
                </div>
              </div>
              <div className="text-xs leading-relaxed text-[#F1F5F9] whitespace-pre-wrap font-mono select-text">
                {result.loraResponse}
              </div>
            </div>

            {/* Untrained AI */}
            {result.baseModelResponse && (
              <div className="p-5 rounded-xl bg-[#12161F] border border-[#222938] space-y-3">
                <div className="flex items-center gap-2 pb-3 border-b border-[#222938]">
                  <Cpu className="w-5 h-5 text-[#94A3B8] shrink-0" />
                  <div>
                    <div className="text-sm font-bold text-[#94A3B8]">
                      Before Training (Raw Starter AI)
                    </div>
                    <div className="text-xs text-[#64748B]">
                      Long textbook method without CET exam shortcuts
                    </div>
                  </div>
                </div>
                <div className="text-xs leading-relaxed text-[#94A3B8] whitespace-pre-wrap font-mono select-text">
                  {result.baseModelResponse}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
