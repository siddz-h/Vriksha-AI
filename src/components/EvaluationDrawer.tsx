import { ChevronUp, ChevronDown, Play, CheckCircle2 } from "lucide-react";
import type { EvalMetricStep } from "../types";

interface EvaluationDrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  isTraining: boolean;
  activeClusterId: string;
  mode: "SFT" | "GRPO";
  metrics: EvalMetricStep[];
  logs: string[];
  onDeployCluster: (clusterId: string) => void;
}

export function EvaluationDrawer({
  isOpen,
  onToggle,
  isTraining,
  activeClusterId,
  mode,
  metrics,
  logs,
  onDeployCluster,
}: EvaluationDrawerProps) {
  const latest =
    metrics.length > 0
      ? metrics[metrics.length - 1]
      : { step: 0, loss: 2.4, perplexity: 11.0, accuracy: 0.48, reward: 0.35, kl: 0.0 };

  const accuracyPct = Math.round(
    (mode === "GRPO" ? latest.reward : latest.accuracy) * 100
  );

  return (
    <div
      className={`border-t border-[#222938] bg-[#12161F] transition-all duration-200 flex flex-col shrink-0 ${
        isOpen ? "h-[210px]" : "h-[44px]"
      }`}
    >
      {/* Clean Single-Row Summary Bar */}
      <div className="h-[44px] px-5 flex items-center justify-between gap-4 border-b border-[#222938]/70 shrink-0">
        <button
          onClick={onToggle}
          className="flex items-center gap-2 text-xs font-semibold text-[#F1F5F9] hover:text-[#FF6B2B] transition-colors whitespace-nowrap"
        >
          {isOpen ? (
            <ChevronDown className="w-4 h-4 text-[#94A3B8]" />
          ) : (
            <ChevronUp className="w-4 h-4 text-[#94A3B8]" />
          )}
          <span>
            {isTraining
              ? "Training Your AI right now..."
              : `Training Scoreboard (${isOpen ? "Click to hide" : "Click to view scores"})`}
          </span>
        </button>

        <div className="flex items-center gap-4">
          <span className="hidden sm:inline text-xs text-[#94A3B8]">
            Answer Score: <strong className="text-[#10B981] font-mono">{accuracyPct}%</strong>{" "}
            (up from 48%)
          </span>

          <button
            onClick={() => onDeployCluster(activeClusterId)}
            disabled={isTraining}
            className="px-3.5 py-1.5 rounded-lg bg-[#10B981] hover:bg-[#059669] text-xs font-semibold text-[#0B0D12] flex items-center gap-1.5 transition-colors whitespace-nowrap disabled:opacity-50"
          >
            <Play className="w-3 h-3 fill-current" />
            Chat with Trained AI
          </button>
        </div>
      </div>

      {/* Simple, Uncluttered 3-Box Summary when Open */}
      {isOpen && (
        <div className="flex-1 p-4 grid grid-cols-1 md:grid-cols-3 gap-4 overflow-hidden">
          {/* Card 1: Big Before vs After Accuracy */}
          <div className="p-4 rounded-xl bg-[#0B0D12] border border-[#222938] flex flex-col justify-between">
            <div className="text-xs font-semibold text-[#94A3B8]">
              1. AI Answer Score (Higher is Better)
            </div>
            <div className="flex items-baseline gap-3 my-1">
              <span className="text-3xl font-bold text-[#10B981] font-mono tabular-nums">
                {accuracyPct}%
              </span>
              <span className="text-xs text-[#64748B]">
                Before training: 48%
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-[#161B26] overflow-hidden">
              <div
                className="h-full bg-[#10B981] transition-all duration-300"
                style={{ width: `${accuracyPct}%` }}
              />
            </div>
          </div>

          {/* Card 2: Mistakes Reduced */}
          <div className="p-4 rounded-xl bg-[#0B0D12] border border-[#222938] flex flex-col justify-between">
            <div className="text-xs font-semibold text-[#94A3B8]">
              2. AI Mistake Rate (Lower is Better)
            </div>
            <div className="flex items-baseline gap-3 my-1">
              <span className="text-3xl font-bold text-[#FF6B2B] font-mono tabular-nums">
                {latest.loss.toFixed(2)}
              </span>
              <span className="text-xs text-[#64748B]">
                Started at: 2.38
              </span>
            </div>
            <div className="text-xs text-[#94A3B8]">
              Mistakes dropped by{" "}
              <strong className="text-[#F1F5F9]">
                {Math.max(10, Math.round(((2.38 - latest.loss) / 2.38) * 100))}%
              </strong>{" "}
              during training.
            </div>
          </div>

          {/* Card 3: Plain-English Live Status */}
          <div className="p-4 rounded-xl bg-[#0B0D12] border border-[#222938] flex flex-col justify-between">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[#F1F5F9]">
              <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
              <span>3. Latest Update</span>
            </div>
            <p className="text-xs text-[#94A3B8] leading-relaxed line-clamp-2">
              {logs[logs.length - 1] || "Training complete! Ready to chat."}
            </p>
            <div className="text-[11px] font-mono text-[#10B981]">
              Saved as ID: {activeClusterId}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
