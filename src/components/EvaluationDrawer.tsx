import { useState } from "react";
import {
  ChevronUp,
  ChevronDown,
  Activity,
  Terminal,
  Play,
} from "lucide-react";
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
  const [viewMode, setViewMode] = useState<"charts" | "logs">("charts");

  const latest =
    metrics.length > 0
      ? metrics[metrics.length - 1]
      : { step: 0, loss: 2.4, perplexity: 11.0, accuracy: 0.48, reward: 0.35, kl: 0.0 };

  const buildPolyline = (
    selector: (m: EvalMetricStep) => number,
    minVal: number,
    maxVal: number,
    width: number,
    height: number
  ) => {
    if (metrics.length === 0) return "";
    return metrics
      .map((m, idx) => {
        const x =
          metrics.length === 1
            ? width / 2
            : (idx / (metrics.length - 1)) * (width - 24) + 12;
        const val = selector(m);
        const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));
        const y = height - 14 - norm * (height - 28);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  return (
    <div
      className={`border-t border-[#222938] bg-[#12161F] transition-all duration-200 flex flex-col shrink-0 ${
        isOpen ? "h-[264px]" : "h-[42px]"
      }`}
    >
      {/* Top Bar of Drawer */}
      <div className="h-[42px] px-4 flex items-center justify-between gap-4 border-b border-[#222938]/70 shrink-0">
        <div className="flex items-center gap-4 min-w-0">
          <button
            onClick={onToggle}
            className="flex items-center gap-2 text-xs font-semibold text-[#F1F5F9] hover:text-[#FF6B2B] transition-colors whitespace-nowrap shrink-0"
          >
            {isOpen ? (
              <ChevronDown className="w-4 h-4 text-[#94A3B8]" />
            ) : (
              <ChevronUp className="w-4 h-4 text-[#94A3B8]" />
            )}
            <span>Live Training Progress &amp; Scoreboard</span>
          </button>

          <div className="hidden sm:flex items-center gap-2 text-xs text-[#94A3B8] font-mono tabular-nums truncate">
            <span>Trained AI ID: {activeClusterId}</span>
            <span aria-hidden="true">·</span>
            <span>Mistake Rate (Loss): {latest.loss.toFixed(2)}</span>
            <span aria-hidden="true">·</span>
            <span>
              Answer Score:{" "}
              {((mode === "GRPO" ? latest.reward : latest.accuracy) * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {isOpen && (
            <div className="flex items-center gap-1 p-0.5 bg-[#0B0D12] rounded border border-[#222938]">
              <button
                onClick={() => setViewMode("charts")}
                className={`px-2.5 py-1 rounded text-[11px] font-medium flex items-center gap-1 transition-colors whitespace-nowrap ${
                  viewMode === "charts"
                    ? "bg-[#1A1F2B] text-[#F1F5F9]"
                    : "text-[#94A3B8] hover:text-[#F1F5F9]"
                }`}
              >
                <Activity className="w-3 h-3" />
                Visual Learning Charts
              </button>
              <button
                onClick={() => setViewMode("logs")}
                className={`px-2.5 py-1 rounded text-[11px] font-medium flex items-center gap-1 transition-colors whitespace-nowrap ${
                  viewMode === "logs"
                    ? "bg-[#1A1F2B] text-[#F1F5F9]"
                    : "text-[#94A3B8] hover:text-[#F1F5F9]"
                }`}
              >
                <Terminal className="w-3 h-3" />
                Step-by-Step Log ({logs.length})
              </button>
            </div>
          )}

          <button
            onClick={() => onDeployCluster(activeClusterId)}
            disabled={isTraining}
            className="px-3 py-1 rounded bg-[#10B981] hover:bg-[#059669] text-[11px] font-semibold text-[#0B0D12] flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
          >
            <Play className="w-3 h-3 fill-current" />
            Step 3: Chat with Trained AI ({activeClusterId})
          </button>
        </div>
      </div>

      {/* Drawer Content */}
      {isOpen && (
        <div className="flex-1 overflow-hidden p-4">
          {viewMode === "charts" ? (
            <div className="h-full grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Chart 1: Training Loss (Mistake Rate) */}
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-[#F1F5F9]">
                      1. AI Mistake Rate (Training Loss)
                    </span>
                    <span className="font-mono tabular-nums text-[#FF6B2B]">
                      {latest.loss.toFixed(3)} (Lower is Better)
                    </span>
                  </div>
                  <p className="text-[11px] text-[#94A3B8] mt-0.5">
                    Line going DOWN means the AI is making fewer mistakes.
                  </p>
                </div>
                <svg className="w-full h-[96px] overflow-visible" viewBox="0 0 320 96">
                  <line x1="12" y1="18" x2="308" y2="18" stroke="#1E293B" strokeDasharray="3 3" />
                  <line x1="12" y1="48" x2="308" y2="48" stroke="#1E293B" strokeDasharray="3 3" />
                  <line x1="12" y1="78" x2="308" y2="78" stroke="#1E293B" strokeDasharray="3 3" />
                  {metrics.length > 1 && (
                    <polyline
                      fill="none"
                      stroke="#FF6B2B"
                      strokeWidth="2.4"
                      points={buildPolyline((m) => m.loss, 0.3, 2.6, 320, 96)}
                    />
                  )}
                </svg>
                <div className="flex items-center justify-between text-[11px] text-[#64748B] font-mono tabular-nums">
                  <span>Start: 2.38 (Many Mistakes)</span>
                  <span>Now: {latest.loss.toFixed(2)} (Well Trained)</span>
                </div>
              </div>

              {/* Chart 2: Answer Quality / Accuracy */}
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-[#F1F5F9]">
                      2. Answer Quality Score ({mode === "GRPO" ? "Reward" : "Accuracy"})
                    </span>
                    <span className="font-mono tabular-nums text-[#10B981]">
                      {((mode === "GRPO" ? latest.reward : latest.accuracy) * 100).toFixed(1)}% (Higher is Better)
                    </span>
                  </div>
                  <p className="text-[11px] text-[#94A3B8] mt-0.5">
                    Line going UP means the AI is giving more accurate answers.
                  </p>
                </div>
                <svg className="w-full h-[96px] overflow-visible" viewBox="0 0 320 96">
                  <line x1="12" y1="18" x2="308" y2="18" stroke="#1E293B" strokeDasharray="3 3" />
                  <line x1="12" y1="48" x2="308" y2="48" stroke="#1E293B" strokeDasharray="3 3" />
                  <line x1="12" y1="78" x2="308" y2="78" stroke="#1E293B" strokeDasharray="3 3" />
                  {metrics.length > 1 && (
                    <polyline
                      fill="none"
                      stroke="#10B981"
                      strokeWidth="2.4"
                      points={buildPolyline(
                        (m) => (mode === "GRPO" ? m.reward : m.accuracy),
                        0.3,
                        1.0,
                        320,
                        96
                      )}
                    />
                  )}
                </svg>
                <div className="flex items-center justify-between text-[11px] text-[#64748B] font-mono tabular-nums">
                  <span>Before Training: 48%</span>
                  <span>Current Score: {((mode === "GRPO" ? latest.reward : latest.accuracy) * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Table 3: Beginner Summary Table */}
              <div className="p-3 rounded-lg bg-[#0B0D12] border border-[#222938] overflow-y-auto">
                <div className="text-xs font-semibold text-[#F1F5F9] mb-1">
                  3. Training Progress Snapshots
                </div>
                <p className="text-[11px] text-[#94A3B8] mb-2">
                  Saved automatically to <span className="font-mono">eval_results/</span>
                </p>
                <table className="w-full text-left text-[11px] font-mono tabular-nums">
                  <thead>
                    <tr className="text-[#64748B] border-b border-[#222938]">
                      <th className="pb-1 font-normal">Stage</th>
                      <th className="pb-1 font-normal text-right">Mistakes</th>
                      <th className="pb-1 font-normal text-right">Accuracy</th>
                      <th className="pb-1 font-normal text-right">Reward</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#161B26]">
                    {metrics.slice(-4).map((m) => (
                      <tr key={m.step} className="text-[#F1F5F9]">
                        <td className="py-1 text-[#94A3B8]">Step {m.step}</td>
                        <td className="py-1 text-right">{m.loss.toFixed(2)}</td>
                        <td className="py-1 text-right text-[#10B981]">
                          {(m.accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="py-1 text-right text-[#8B5CF6]">
                          {Math.round(m.reward * 100)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="h-full rounded-lg bg-[#0B0D12] border border-[#222938] p-3 overflow-y-auto font-mono text-xs space-y-1 select-text">
              {logs.map((line, i) => (
                <div key={i} className="text-[#94A3B8] leading-relaxed">
                  {line}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
