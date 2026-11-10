import { useEffect, useState } from "react";
import { FolderTree, FileCode, Copy, Check, Download, Terminal } from "lucide-react";

interface RepoFileItem {
  path: string;
  category: string;
  content: string;
}

export function RepoArchitectureView() {
  const [files, setFiles] = useState<RepoFileItem[]>([]);
  const [selectedPath, setSelectedPath] = useState<string>("n2n-back/apis/main.py");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch("/api/repo-files")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.files)) {
          setFiles(data.files);
        }
      })
      .catch(() => {});
  }, []);

  const currentFile =
    files.find((f) => f.path === selectedPath) ||
    files[0] || {
      path: "n2n-back/apis/main.py",
      category: "Backend FastAPI",
      content: "Loading repository file...",
    };

  const handleCopy = () => {
    navigator.clipboard.writeText(currentFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownload = () => {
    const blob = new Blob([currentFile.content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = currentFile.path.split("/").pop() || "file.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#0B0D12] p-6">
      <div className="max-w-[1320px] mx-auto space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-[#222938]">
          <div>
            <div className="text-xs text-[#94A3B8] mb-1">
              Full Repository Blueprint · FastAPI + Oumi + vLLM PagedAttention + WSL2
            </div>
            <h1 className="text-2xl font-bold text-[#F1F5F9] font-display">
              Vriksha AI Repository Structure &amp; Local WSL2 Deployment
            </h1>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Tree & Local Setup Guide (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            <div className="p-4 rounded-lg bg-[#12161F] border border-[#222938] space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-[#F1F5F9]">
                <FolderTree className="w-4 h-4 text-[#FF6B2B]" />
                <span>vriksha-ai/ File Explorer</span>
              </div>

              <div className="space-y-1.5">
                {files.map((file) => {
                  const active = file.path === currentFile.path;
                  return (
                    <button
                      key={file.path}
                      onClick={() => setSelectedPath(file.path)}
                      className={`w-full text-left px-3 py-2 rounded-lg border text-xs transition-colors flex items-center justify-between gap-2 ${
                        active
                          ? "bg-[#161B26] border-[#FF6B2B] text-[#F1F5F9]"
                          : "bg-[#0B0D12] border-[#222938] text-[#94A3B8] hover:text-[#F1F5F9]"
                      }`}
                    >
                      <span className="font-mono truncate">{file.path}</span>
                      <span className="text-[11px] text-[#64748B] shrink-0">
                        {file.category}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="p-4 rounded-lg bg-[#12161F] border border-[#222938] space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-[#F1F5F9]">
                <Terminal className="w-4 h-4 text-[#10B981]" />
                <span>Local Consumer GPU Setup (WSL2)</span>
              </div>
              <p className="text-xs text-[#94A3B8]">
                Optimized for NVIDIA RTX 3050 / 4060 laptops (4GB–8GB VRAM) with CUDA 12.1+ on Ubuntu 22.04 or Windows 11 WSL2.
              </p>
              <pre className="p-3 rounded bg-[#0B0D12] border border-[#222938] text-[11px] font-mono text-[#F1F5F9] overflow-x-auto select-text">
{`# 1. Backend (Oumi + vLLM + FastAPI)
cd n2n-back
python -m venv .venv
source .venv/bin/activate
cd apis && pip install -r requirements.txt
python main.py

# 2. Frontend (React Flow UI)
cd chain-prompt-ui
npm install && npm run dev`}
              </pre>
            </div>
          </div>

          {/* Right Code Viewer (8 cols) */}
          <div className="lg:col-span-8">
            <div className="rounded-lg bg-[#12161F] border border-[#222938] overflow-hidden">
              <div className="px-4 py-3 border-b border-[#222938] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-[#38BDF8]" />
                  <span className="text-xs font-mono font-semibold text-[#F1F5F9]">
                    {currentFile.path}
                  </span>
                  <span className="text-xs text-[#64748B]">
                    · {currentFile.category}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopy}
                    className="px-2.5 py-1 rounded bg-[#1A1F2B] hover:bg-[#222938] border border-[#222938] text-xs text-[#F1F5F9] flex items-center gap-1 whitespace-nowrap"
                  >
                    {copied ? (
                      <Check className="w-3.5 h-3.5 text-[#10B981]" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    {copied ? "Copied" : "Copy File"}
                  </button>
                  <button
                    onClick={handleDownload}
                    className="px-2.5 py-1 rounded bg-[#FF6B2B] hover:bg-[#e55a1f] text-xs font-medium text-white flex items-center gap-1 whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download
                  </button>
                </div>
              </div>
              <pre className="p-4 bg-[#0B0D12] text-xs font-mono text-[#F1F5F9] leading-relaxed overflow-x-auto max-h-[600px] select-text">
                {currentFile.content}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
