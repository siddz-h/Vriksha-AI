# Vriksha AI — Visual No-Code Studio for Training Small Language Models

**Build, fine-tune, and test your own custom AI models using 4 visual blocks—designed to run on everyday laptops.**

---

## 1. What is Vriksha AI? (In Simple Words)

Training an AI model usually requires writing hundreds of lines of Python code, setting up complex memory flags, and renting expensive cloud servers.

**Vriksha AI** turns that entire process into a visual, drag-and-drop workspace. Think of it like teaching a student using four connected building blocks:

1. **Block 1 — Study Data (The Textbook):** Pick the example questions and answers you want your AI to learn from (such as math problems, logical reasoning puzzles, or helpful customer chats).
2. **Block 2 — Starter AI (The Student):** Pick a compact, lightweight AI model (such as `Qwen-0.5B` or `SmolLM2-135M`) that is small enough to run on a standard 4GB–8GB laptop graphics card.
3. **Block 3 — AI Trainer (The Teaching Method):** Choose *how* your AI learns using the **Oumi** training engine:
   - **Learn by Example (SFT):** The AI studies good question-and-answer pairs and learns to follow that exact style.
   - **Learn by Reward (GRPO):** The AI attempts multiple ways to solve a problem and receives a reward whenever it shows clear step-by-step logic (`<think>...</think>`) and arrives at the verified answer.
4. **Block 4 — Finished AI (The Graduate):** Every training run produces a unique model tag called a **ClusterID**. You can immediately type questions and compare your newly trained AI side-by-side against the untrained starter model.

---

## 2. Why Use Vriksha AI?

- **Zero Coding Required:** Connect visual blocks from left to right. Vriksha AI automatically writes the underlying **Oumi YAML** training configuration for you.
- **Built for Normal Laptops:** Automatically applies **4-bit QLoRA memory compression** and **8-bit Paged Optimizers**, keeping GPU memory usage around **2.1 GB to 3.5 GB VRAM** so entry-level laptops (like an NVIDIA RTX 3050 or 4060) never freeze.
- **Side-by-Side Before & After Testing:** Don't just guess if training worked—ask any question in the built-in playground to see the **After Training (Custom AI)** answer right next to the **Before Training (Raw Starter AI)** answer.
- **Small-Scale Exam & Domain Ready:** Includes a ready-to-run **MAH MCA CET Math & Logical Reasoning** small-scale dataset (Set Theory, Permutations & Combinations, Probability, Number Series, Coding-Decoding, and Coordinate Geometry) to test domain adaptation in minutes.

---

## 3. Technology Used Under the Hood

| Layer | Tools Used | Purpose |
| :--- | :--- | :--- |
| **Visual Interface** | React 19, Vite, React Flow (`@xyflow/react`), Tailwind CSS | Interactive 4-block pipeline canvas, live scoreboard, and side-by-side chat playground |
| **Training Engine** | Oumi Framework, PyTorch, HuggingFace PEFT / TRL | Runs Supervised Fine-Tuning (`TRL_SFT`) and Group Relative Policy Optimization (`TRL_GRPO`) |
| **Fast Serving** | vLLM (PagedAttention) + Dynamic LoRA Routing | Loads trained `ClusterID` adapters on demand without restarting the base model |
| **Backend Gateway** | Express / FastAPI (`n2n-back`) | Compiles visual blocks into `.yaml` recipes, tracks evaluation scores, and routes inference |

---

## 4. Folder Structure Explained Simply

```text
vriksha-ai/
├── src/                         # Visual Frontend (React + React Flow)
│   ├── components/
│   │   ├── DagNodes.tsx         # The 4 visual blocks on the canvas
│   │   ├── NodeInspector.tsx    # Simple right-side settings panel & YAML exporter
│   │   ├── EvaluationDrawer.tsx # Bottom scoreboard (Accuracy & Mistake Rate)
│   │   ├── ClusterInferenceView.tsx # Side-by-side "Before vs. After" AI testing page
│   │   └── RepoArchitectureView.tsx # Built-in code viewer & downloader
│   ├── data/presets.ts          # Starter templates (MAH MCA CET & General Chat)
│   └── App.tsx                  # Main workspace layout
│
├── n2n-back/                    # Local Python GPU Training & Serving Stack
│   ├── data/
│   │   └── mah_mca_cet_small.jsonl # Curated small-scale Math & Reasoning dataset
│   └── apis/
│       ├── main.py              # FastAPI server that launches Oumi training jobs
│       ├── vllm_server.py       # Connects trained ClusterID adapters to vLLM
│       ├── utils/oumi_generator.py # Converts visual block settings into Oumi YAML
│       └── requirements.txt     # Python libraries for GPU training
│
├── eval_results/                # Saved accuracy & mistake-rate logs per run
├── output/                      # Saved LoRA adapter weights per ClusterID
└── server.ts                    # Full-stack web orchestrator & live comparison API
```

---

## 5. How to Run This Project

You can run Vriksha AI in **two ways**:

### Option A: Quick Start (Web Studio & Live Comparison)

Requires **Node.js 18+** installed on your computer.

```bash
# 1. Install web packages
npm install

# 2. Start the Vriksha AI studio on http://localhost:3000
npm run dev
```

Open `http://localhost:3000` in your browser:
1. Click **Block 1 (`MAH MCA CET Math & LR`)** to inspect the sample questions or add your own question.
2. Click **"Train AI Now"** in the top-right corner to run the training pipeline and view the scoreboard.
3. Click **"3. Chat with Trained AI"** to test questions and compare **After Training** vs. **Before Training**.

---

### Option B: Full Local GPU Training (Windows WSL2 or Linux)

If you have an **NVIDIA GPU (4GB+ VRAM)** and want to fine-tune real PyTorch weights locally using Oumi and vLLM:

#### Hardware & Software Needed
- **Operating System:** Ubuntu 22.04+ or Windows 11 with **WSL2** enabled
- **Graphics Card:** NVIDIA GPU with at least **4 GB VRAM** (e.g., RTX 3050 / 4060 Laptop)
- **Python:** Version **3.10+** with CUDA 12.1+ drivers

#### Step-by-Step Terminal Commands

```bash
# 1. Go to the Python backend folder
cd n2n-back

# 2. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Oumi, vLLM, and FastAPI dependencies
cd apis
pip install -r requirements.txt

# 4. Start the local Python training gateway
python main.py
```

Or run Oumi directly using the `.yaml` file downloaded from the Vriksha AI right-hand panel:

```bash
oumi train -c output/cluster-cet-2026/oumi_config.yaml
```

---

## 6. Frequently Asked Questions (For Beginners)

- **What is the difference between SFT and GRPO?**
  - **SFT (Learn by Example):** Best for general chat. You show the AI a question and a good answer, and it learns to imitate that answer.
  - **GRPO (Learn by Reward):** Best for math and reasoning exams. The AI generates 4 trial answers for every question and gets rewarded when its step-by-step calculation matches the verified answer.
- **What is a `ClusterID`?**
  - Every time you train a model, Vriksha AI saves the newly learned skills in a lightweight folder (`output/cluster-cet-2026`) instead of duplicating the entire base model. Entering that `ClusterID` loads your custom skills instantaneously.
- **Why does it use so little laptop memory?**
  - Vriksha AI uses **4-bit QLoRA**, which compresses the base AI brain to a quarter of its normal size and trains a small "adapter" layer on top.

---

## License

Released under the **Apache 2.0 License**. Free for students, researchers, and developers to use, modify, and share.
