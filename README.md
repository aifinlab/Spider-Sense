<div align="center">

# <img src="./images/spider_icon.png" width="40" style="vertical-align: middle;"> Spider-Sense

<p align="center">
  <strong>Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg" alt="arXiv"></a>
  <a href="./SpiderSense-paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-orange.svg?logo=adobe-acrobat-reader&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/aifinlab/S2Bench"><img src="https://img.shields.io/badge/Huggingface-S2Bench-yellow.svg" alt="Benchmark"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a> 
  <a href="./README_ch.md"><img src="https://img.shields.io/badge/Language-中文版-db7093.svg" alt="Chinese"></a>
</p>

<p align="center">
  <em>An event-driven defense framework allowing agents to maintain latent vigilance and trigger defenses only upon risk perception.</em>
</p>

<!-- [Insert Image Here: Project Logo or Banner] -->

<div align="center">
<!-- <img src="static/imgs/logo.png" alt="Spider-Sense Logo" width="300px" /> -->
</div>

</div>

---

## 📖 Abstract

As large language models (LLMs) evolve into autonomous agents, most existing defense mechanisms adopt a **mandatory checking paradigm**, which forcibly triggers security validation at predefined stages regardless of actual risk. This approach leads to high latency and computational redundancy.

We propose **Spider-Sense**, an event-driven defense framework based on **Intrinsic Risk Sensing (IRS)**. It allows agents to maintain **latent vigilance** and trigger defenses only upon risk perception. Once triggered, Spider-Sense invokes a **Hierarchical Adaptive Screening (HAC)** mechanism that trades off efficiency and precision: resolving known patterns via lightweight similarity matching while escalating ambiguous cases to deep internal reasoning.

---

## ⚡ Framework Comparison

### The Problem: Mandatory Checking

Existing frameworks rely on forced, repetitive external security checks at every stage (Plan, Action, Observation), leading to rapidly accumulating latency and disrupting normal user interaction.

<!-- [Insert Figure 1 Here: Comparison between the Existing Framework and the Spider-Sense Framework] -->

<div align="center">
<img src="images/fig1_process (1).png" alt="Framework Comparison" width="800px" />
</div>

### The Solution: Intrinsic Risk Sensing (IRS)

Spider-Sense utilizes **proactive, endogenous risk awareness** to dynamically trigger targeted analysis only when anomalies are sensed, significantly reducing overhead.

---

## 🔬 Method Overview

<!-- [Insert Figure 2 Here: Overview of Spider-Sense Framework] -->

<div align="center">
<img src="images/fig2_21.png" alt="Framework Overview" width="900px" />
</div>

The framework operates on a **Detect-Audit-Respond** cycle:

1. **Intrinsic Risk Sensing (IRS)**: The agent maintains a latent state of vigilance. It continuously monitors artifacts across four stages (Query, Plan, Action, Observation).
2. **Sensing Indicator**: Upon perceiving a risk, the agent generates a specific indicator (e.g., `<|verify_user_intent|>`), pausing execution.
3. **Hierarchical Adaptive Screening (HAC)**:
   * **Coarse-grained Detection**: Fast vector matching against a database of known attack patterns.
   * **Fine-grained Analysis**: Deep reasoning by an LLM for ambiguous or low-similarity cases.
4. **Autonomous Decision**: The agent decides to **Resume** execution (if safe) or **Refuse/Sanitize** (if unsafe).

---

## 🛡️ Defense Modules

Spider-Sense protects four security-critical stages using specialized defense tags:

| Stage                 | Module Tag                      | Function                               | Trigger Condition                                                 |
| :-------------------- | :------------------------------ | :------------------------------------- | :---------------------------------------------------------------- |
| **Query**       | `<\|verify_user_intent\|>`      | **Agent Logic Hijacking**        | When user input attempts to jailbreak or override instructions.   |
| **Plan**        | `<\|validate_memory_plan\|>`    | **Thought-Process Manipulation** | When reasoning or retrieved memories show signs of poisoning.     |
| **Action**      | `<\|audit_action_parameters\|>` | **Tool-Use Exploitation**        | Before executing high-risk tools with suspicious parameters.      |
| **Observation** | `<\|sanitize_observation\|>`    | **Indirect Prompt Injection**    | Upon receiving tool outputs containing hidden malicious commands. |

---

## 📊 $S^2$Bench Benchmark

To facilitate rigorous evaluation, we introduce **S2Bench**, a lifecycle-aware benchmark featuring:

* **Multi-stage Attacks**: Covers Query, Plan, Action, and Observation stages.
* **Realistic Tool Execution**: Involves actual tool selection and parameter generation (approx. 300 functions).
* **Hard Benign Prompts**: Includes 153 carefully constructed benign samples to test for over-defense (False Positives).

---

## 📈 Performance Results

Spider-Sense achieves state-of-the-art defense performance with minimal latency on **S2Bench**, **Mind2Web**, and **eICU**.

<!-- [Insert Table 2 or 3 Here: Main Performance Results] -->

<div align="center">
  <img src="images/table2.png" alt="Table 2" width="600px" />
</div>

<div align="center">
  <img src="images/table3.png" alt="Table 3" width="600px" />
</div>

* **Lowest Attack Success Rate (ASR)**: Effectively blocks sophisticated multi-stage attacks.
* **Lowest False Positive Rate (FPR)**: Distinguishes subtle benign intents from attacks.
* **Marginal Latency Overhead**: Only **~8.3%**, compared to >200% for some mandatory checking baselines.

### Ablation Studies

<!-- [Insert Figure 3 & 4 Here: Ablation Study on IRS and HAC] -->

<div align="center">
<img src="images/fig3_tag_ablation.png" alt="Ablation Study 1" width="375px" />
<img src="images/fig4_thinking_ablation.png" alt="Ablation Study 2" width="400px" />
</div>

* **IRS Importance**: Removing sensing at any stage leads to significant ASR proliferation, especially at the Action stage.
* **HAC Balance**: Combining coarse detection and fine analysis yields the best trade-off between safety and efficiency.

---

## 🔍 Case Study

**Scenario**: A clinical analysis agent retrieving patient records via a utility tool.

<!-- [Insert Figure 5 Here: Case Study of Clinical Agent Interception] -->

<div align="center">
<img src="images/fig5_casestudy_modified2.png" alt="Case Study" width="500px" />
</div>

1. **Attack**: The tool return content is poisoned with injected code `import fake_module` to induce unauthorized execution.
2. **Detection**: The agent's IRS activates the sensing indicator `<|sanitize_observation|>`.
3. **HAC Screening**: The content is routed to the inspector. It is identified as contextually unjustified.
4. **Response**: The agent autonomously terminates execution, intercepting the attack.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-repo/SpiderSense.git
cd SpiderSense
pip install -r requirements.txt
```

### 2. Configuration

Set up your API keys. We recommend using **OpenRouter** for broad model support.

```bash
# Create .env file
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

### 3. Run Defense Test

Enable **Intrinsic Risk Sensing** (IRS) using the defense template:

```bash
python main_attacker.py \
    --llm_name qwen-max \
    --defense_template template/spider_template.txt \
    --template_position system \
    --res_file results/spider_sense_test.csv
```

---

---

## 📂 Project Structure

```text
SpiderSense/
├── config/                         # Experiment configs
│   ├── Defense_2.yml               # Default SpiderSense config
│   ├── DPI.yml                     # Direct Prompt Injection config
│   ├── OPI.yml                     # Indirect Prompt Injection config
│   └── MP.yml                      # Memory Poisoning config
├── data/                           # Benchmarks & Attack Datasets
├── pyopenagi/
│   └── agents/                     # Agent implementations (inc. sandbox.py)
├── template/
│   ├── spider_template.txt         # Core Defense-y1 Protocol
│   └── sandbox_judge_*.txt         # Judge prompts for each stage
├── main_attacker.py                # Entry point for single-case testing
└── scripts/                        # Batch execution scripts
```

---

## 🔧 Configuration

SpiderSense uses YAML files in the `config/` directory to manage experiment settings.

| Config File                  | Description                                                                               |
| :--------------------------- | :---------------------------------------------------------------------------------------- |
| **Defense_2.yml**      | Standard configuration enabling IRS and all defense modules.                              |
| **DPI.yml**            | Specialized for testing**Direct Prompt Injection** attacks.                         |
| **OPI.yml**            | Specialized for**Indirect Prompt Injection** (Observation stage).                   |
| **MP.yml**             | Specialized for**Memory Poisoning** (Retrieval stage).                              |
| **DPI_MP.yml**         | Combined**DPI + Memory Poisoning** attack.                                          |
| **DPI_OPI.yml**        | Combined**DPI + Indirect Prompt Injection** attack.                                 |
| **OPI_MP.yml**         | Combined**OPI + Memory Poisoning** attack.                                          |
| **Tool_Injection.yml** | **Tool Description Injection**: Manipulating tool definitions to mislead agents.    |
| **Adv_Tools.yml**      | **Adversarial Tools**: Injecting decoy tools with confusing names.                  |
| **Lies_Loop.yml**      | **Lies in the Loop**: Forcing agents to deceive simulated human approvers.          |
| **Logic_Backdoor.yml** | **Reasoning Backdoor**: Triggering logical fallacies or suspended safety protocols. |

To use a specific config:

```bash
python scripts/agent_attack.py --cfg_path config/DPI.yml
```

---

## 🚀 Advanced Usage

For large-scale benchmarking, use the scripts in the `scripts/` directory.

### Run Serial Benchmark (Stage 1)

Executes attacks sequentially. Useful for debugging or smaller models.

```bash
python scripts/run_stage_1_serial.py \
    --llm_name gpt-4o-mini \
    --num_attack 10 \
    --num_fp 10
```

### Run Parallel Benchmark (Stage 4)

Executes attacks in parallel to speed up evaluation.

```bash
python scripts/run_stage_4_parallel.py \
    --llm_name qwen-max \
    --num_attack 50
```

---

## 📖 Citation

If you use Spider-Sense or S2Bench in your research, please cite our paper:

```bibtex
@article{yu2026spidersense,
  title={Spider-Sense: Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening},
  author={Zhenxiong Yu and Zhi Yang and Zhiheng Jin and Shuhe Wang and Heng Zhang and Yanlin Fei and Lingfeng Zeng and Fangqi Lou and Shuo Zhang and Tu Hu and Jingping Liu and Rongze Chen and Xingyu Zhu and Kunyi Wang and Chaofa Yuan and Xin Guo and Zhaowei Liu and Feipeng Zhang and Jie Huang and Huacan Wang and Ronghao Chen and Liwen Zhang},
  journal={arXiv preprint arXiv:2601.xxxxx},
  year={2026}
}
```

---

## 🙏 Acknowledgments

This work was supported by the **AIFin Lab** and the **QuantaAlpha**.
### About AIFin Lab
Initiated by Professor Liwen Zhang from Shanghai University of Finance and Economics (SUFE), **AIFin Lab** is deeply rooted in the interdisciplinary fields of **AI + Finance, Statistics, and Data Science**. The team brings together cutting-edge scholars from top institutions such as SUFE, FDU, SEU, CMU, and CUHK. We are dedicated to building a comprehensive "full-link" system covering data, models, benchmarks, and intelligent prompting. We warmly welcome interested students to join us! 🤝  
**Email:** [aifinlab.sufe@gmail.com](mailto:aifinlab.sufe@gmail.com)

### About QuantaAlpha
Founded in April 2025, **QuantaAlpha** is composed of professors, postdoctoral researchers, PhDs, and Master's students from prestigious universities including THU, PKU, CAS, CMU, and HKUST. Our mission is to explore the "Quanta" of intelligence and lead the "Alpha" frontier of agent research—ranging from CodeAgents and self-evolving intelligence to financial and cross-domain specialized agents—dedicated to reshaping the boundaries of artificial intelligence.

In 2026, we will continue to produce high-quality research results in areas such as **CodeAgent** (end-to-end autonomous execution of real-world tasks), **DeepResearch**, **Agentic Reasoning/Agentic RL**, and **Self-Evolution & Collaborative Learning**. We welcome students interested in our research directions to join us!  
**Team Homepage:** [https://quantaalpha.github.io/](https://quantaalpha.github.io/)
