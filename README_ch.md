


<div align="center">

# <img src="./images/spider_icon2.png" width="60" style="vertical-align: text-bottom;"> Spider-Sense


<p align="center">
  <strong>基于内源性风险感知与分层自适应筛选的高效智能体防御框架</strong>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.05386-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2602.05386)
[![Paper](https://img.shields.io/badge/Paper-HF-orange.svg?logo=adobe-acrobat-reader&logoColor=white&style=for-the-badge)](https://huggingface.co/papers/2602.05386)
[![S2Bench](https://img.shields.io/badge/Dataset-S2Bench-yellow.svg?style=for-the-badge)](https://huggingface.co/datasets/aifinlab/S2Bench)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge)](#)

<br>

[English](./README.md) | [简体中文](./README_ch.md)

</div>

<p align="center">
  <em>一个事件驱动的防御框架，允许智能体保持潜在警惕，仅在感知到风险时触发防御。</em>
</p>

<div align="center">
</div>

</div>

---

## 📖 简介 (Abstract)

随着大语言模型（LLMs）向自主智能体（Autonomous Agents）进化，现有的防御机制大多采用 **“强制检查范式” (Mandatory Checking Paradigm)**，即无论是否存在实际风险，都会在预设阶段强制触发安全验证。这种方法导致了极高的延迟和计算冗余。

我们提出了 **Spider-Sense**，这是一个基于 **内源性风险感知 (Intrinsic Risk Sensing, IRS)** 的事件驱动防御框架。它允许智能体保持 **“潜在警惕” (Latent Vigilance)**，仅在感知到风险时才触发防御。一旦触发，Spider-Sense 将启动 **分层自适应筛选 (Hierarchical Adaptive Screening, HAS)** 机制，在效率与精度之间取得平衡：通过轻量级的相似度匹配解决已知攻击模式，同时将模糊的复杂案例升级为深度的内部推理分析。

---

## ⚡ 框架对比

### 痛点：强制检查

现有框架依赖于在每个阶段（规划、行动、观察）进行强制性的、重复的外部安全检查，导致延迟迅速累积，严重干扰了正常的用户交互。

<div align="center">
<img src="images/fig1_process (1).png" alt="Framework Comparison" width="800px" />
</div>

### 解决方案：内源性风险感知 (IRS)

Spider-Sense 利用 **主动的、内源性的风险意识**，仅在感知到异常时动态触发针对性的分析，从而显著降低系统开销。

---

## 🔬 方法概览

<div align="center">
<img src="images/fig2_21.png" alt="Framework Overview" width="900px" />
</div>

该框架基于 **“检测-审计-响应” (Detect-Audit-Respond)** 循环运行：

1. **内源性风险感知 (IRS)**：智能体保持潜在的警惕状态，持续监控四个阶段（Query, Plan, Action, Observation）的产物。
2. **感知标识 (Sensing Indicator)**：一旦感知到风险，智能体生成特定标识（如 `<|verify_user_intent|>`），暂停执行。
3. **分层自适应筛选 (HAS)**：
   * **粗粒度检测**：针对已知攻击模式数据库进行快速向量匹配。
   * **细粒度分析**：针对模糊或低相似度案例，调用 LLM 进行深度推理。
4. **自主决策**：智能体决定 **恢复执行 (Resume)**（如果安全）或 **拒绝/清洗 (Refuse/Sanitize)**（如果不安全）。

---

## 🛡️ 防御模块

Spider-Sense 使用专门的防御标签保护四个关键安全阶段：

| 阶段 (Stage)          | 模块标签 (Module Tag)           | 功能 (Function)               | 触发条件 (Trigger Condition)               |
| :-------------------- | :------------------------------ | :---------------------------- | :----------------------------------------- |
| **Query**       | `<\|verify_user_intent\|>`      | **智能体逻辑劫持**      | 当用户输入试图越狱或覆盖指令时。           |
| **Plan**        | `<\|validate_memory_plan\|>`    | **思维链操纵/记忆中毒** | 当推理过程或检索到的记忆显示出中毒迹象时。 |
| **Action**      | `<\|audit_action_parameters\|>` | **工具滥用**            | 在执行带有可疑参数的高风险工具之前。       |
| **Observation** | `<\|sanitize_observation\|>`    | **间接提示注入**        | 当接收到包含隐藏恶意指令的工具输出时。     |

---

## 📊 S<sup>2</sup>Bench 基准测试

为了进行严格的评估，我们推出了 **S<sup>2</sup>Bench**，这是一个全生命周期的基准测试集，具有以下特点：

* **多阶段攻击**：覆盖 Query, Plan, Action, 和 Observation 阶段。
* **真实工具调用**：涉及实际的工具选择和参数生成（约 300 个函数）。
* **高质量假阳性样本 (Hard Benign Prompts)**：包含 153 个精心构建的良性样本，用于测试过度防御（误报）。

---

## 📈 性能表现

Spider-Sense 在 **S<sup>2</sup>Bench**, **Mind2Web**, 和 **eICU** 上实现了 SOTA 级的防御性能，且延迟极低。

<div align="center">
  <img src="images/table2.png" alt="Table 2" width="600px" />
  <br />
  <img src="images/table3.png" alt="Table 3" width="600px" />
</div>

* **最低攻击成功率 (ASR)**：有效拦截复杂的多阶段攻击。
* **最低误报率 (FPR)**：能够区分微妙的良性意图与攻击行为。
* **极低的延迟损耗**：仅 **~8.3%**，相比之下，某些强制检查基线的延迟高达 200% 以上。

### 消融实验

<div align="center">
<img src="images/fig3_tag_ablation.png" alt="Ablation Study 1" width="375px" />
<img src="images/fig4_thinking_ablation.png" alt="Ablation Study 2" width="400px" />
</div>

* **IRS 的重要性**：移除任何阶段的感知都会导致 ASR 显著上升，尤其是 Action 阶段。
* **HAS 的平衡性**：结合粗粒度检测和细粒度分析，在安全性与效率之间取得了最佳平衡。

---

## 🔍 案例研究

**场景**：一个临床分析智能体通过工具检索患者记录。

<div align="center">
<img src="images/fig5_casestudy_modified2.png" alt="Case Study" width="500px" />
</div>

1. **攻击**：工具返回内容被注入了代码 `import fake_module`，试图诱导未授权执行。
2. **检测**：智能体的 IRS 激活感知标识 `<|sanitize_observation|>`。
3. **HAS 筛选**：内容被路由至检查器，被识别为在当前上下文中不合理。
4. **响应**：智能体自主终止执行，成功拦截攻击。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone [https://github.com/your-repo/SpiderSense.git](https://github.com/your-repo/SpiderSense.git)
cd SpiderSense
pip install -r requirements.txt
```


### 2. 配置

设置 API 密钥。我们推荐使用 **OpenRouter** 以支持多种模型。

**Bash**

```
# 创建 .env 文件
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

### 3. 运行防御测试

启用 **内源性风险感知 (IRS)** 模板运行测试：

**Bash**

```
python main_attacker.py \
    --llm_name qwen-max \
    --defense_template template/spider_template.txt \
    --template_position system \
    --res_file results/spider_sense_test.csv
```

---

---

## 📂 项目结构

**Plaintext**

```
SpiderSense/
├── config/                     # 实验配置文件
│   ├── Defense_2.yml           # 默认 SpiderSense 配置
│   ├── DPI.yml                 # 直接提示注入 (DPI) 配置
│   ├── OPI.yml                 # 间接提示注入 (OPI) 配置
│   └── MP.yml                  # 记忆中毒 (Memory Poisoning) 配置
├── data/                       # 基准测试与攻击数据集
├── pyopenagi/
│   └── agents/                 # Agent 实现 (包含 sandbox.py)
├── template/
│   ├── spider_template.txt     # 核心防御协议模板
│   └── sandbox_judge_*.txt     # 各阶段的裁判 (Judge) 提示词
├── main_attacker.py            # 单案例测试入口
└── scripts/                    # 批量执行脚本
```

---

## 🔧 配置详解

SpiderSense 使用 `config/` 目录下的 YAML 文件管理实验设置。

| **配置文件**           | **描述**                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Defense_2.yml**      | 标准配置，启用 IRS 和所有防御模块。                                                   |
| **DPI.yml**            | 专门用于测试**直接提示注入 (Direct Prompt Injection)**攻击。                          |
| **OPI.yml**            | 专门用于测试**间接提示注入 (Indirect Prompt Injection)** （Observation 阶段）。 |
| **MP.yml**             | 专门用于测试**记忆中毒 (Memory Poisoning)** （Retrieval 阶段）。                |
| **DPI_MP.yml**         | **DPI + 记忆中毒**组合攻击。                                                    |
| **DPI_OPI.yml**        | **DPI + 间接提示注入**组合攻击。                                                |
| **OPI_MP.yml**         | **OPI + 记忆中毒**组合攻击。                                                    |
| **Tool_Injection.yml** | **工具描述注入** ：篡改工具定义以误导智能体。                                   |
| **Adv_Tools.yml**      | **对抗性工具** ：注入带有混淆名称的诱饵工具。                                   |
| **Lies_Loop.yml**      | **循环欺骗** ：强迫智能体欺骗模拟的人类审批者。                                 |
| **Logic_Backdoor.yml** | **逻辑后门** ：触发逻辑谬误或暂停安全协议。                                     |

使用特定配置运行：

**Bash**

```
python scripts/agent_attack.py --cfg_path config/DPI.yml
```

---

## 🚀 进阶用法

对于大规模基准测试，请使用 `scripts/` 目录下的脚本。

### 运行串行基准测试 (Stage 1)

按顺序执行攻击。适用于调试或较小的模型。

**Bash**

```
python scripts/run_stage_1_serial.py \
    --llm_name gpt-4o-mini \
    --num_attack 10 \
    --num_fp 10
```

### 运行并行基准测试 (Stage 4)

并行执行攻击以加速评估。

**Bash**

```
python scripts/run_stage_4_parallel.py \
    --llm_name qwen-max \
    --num_attack 50
```

---

## 📖 引用

如果您在研究中使用了 Spider-Sense 或 S<sup>2</sup>Bench，请引用我们的论文：

**Code snippet**

```
@misc{yu2026spidersenseintrinsicrisksensing,
      title={Spider-Sense: Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening}, 
      author={Zhenxiong Yu and Zhi Yang and Zhiheng Jin and Shuhe Wang and Heng Zhang and Yanlin Fei and Lingfeng Zeng and Fangqi Lou and Shuo Zhang and Tu Hu and Jingping Liu and Rongze Chen and Xingyu Zhu and Kunyi Wang and Chaofa Yuan and Xin Guo and Zhaowei Liu and Feipeng Zhang and Jie Huang and Huacan Wang and Ronghao Chen and Liwen Zhang},
      year={2026},
      eprint={2602.05386},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.05386}, 
}
```

---

## 🙏 致谢与团队

本工作由 **QuantaAlpha** 和 **AIFin Lab** 支持。

### 关于 AIFin Lab

**AIFin Lab** 由上财张立文教授发起，深耕 **AI + 金融 / 统计 / 数据科学** 交叉领域，团队汇聚上财、复旦、东大、CMU、港中文等校前沿学者，打造数据、模型、评测、智能提示全链路体系。我们诚挚欢迎全球优秀的本科、硕士、博士生以及前沿学者加入 **AIFin Lab**，共同探索 AI Agent 安全与金融智能的边界！

如果您对此项目感兴趣并希望参与贡献或开展研究合作，请将您的简历/简介发送至：
📩 **[aifinlab.sufe@gmail.com](mailto:aifinlab.sufe@gmail.com)** 并同时抄送 (CC) 至：
📧 **[zhang.liwen@shufe.edu.cn](mailto:zhang.liwen@shufe.edu.cn)**

期待你的加入！

### 关于 QuantaAlpha

**QuantaAlpha** 成立于 2025 年 4 月，由来自清华、北大、中科院、CMU、港科大等名校的教授、博士后、博士与硕士组成。我们的使命是探索智能的“量子”，引领智能体研究的“阿尔法”前沿——从 CodeAgent 到自进化智能，再到金融与跨领域专用智能体，致力于重塑人工智能的边界。

2026 年，我们将在  **CodeAgent** （真实世界任务的端到端自主执行）、 **DeepResearch** 、 **Agentic Reasoning/Agentic RL** 、**自进化与协同学习** 等方向持续产出高质量研究成果，欢迎对我们方向感兴趣的同学加入我们！

**团队主页：** [https://quantaalpha.github.io/](https://quantaalpha.github.io/)
