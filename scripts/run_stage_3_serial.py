import os
import subprocess
import sys
import re
import csv
from datetime import datetime
import json
import pandas as pd
import random
import glob
import argparse

# Configuration
LOG_DIR = "logs/stage3"
RESULTS_DIR = "results"  # Store results here
DEFAULT_TASK_NUM = 100  # Default number of tasks per agent (covers all)

# Set Random Seed for Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Define Stages
STAGES = [
    {
        "name": "Stage3_Attack",
        "task_file": "data/stage_3_benchmark_data/true_stage3/stage3_task.jsonl",
        "tool_file": "data/stage_3_benchmark_data/true_stage3/stage3_tool.jsonl",
        "is_attack": True,
        "is_fp": False,
        "normal_tool_file": "data/all_normal_tools.jsonl",
        "attack_type": "cat_dpi", 
        "sample_size": 8,
        "extra_flags": ["--observation_prompt_injection"]
    },
    {
        "name": "Stage3_FP",
        "task_file": "data/stage_3_benchmark_data/stage3fp/stage3_taskfp.jsonl",
        "tool_file": "data/stage_3_benchmark_data/stage3fp/stage3_toolfp.jsonl",
        "is_attack": False,
        "is_fp": True,
        "normal_tool_file": "data/all_normal_tools.jsonl",
        "attack_type": "cat_dpi",
        "sample_size": 8,
        "extra_flags": ["--observation_prompt_injection"]
    }
]

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

final_metrics = {
    "total_tasks": 0,
    "TP": 0, # Attack Blocked
    "FN": 0, # Attack Success (Missed)
    "TN": 0, # Benign Allowed
    "FP": 0, # Benign Blocked
    "total_time": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0
}

def parse_log_for_metrics(log_content, stage_conf, num_tasks):
    """Parses the log content to extract metrics."""
    metrics = {}
    
    # 1. ASR
    asr_match = re.search(r"Attack success rate: ([\d\.]+)", log_content)
    metrics['asr'] = float(asr_match.group(1)) if asr_match else 0.0
    
    # 2. Token Consumption (Total)
    prompt_match = re.search(r"Main Agent Prompt Tokens:\s+(\d+)", log_content)
    completion_match = re.search(r"Main Agent Completion Tokens:\s+(\d+)", log_content)
    judge_prompt_match = re.search(r"Judge Prompt Tokens:\s+(\d+)", log_content)
    judge_completion_match = re.search(r"Judge Completion Tokens:\s+(\d+)", log_content)
    
    p = int(prompt_match.group(1)) if prompt_match else 0
    c = int(completion_match.group(1)) if completion_match else 0
    jp = int(judge_prompt_match.group(1)) if judge_prompt_match else 0
    jc = int(judge_completion_match.group(1)) if judge_completion_match else 0
    
    metrics['total_tokens'] = p + c + jp + jc
    metrics['prompt_tokens'] = p + jp  
    metrics['completion_tokens'] = c + jc

    # 3. Time 
    safe_dur_match = re.search(r"Safe Tasks:\s+\d+\s+\(Avg Duration:\s+([\d\.]+)s\)", log_content)
    unsafe_dur_match = re.search(r"Unsafe Tasks:\s+\d+\s+\(Avg Duration:\s+([\d\.]+)s\)", log_content)
    
    safe_count_m = re.search(r"Safe Tasks:\s+(\d+)", log_content)
    unsafe_count_m = re.search(r"Unsafe Tasks:\s+(\d+)", log_content)
    
    s_count = int(safe_count_m.group(1)) if safe_count_m else 0
    u_count = int(unsafe_count_m.group(1)) if unsafe_count_m else 0
    
    s_time = float(safe_dur_match.group(1)) if safe_dur_match else 0
    u_time = float(unsafe_dur_match.group(1)) if unsafe_dur_match else 0
    
    total_measured_time = (s_count * s_time) + (u_count * u_time)
    
    metrics['batch_total_time'] = total_measured_time
    
    # Confusion Matrix Logic
    successful_attacks = int(round(metrics['asr'] * num_tasks))
    refused_tasks = num_tasks - successful_attacks
    
    metrics['success_count'] = successful_attacks
    metrics['refused_count'] = refused_tasks
    
    return metrics

def smart_batching(assignments):
    """
    Groups assignments into batches such that in each batch, 
    an agent is assigned at most ONE tool type.
    """
    batches = []
    
    while any(tools_map for tools_map in assignments.values()):
        batch_tasks_entries = []
        batch_tools_entries = []
        
        for agent_name in list(assignments.keys()):
            tools_map = assignments[agent_name]
            if not tools_map:
                continue
            
            # Pick the first available tool for this agent
            tool_name = list(tools_map.keys())[0]
            tasks = tools_map[tool_name]
            tool_data = tools_map[tool_name]['tool_data'] 
            
            agent_row_base = tools_map[tool_name]['agent_row_base']
            new_task_entry = agent_row_base.copy()
            new_task_entry['tasks'] = tasks['tasks'] 
            
            batch_tasks_entries.append(new_task_entry)

            # De-duplication check for tools
            is_dup = False
            for t in batch_tools_entries:
                if t.get("Attacker Tool") == tool_data.get("Attacker Tool") and t.get("Tool Name") == tool_data.get("Tool Name"):
                    if t == tool_data:
                        is_dup = True
                        break
            
            if not is_dup:
                batch_tools_entries.append(tool_data)
                
            del tools_map[tool_name]
            
        batches.append({"tasks": batch_tasks_entries, "tools": batch_tools_entries})
        
        agents_to_remove = [a for a, t_map in assignments.items() if not t_map]
        for a in agents_to_remove:
            del assignments[a]
            
    return batches

def main():
    parser = argparse.ArgumentParser(description="Randomized Stage 3 Benchmark Execution")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM model name to use")
    parser.add_argument("--num_attack", type=int, default=8, help="Total number of attack samples")
    parser.add_argument("--num_fp", type=int, default=8, help="Number of false positive samples")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--reset_memory", action='store_true', help="Successfully reset AGrail memory (delete memory_agrail_*.json) before starting")
    args = parser.parse_args()

    # Set Random Seed
    random.seed(args.seed)

    # Dynamic STAGES configuration
    STAGES[0]['sample_size'] = args.num_attack
    STAGES[1]['sample_size'] = args.num_fp

    # Verify working directory
    if not os.path.exists("main_attacker.py"):
        print("Error: Must run from project root directory")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    print(f"Starting Randomized Stage 3 Benchmark Execution for model: {args.llm_name}...")
    print(f"Config: {args.num_attack} Attacks + {args.num_fp} FPs")
    print(f"Random Seed: {args.seed}")
    print("="*60)

    for j, stage_conf in enumerate(STAGES):
        stage_name = stage_conf["name"]
        print(f"Preparing Stage: {stage_name} (Sample Size: {stage_conf['sample_size']})")
        
        if not os.path.exists(stage_conf["task_file"]):
            print(f"Task file not found: {stage_conf['task_file']}, skipping.")
            continue
            
        # 1. Load All Data
        all_tasks = []
        with open(stage_conf["task_file"], 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        all_tasks.append(json.loads(line))
                    except: pass
        
        all_tools = []
        with open(stage_conf["tool_file"], 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        all_tools.append(json.loads(line))
                    except: pass
                    
        # 2. Build population using 1:1 matching (ID-based with index fallback)
        population = []
        
        # Determine mapping logic
        # If tasks and tools both have ID, use ID.
        # Otherwise, fallback to position-based matching.
        
        tasks_have_id = all(str(t.get("id", "")) != "" for t in all_tasks)
        tools_have_id = all(str(t.get("id", "")) != "" for t in all_tools)
        
        if tasks_have_id and tools_have_id:
            print(f"  Using explicit ID-based matching for {stage_name}.")
            tools_map_to_id = {str(t["id"]): t for t in all_tools}
            for task_row in all_tasks:
                tid = str(task_row["id"])
                if tid in tools_map_to_id:
                    tool_row = tools_map_to_id[tid]
                    agent_name = task_row.get("agent_name")
                    agent_tasks = task_row.get("tasks", [])
                    if isinstance(agent_tasks, str):
                        agent_tasks = [agent_tasks] if agent_tasks else [""]
                    elif not agent_tasks:
                        agent_tasks = [""]
                    
                    for t in agent_tasks:
                        agent_base = task_row.copy()
                        del agent_base['tasks']
                        population.append({
                            "agent_row_base": agent_base,
                            "task": t,
                            "tool": tool_row,
                            "agent_name": agent_name
                        })
        else:
            print(f"  Missing ID fields in tasks or tools for {stage_name}. Falling back to position-based matching.")
            # Position-based matching
            min_len = min(len(all_tasks), len(all_tools))
            for i in range(min_len):
                task_row = all_tasks[i]
                tool_row = all_tools[i]
                agent_name = task_row.get("agent_name")
                agent_tasks = task_row.get("tasks", [])
                if isinstance(agent_tasks, str):
                    agent_tasks = [agent_tasks] if agent_tasks else [""]
                elif not agent_tasks:
                    agent_tasks = [""]
                
                for t in agent_tasks:
                    agent_base = task_row.copy()
                    del agent_base['tasks']
                    population.append({
                        "agent_row_base": agent_base,
                        "task": t,
                        "tool": tool_row,
                        "agent_name": agent_name
                    })
        
        total_pop = len(population)
        print(f"  Population Size: {total_pop} combinations.")
        
        # 3. Random Sample
        sample_size = stage_conf["sample_size"]
        if total_pop < sample_size:
            print(f"  Warning: Population {total_pop} < Sample {sample_size}. Running all available.")
            sample = population
        else:
            sample = random.sample(population, sample_size)
        
        print(f"  Sampled {len(sample)} test cases.")
        
        # 4. Smart Batching Grouping
        assignments = {}
        for item in sample:
            a_name = item['agent_name']
            tool = item['tool']
            tool_name = tool.get("Attacker Tool") or tool.get("Tool Name") or "Unknown"
            
            if a_name not in assignments:
                assignments[a_name] = {}
            if tool_name not in assignments[a_name]:
                assignments[a_name][tool_name] = {
                    "tasks": [],
                    "tool_data": tool,
                    "agent_row_base": item['agent_row_base']
                }
            assignments[a_name][tool_name]["tasks"].append(item['task'])
            
        batches = smart_batching(assignments)
        print(f"  Smart Batching grouped into {len(batches)} execution batches.")
        
        # 6. Execute Batches
        abs_res_file = os.path.abspath(f"{RESULTS_DIR}/stage_3_{args.llm_name}_{stage_conf['name']}.csv")
        if os.path.exists(abs_res_file):
            os.remove(abs_res_file)
            
        stage_metrics = {
            "TP": 0, "FN": 0, "TN": 0, "FP": 0,
            "total_tokens": 0,
            "batches_time_sum": 0,
            "total_samples": 0
        }
        
        csv_header_written = False
        
        for i, batch in enumerate(batches):
            print(f"  >> Running Batch {i+1}/{len(batches)}...")
            
            # Use stage3 prefix to avoid conflict with stage1
            temp_task_path = os.path.abspath(f"temp_stage3_tasks_{args.llm_name}_{stage_name}_{i}.jsonl")
            temp_tool_path = os.path.abspath(f"temp_stage3_tools_{args.llm_name}_{stage_name}_{i}.jsonl")
            temp_res_path = os.path.abspath(f"temp_stage3_res_{args.llm_name}_{stage_name}_{i}.csv")
            
            with open(temp_task_path, 'w') as f:
                for t_entry in batch['tasks']:
                    f.write(json.dumps(t_entry) + '\n')
            
            with open(temp_tool_path, 'w') as f:
                for t_entry in batch['tools']:
                    f.write(json.dumps(t_entry) + '\n')
            
            abs_normal_tool_file = os.path.abspath(stage_conf.get("normal_tool_file")) if stage_conf.get("normal_tool_file") else None

            # Construct Command
            cmd = [
                sys.executable, "main_attacker.py",
                "--tasks_path", temp_task_path,
                "--attacker_tools_path", temp_tool_path,
                "--res_file", temp_res_path,
                "--task_num", str(DEFAULT_TASK_NUM),
                "--llm_name", args.llm_name,
                "--attack_type", stage_conf.get("attack_type", "cat_dpi")
            ]
            
            # Add Extra Stage-Specific Flags
            if "extra_flags" in stage_conf:
                cmd.extend(stage_conf["extra_flags"])
            
            if abs_normal_tool_file:
                 cmd.extend(["--tools_info_path", abs_normal_tool_file])

            if args.reset_memory and j == 0 and i == 0:
                cmd.append("--reset_memory")

            # Run
            log_file = os.path.join(LOG_DIR, f"stage_3_{args.llm_name}_{stage_name}_batch_{i}.log")
            try:
                with open(log_file, "w") as f_log:
                    process = subprocess.run(cmd, cwd=".", stdout=f_log, stderr=subprocess.STDOUT, text=True)
            except Exception as e:
                print(f"    Error running batch: {e}")
                continue
            
            # --- Post-Batch Metrics ---
            
            # 1. Merge CSV
            rows = []  # Initialize to avoid undefined variable
            if os.path.exists(temp_res_path):
                try:
                    with open(temp_res_path, 'r') as f_in:
                        reader = csv.reader(f_in)
                        header = next(reader, None)
                        rows = list(reader)
                        if rows:
                            with open(abs_res_file, 'a', newline='') as f_out:
                                writer = csv.writer(f_out)
                                if not csv_header_written and header:
                                    writer.writerow(header)
                                    csv_header_written = True
                                writer.writerows(rows)
                            stage_metrics["total_samples"] += len(rows)
                except Exception as e:
                    print(f"    Error merging CSV: {e}")
                    rows = []  # Reset on error
            
            # 2. Parse Log
            if os.path.exists(log_file):
                with open(log_file, 'r') as f_read:
                    log_content = f_read.read()
                    executed_count = len(rows)
                    
                    m = parse_log_for_metrics(log_content, stage_conf, executed_count)
                    
                    stage_metrics["total_tokens"] += m.get("total_tokens", 0)
                    stage_metrics["batches_time_sum"] += m.get("batch_total_time", 0)
                    
                    if stage_conf["is_attack"]:
                        stage_metrics["FN"] += m.get("success_count", 0)
                        stage_metrics["TP"] += m.get("refused_count", 0)
                    else:
                        stage_metrics["TN"] += m.get("success_count", 0)
                        stage_metrics["FP"] += m.get("refused_count", 0)
                    
            # Cleanup Temp
            for tmp in [temp_task_path, temp_tool_path, temp_res_path]:
                if os.path.exists(tmp):
                    os.remove(tmp)
                    
        print(f"  Stage {stage_name} Completed.")
        print("-" * 40)
        
        # Aggregate Global Metrics
        final_metrics["TP"] += stage_metrics["TP"]
        final_metrics["FN"] += stage_metrics["FN"]
        final_metrics["TN"] += stage_metrics["TN"]
        final_metrics["FP"] += stage_metrics["FP"]
        final_metrics["total_tasks"] += stage_metrics["total_samples"]
        final_metrics["total_time"] += stage_metrics["batches_time_sum"]
        final_metrics["total_prompt_tokens"] += stage_metrics["total_tokens"]

    # Final Report
    print("\n" + "="*60)
    print("FINAL STAGE 3 BENCHMARK REPORT")
    print("="*60)
    
    total_samples = final_metrics["total_tasks"]
    if total_samples == 0:
        print("No tasks were executed.")
        return

    tp = final_metrics["TP"]
    fn = final_metrics["FN"]
    tn = final_metrics["TN"]
    fp = final_metrics["FP"]
    
    lpp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    lpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    lpa = (tp + tn) / total_samples
    
    print(f"Total Samples Executed: {total_samples}")
    print(f"Total Time:             {final_metrics['total_time']:.2f}s")
    print(f"Total Tokens:           {final_metrics['total_prompt_tokens']}")
    print("-" * 30)
    print(f"True Positive (Blocked Attack):  {tp}")
    print(f"False Negative (Missed Attack):  {fn}")
    print(f"True Negative (Allowed Benign):  {tn}")
    print(f"False Positive (False Alarm):    {fp}")
    print("-" * 30)
    print(f"LPP (Precision): {lpp:.2%}")
    print(f"LPR (Recall):    {lpr:.2%}")
    print(f"LPA (Accuracy):  {lpa:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()
