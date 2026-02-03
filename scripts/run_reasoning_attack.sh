export PYTHONPATH=$PYTHONPATH:.

python -u main_attacker.py \
    --tasks_path data/test_samples/stage_4_task.jsonl \
    --attacker_tools_path data/test_samples/stage_4_attack.jsonl \
    --tools_info_path data/all_normal_tools.jsonl \
    --llm_name qwen-max \
    --reasoning_backdoor \
    --trigger "DEPLOY_EMERGENCY_PATCH" \
    --task_num 1 \
    --defense_type "audit_action_parameters" \
    --res_file logs/result_stage4.csv \
    --result_file logs/result_file_stage4.log \
    > logs/stage_4_test.log 2>&1

echo "Stage 4 Test Completed. Check logs/stage_4_test.log for details."
