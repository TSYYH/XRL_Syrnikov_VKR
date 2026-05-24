# Установка

```
git clone https://github.com/TSYYH/XRL_Syrnikov_VKR.git
cd XRL_Syrnikov_VKR
git clone https://github.com/FLAIROx/JaxMARL.git
cp mappo_rnn.py JaxMARL/baselines/MAPPO/mappo_rnn.py
cp coin_game.py JaxMARL/jaxmarl/environments/coin_game/coin_game.py
pip install -r requirements.txt
```

# Обучение

```
cd JaxMARL
```

## Кооперативная политика

```
python baselines/MAPPO/mappo_rnn.py \
  ENV_NAME=coin_game \
  '~ENV_KWARGS' \
  '+ENV_KWARGS={}' \
  SEED=0 \
  NUM_SEEDS=3 \
  +SAVE_PATH=SAVE_PATH_FOR_CHECKPOINTS/cooperative_team \
  WANDB_MODE=disabled \
  PROJECT=coin_game_coop_team \
  TOTAL_TIMESTEPS=5000000 \
  NUM_ENVS=32 \
  NUM_STEPS=128 \
  TEST_DURING_TRAINING=false \
  +REWARD_MODE=team
```

## Эгоистичная политика

```
python baselines/MAPPO/mappo_rnn.py \
  ENV_NAME=coin_game \
  '~ENV_KWARGS' \
  '+ENV_KWARGS={}' \
  SEED=0 \
  NUM_SEEDS=3 \
  +SAVE_PATH=SAVE_PATH_FOR_CHECKPOINTS/selfish \
  WANDB_MODE=disabled \
  PROJECT=coin_game_selfish \
  TOTAL_TIMESTEPS=5000000 \
  NUM_ENVS=64 \
  NUM_STEPS=128 \
  TEST_DURING_TRAINING=false

```

# Обучение surrogate-модели и подсчет метрик

## Кооперативная политика
```
python pipeline_main.py \
  --project-dir ./coin_game_project_trained_coop \
  --collection-policy python \
  --policy-module-path ./coin_game_trained_policy.py \
  --checkpoint-path SAVE_PATH_FOR_CHECKPOINTS/cooperative_team/mappo/coin_game/CHECKPOINT_NAME.safetensors \
  --num-episodes 1000 \
  --max-steps-per-episode 50 \
  --history-window 5 \
  --tree-max-depth 8 \
  --tree-min-samples-leaf 20 \
  --tree-min-samples-split 40 \
  --rule-test-size 0.25

python compute_behavior_metrics.py \
  --per-agent-csv ./coin_game_project_trained_coop/logs/coin_game_per_agent.csv \
  --episode-csv ./coin_game_project_trained_coop/logs/coin_game_episodes.csv \
  --out-metrics ./coin_game_project_trained_coop/behavior_metrics_coop.csv \
  --out-summary ./coin_game_project_trained_coop/behavior_summary_coop.csv

python summarize_rule_results.py \
  --rules-csv ./coin_game_project_trained_coop/rules/coin_game_rules.csv \
  --top-k 10 \
  --out-top-rules ./coin_game_project_trained_coop/top_rules_coop.csv \
  --out-by-class ./coin_game_project_trained_coop/rules_by_class_coop.csv \
  --out-feature-frequency ./coin_game_project_trained_coop/rule_feature_frequency_coop.csv

python advanced_visualizations_two_coins.py \
  --steps-jsonl ./coin_game_project_trained_coop/logs/coin_game_steps.jsonl \
  --per-agent-csv ./coin_game_project_trained_coop/logs/coin_game_per_agent.csv \
  --features-csv ./coin_game_project_trained_coop/features/coin_game_features.csv \
  --rules-csv ./coin_game_project_trained_coop/rules/coin_game_rules.csv \
  --rule-feature-frequency-csv ./coin_game_project_trained_coop/rule_feature_frequency_coop.csv \
  --out-dir ./coin_game_project_trained_coop/visualizations_two_coins \
  --episode-id 0 \
  --agent 0
```

Все файлы с результатами в `./coin_game_project_trained_coop`

## Эгоистичная политика
```
python pipeline_main.py \
  --project-dir ./coin_game_project_trained_selfish \
  --collection-policy python \
  --policy-module-path ./coin_game_trained_policy.py \
  --checkpoint-path SAVE_PATH_FOR_CHECKPOINTS/selfish/mappo/coin_game/CHECKPOINT_NAME.safetensors \
  --num-episodes 1000 \
  --max-steps-per-episode 50 \
  --history-window 5 \
  --tree-max-depth 8 \
  --tree-min-samples-leaf 20 \
  --tree-min-samples-split 40 \
  --rule-test-size 0.25

python compute_behavior_metrics.py \
  --per-agent-csv ./coin_game_project_trained_selfish/logs/coin_game_per_agent.csv \
  --episode-csv ./coin_game_project_trained_selfish/logs/coin_game_episodes.csv \
  --out-metrics ./coin_game_project_trained_selfish/behavior_metrics.csv \
  --out-summary ./coin_game_project_trained_selfish/behavior_summary.csv

python summarize_rule_results.py \
  --rules-csv ./coin_game_project_trained_selfish/rules/coin_game_rules.csv \
  --top-k 10 \
  --out-top-rules ./coin_game_project_trained_selfish/top_rules.csv \
  --out-by-class ./coin_game_project_trained_selfish/rules_by_class.csv \
  --out-feature-frequency ./coin_game_project_trained_selfish/rule_feature_frequency.csv

python advanced_visualizations_two_coins.py \
  --steps-jsonl ./coin_game_project_trained_selfish/logs/coin_game_steps.jsonl \
  --per-agent-csv ./coin_game_project_trained_selfish/logs/coin_game_per_agent.csv \
  --features-csv ./coin_game_project_trained_selfish/features/coin_game_features.csv \
  --rules-csv ./coin_game_project_trained_selfish/rules/coin_game_rules.csv \
  --rule-feature-frequency-csv ./coin_game_project_trained_selfish/rule_feature_frequency.csv \
  --out-dir ./coin_game_project_trained_selfish/visualizations_two_coins \
  --episode-id 0 \
  --agent 0

```

Все файлы с результатами в `./coin_game_project_trained_selfish`
