python main_ppo.py \
--env DKittyWalkRandom-v0 \
--log-dir ./logs-ppo-walk \
--cpu 2 \
--plr 0.0003 \
--vflr 0.001 \
--epochs 5000 \
--steps-per-epoch 10000 \
--repeat-times 40
