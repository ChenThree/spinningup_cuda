python ../main_ppo.py \
--env DKittyStandRandom-v0 \
--log-dir ../logs-ppo-stand \
--cpu 2 \
--plr 0.0003 \
--vflr 0.001 \
--epochs 2000 \
--steps-per-epoch 10000 \
--repeat-times 40
