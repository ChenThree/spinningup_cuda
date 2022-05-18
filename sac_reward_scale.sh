python main_sac.py \
--env DKittyStandRandom-v0 \
--log-dir ./logs-sac-stand-lr-0.0001-bs-128-a-auto-reward-scale \
--gpu-ids 0 \
--cpu 1 \
--batch-size 128 \
--lr 0.0001 \
--polyak 0.995 \
--epochs 200 \
--replay-size 1000000 \
--steps-per-epoch 10000 \
--warmup 10000 \
--random-steps 20000 \
--update-every 50 \
--validate-episodes 100 \
--reward-scale 0.5
