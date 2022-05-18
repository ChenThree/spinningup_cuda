python ../main_sac.py \
--env DKittyWalkRandom-v0 \
--log-dir ../logs-sac-walk-lr-0.0001-bs-128-a-auto \
--gpu-ids 0 \
--cpu 1 \
--batch-size 128 \
--lr 0.0001 \
--polyak 0.995 \
--epochs 500 \
--replay-size 2000000 \
--steps-per-epoch 10000 \
--warmup 10000 \
--random-steps 20000 \
--update-every 50 \
--validate-episodes 100 \
