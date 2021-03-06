python ../main_sac.py \
--mode test \
--resume ../logs-sac-walk-random-lr-0.0001-bs-128-a-auto/pyt_save/model400.pt \
--env DKittyWalkRandomDynamics-v0 \
--log-dir ../logs-sac-walk-random-lr-0.0001-bs-128-a-auto \
--gpu-ids 0 \
--cpu 1 \
--batch-size 128 \
--lr 0.0001 \
--polyak 0.995 \
--epochs 400 \
--replay-size 2000000 \
--steps-per-epoch 10000 \
--warmup 10000 \
--random-steps 20000 \
--update-every 50 \
--validate-episodes 100 \
