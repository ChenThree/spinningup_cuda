python ../main_d3qn_atari.py \
--env BreakoutNoFrameskip-v4 \
--log-dir ../logs-d3qn-atari-bs128 \
--cpu 1 \
--gpu-ids 0 \
--lr 3e-4 \
--epochs 1000 \
--gamma 0.99 \
--steps-per-epoch 10000 \
--replay-size 500000 \
--target-update-interval 1000 \
--batch-size 128 \
--warmup 1000 \
--random-steps 10000 \
--eps-decay 200000 \
--update-every 50 \
--validate-episodes 100
