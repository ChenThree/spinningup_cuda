python main_d3qn.py \
--env LunarLander-v2 \
--log-dir ./logs-d3qn \
--cpu 1 \
--lr 3e-4 \
--epochs 100 \
--gamma 0.99 \
--steps-per-epoch 10000 \
--replay-size 1000000 \
--target-update-interval 2000 \
--batch-size 128 \
--warmup 1000 \
--random-steps 10000 \
--eps-decay 50000
