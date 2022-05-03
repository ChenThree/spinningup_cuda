python main_sac.py \
--env DKittyWalkRandom-v0 \
--log-dir ./logs-sac-walk \
--cpu 1 \
--batch-size 1024 \
--lr 0.001 \
--alpha 0.2 \
--polyak 0.995 \
--epochs 500 \
--steps-per-epoch 10000 \
--warmup 100000 \
--random-steps 300000 \
--update-every 50 \
--validate-episodes 100
