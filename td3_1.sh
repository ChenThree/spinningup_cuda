python main_td3.py \
--batch-size 1024 \
--plr 0.001 \
--qlr 0.001 \
--policy-delay 2 \
--polyak 0.995 \
--epochs 100 \
--steps-per-epoch 10000 \
--warmup 5000 \
--eps-decay 100000 \
--update-every 1 \
--validate-episodes 100
