python main_ddpg.py \
--env DKittyStandRandom-v0 \
--output ./checkpoint/ddpg1 \
--batch-size 1024 \
--rate 0.001 \
--prate 0.0001 \
--validate_episodes 100 \
--validate_steps 10000 \
--epsilon_decay 200000 \
--train_iter 500000
