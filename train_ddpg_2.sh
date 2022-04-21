python main_ddpg.py \
--env DKittyWalkRandom-v0 \
--output ./checkpoint/ddpg2 \
--batch-size 1024 \
--rate 0.001 \
--prate 0.0001 \
--validate_episodes 100 \
--validate_steps 20000 \
--epsilon_decay 400000 \
--train_iter 1000000
