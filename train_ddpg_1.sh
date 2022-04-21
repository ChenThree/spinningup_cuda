python main_ddpg.py \
--output checkpoint1 \
--batch-size 1024 \
--rate 0.001 \
--prate 0.0001 \
--validate_episodes 200 \
--validate_steps 20000 \
--epsilon_decay 200000 \
--train_iter 500000
