# Algorithms
from .algos.d3qn.d3qn import d3qn as d3qn_pytorch
from .algos.d3qn.d3qn_atari import d3qn as d3qn_atari_pytorch
from .algos.ddpg.ddpg import ddpg as ddpg_pytorch
from .algos.ppo.ppo import ppo as ppo_pytorch
from .algos.sac.sac import sac as sac_pytorch
from .algos.sac.sac import test_sac_pytorch
from .algos.td3.td3 import td3 as td3_pytorch
from .algos.td3.td3 import test_td3_pytorch
from .algos.vpg.vpg import vpg as vpg_pytorch
# Loggers
from .utils.logx import EpochLogger, Logger
