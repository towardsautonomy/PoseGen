import os
import sys

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

# import utils
from .stylegan2 import Discriminator as StyleGAN2_Discriminator
from .stylegan3 import Generator as StyleGAN3_Generator
from .posegen_network import Generator as PoseGen_Generator
from .posegen_network import Discriminator as PoseGen_Discriminator
from .posegen_network import PoseGen
