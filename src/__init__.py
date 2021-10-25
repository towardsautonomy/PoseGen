import os
import sys

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.abspath('src'))
sys.path.append(BASE_DIR)
sys.path.append(SRC_DIR)