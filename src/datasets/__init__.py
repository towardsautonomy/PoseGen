import os
import sys

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.dirname(os.path.abspath('src/datasets'))
sys.path.append(BASE_DIR)
sys.path.append(DATASETS_DIR)