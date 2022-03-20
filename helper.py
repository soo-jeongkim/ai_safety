import sys
import datetime
import random
import itertools

import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from collections import deque

from IPython.display import clear_output

from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ai_safety_gridworlds.environments.safe_interruptibility import (
    SafeInterruptibilityEnvironment, 
    GAME_ART, 
    Actions
)

from ai_safety_gridworlds.environments.shared.rl import environment

def get_new_env(level=0, interruption_probability=0.5):
    env = SafeInterruptibilityEnvironment(
        level=level,
        interruption_probability=interruption_probability,
    )
    return env

# Action definitions
valid_actions = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "no op",
    5: "quit"
}
valid_action_strings = {v[0]: k for k, v in valid_actions.items()}

# State representation
state_num2str = {v: k for k, v in get_new_env()._value_mapping.items()}

def print_game(game_board_status):
    """Print the state of the game to view"""
    for row_string_representation in game_board_status:
        print(row_string_representation)

def convert_board_num2str(num_reps):
    """Converts state from number to string representation"""
    representation = []
    for row in num_reps:
        representation_row = []
        for item in row:
            representation_row.append(state_num2str[item])
        representation.append(" ".join(representation_row))
    return representation

LEVEL = 0
print(f"This is what the game state looks like for level {LEVEL}:")
print_game(GAME_ART[LEVEL])