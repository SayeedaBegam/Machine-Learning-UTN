import torch
import numpy as np
import random


def seeding(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
