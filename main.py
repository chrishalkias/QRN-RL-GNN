# -*- coding: utf-8 -*-

"""
Created on Thu 6 Mar 2025
"""

import experiment as exp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    experiment = exp(model = "DQN", n=7, tau=10000, p_entangle=1, p_swap=1)
    experiment.display_info()