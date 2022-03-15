#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:00:31 2022

@author: rohit
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_log_dir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)