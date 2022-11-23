#!/usr/bin/env python3

import os
import argparse
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib
from matplotlib import font_manager


def create_dataframe(directory, **kwargs):
    
    os.chdir(directory)
    path = os.getcwd()

    cycles_mean = []
    snitch_occ_mean = []
    fpsubs_occ_mean = []
    corecc_occ_mean = []

    cycles_std = []
    snitch_occ_std = []
    fpsubs_occ_std = []
    corecc_occ_std = []

    #loops over different dimensions
    for subdir in os.listdir(path):
        dim_path = os.path.join(path, subdir)
        res1 = []
        res2 = []
        res3 = []
        res4 = []
        #loops over different tests
        for subdir in os.listdir(dim_path):
            textfile = open(os.path.join(dim_path, subdir, 'trace_hart_00000000.txt'))
            filetext =  textfile.read()
            res1.append((re.findall(r'cycles\s*[+-]?([0-9]*[.]?[0-9]+)', filetext))[1])
            res2.append((re.findall(r'snitch_occupancy\s*[+-]?([0-9]*[.]?[0-9]+)', filetext))[1])
            res3.append((re.findall(r'fpss_occupancy\s*[+-]?([0-9]*[.]?[0-9]+)', filetext))[1])
            res4.append((re.findall(r'total_ipc\s*[+-]?([0-9]*[.]?[0-9]+)', filetext))[1])
        # mean
        print(res1)
        cycles_mean.append(np.mean(np.array(res1).astype(float)))
        snitch_occ_mean.append(np.mean(np.array(res2).astype(float)))
        fpsubs_occ_mean.append(np.mean(np.array(res3).astype(float)))
        corecc_occ_mean.append(np.mean(np.array(res4).astype(float)))
        # std_deviation
        cycles_std.append(np.std(np.array(res1).astype(float)))
        snitch_occ_std.append(np.std(np.array(res2).astype(float)))
        fpsubs_occ_std.append(np.std(np.array(res3).astype(float)))
        corecc_occ_std.append(np.std(np.array(res4).astype(float)))
    return pd.DataFrame({   'cycles_mean':cycles_mean,
                            'snitch_occ_mean':snitch_occ_mean,
                            'fpsubs_occ_mean':fpsubs_occ_mean,
                            'corecc_occ_mean':corecc_occ_mean,
                            'cycles_std':cycles_std,
                            'snitch_occ_std':snitch_occ_std,
                            'fpsubs_occ_std':fpsubs_occ_std,
                            'corecc_occ_std':corecc_occ_std})



directory = '/scratch2/mbertuletti/snitch/results_softmax'
single_core = create_dataframe(directory)
print(single_core);


dims = np.array([5, 10, 20, 30 , 40, 50]);

# plot:
fig, ax = plt.subplots()
ax.plot(dims, (single_core['cycles_mean']).to_numpy(), linewidth=2.0)
ax.errorbar(dims, (single_core['cycles_mean']).to_numpy(), (single_core['cycles_std']).to_numpy(), fmt='o', linewidth=2, capsize=6)

ax.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(500, 35000), yticks=np.arange(500, 35000, 2000))
plt.grid(True)
plt.show()




