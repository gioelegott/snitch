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
            cycles = (re.findall(r'cycles\s*[+-]?([0]?[xX]?[0-9a-fA-F]+]?)', filetext))[1]
            if 'x' in cycles:
                cycles=int(cycles, base=16)
            res1.append(cycles)
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

directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis0'
single_core_0 = create_dataframe(directory)
print(single_core_0);

directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis1'
single_core_1 = create_dataframe(directory)
print(single_core_1);

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot cycles:
fig1, ax1 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('Cycles')
plt.title('SOFTMAX Single-core')

dims = np.array([5, 10, 20, 30 , 40, 50]);
ax1.plot(dims, (single_core_0['cycles_mean']).to_numpy(), marker="o", linewidth=2.0)
ax1.plot(dims, (single_core_1['cycles_mean']).to_numpy(), marker="o", linewidth=2.0)
#ax.errorbar(dims, (single_core['cycles_mean']).to_numpy(), (single_core['cycles_std']).to_numpy(), fmt='o', linewidth=2, capsize=6)

ax1.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(500, 300000), yticks=np.arange(500, 300000, 50000))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1.legend(['axis 0', 'axis 1'], loc='best')

plt.grid(True)
plt.tight_layout()

# plot IPC:
fig2, ax2 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('IPC')
plt.title('SOFTMAX Single-core')

dims = np.array([5, 10, 20, 30 , 40, 50]);
width = 3
ax2.bar(dims, (single_core_0['snitch_occ_mean']).to_numpy(), width, yerr=3*(single_core_0['snitch_occ_std']).to_numpy(), color='tab:blue')
ax2.bar(dims, (single_core_0['fpsubs_occ_mean']).to_numpy(), width, yerr=3*(single_core_0['fpsubs_occ_std']).to_numpy(), bottom=(single_core_0['snitch_occ_mean']).to_numpy(), color='tab:red')

ax2.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(0, 1), yticks=np.arange(0, 1, 0.2))

ax2.legend(['Snitch IPC', 'FSS IPU'], loc='best')

plt.grid(True)
plt.tight_layout()

# plot IPC:
fig3, ax3 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('IPC')
plt.title('SOFTMAX Single-core')

dims = np.array([5, 10, 20, 30 , 40, 50]);
width = 3
ax3.bar(dims, (single_core_1['snitch_occ_mean']).to_numpy(), width, yerr=3*(single_core_1['snitch_occ_std']).to_numpy(), color='tab:blue')
ax3.bar(dims, (single_core_1['fpsubs_occ_mean']).to_numpy(), width, yerr=3*(single_core_1['fpsubs_occ_std']).to_numpy(), bottom=(single_core_1['snitch_occ_mean']).to_numpy(), color='tab:red')

ax3.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(0, 1), yticks=np.arange(0, 1, 0.2))

ax3.legend(['Snitch IPC', 'FSS IPU'], loc='best')

plt.grid(True)
plt.tight_layout()

plt.show()




