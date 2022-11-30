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

def create_dataframe_single(directory, **kwargs):

    os.chdir(directory)
    path = os.getcwd()

    cycles_mean = []
    cycles_max = []
    cycles_min = []
    cycles_std = []

    snitch_occ_mean = []
    fpsubs_occ_mean = []
    corecc_occ_mean = []

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

        cycles_mean.append(np.mean(np.array(res1).astype(float)))
        cycles_max.append(np.max(np.array(res1).astype(int)))
        cycles_min.append(np.min(np.array(res1).astype(int)))
        cycles_std.append(np.std(np.array(res1).astype(float)))

        # mean
        snitch_occ_mean.append(np.mean(np.array(res2).astype(float)))
        fpsubs_occ_mean.append(np.mean(np.array(res3).astype(float)))
        corecc_occ_mean.append(np.mean(np.array(res4).astype(float)))
        # std_deviation
        snitch_occ_std.append(np.std(np.array(res2).astype(float)))
        fpsubs_occ_std.append(np.std(np.array(res3).astype(float)))
        corecc_occ_std.append(np.std(np.array(res4).astype(float)))
    return pd.DataFrame({   'cycles_mean':cycles_mean,
                            'cycles_max':cycles_max,
                            'cycles_min':cycles_min,
                            'cycles_std':cycles_std,
                            'snitch_occ_mean':snitch_occ_mean,
                            'fpsubs_occ_mean':fpsubs_occ_mean,
                            'corecc_occ_mean':corecc_occ_mean,
                            'snitch_occ_std':snitch_occ_std,
                            'fpsubs_occ_std':fpsubs_occ_std,
                            'corecc_occ_std':corecc_occ_std})

def create_dataframe(directory, dims, **kwargs):
    
    os.chdir(directory)
    path = os.getcwd()
    keys = ['snitch_loads',
            'snitch_stores',
            'fpss_stores',
            'fpss_loads',
            'snitch_avg_load_latency',
            'snitch_occupancy',
            'snitch_fseq_rel_offloads',
            'fseq_yield',
            'fseq_fpu_yield',
            'fpss_section_latency',
            'fpss_avg_fpu_latency',
            'fpss_avg_load_latency',
            'fpss_occupancy',
            'fpss_fpu_occupancy',
            'fpss_fpu_rel_occupancy',
            'cycles',
            'total_ipc']
    metrics = ['max', 'mean', 'min', 'std']

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
        #loops over different tests
        flag = 0
        df_out = pd.DataFrame(0, index=metrics, columns=keys)
        STD = np.empty([0,18])
        for subdir in os.listdir(dim_path):
            # Mean
            df = pd.read_csv(os.path.join(dim_path, subdir, 'results.csv'))
            add = df.loc[df['desc'] == 'mean'].to_numpy()[0][1:]
            df_out.loc['mean'] = np.add(df_out.loc['mean'].to_numpy(), np.divide(add, 10))

            # STD
            STD = np.append(STD, df.loc[df['desc'] == 'mean'].to_numpy()[0][1:])
            # df = pd.read_csv(os.path.join(dim_path, subdir, 'results.csv'))
            # add = df.loc[df['desc'] == 'std'].to_numpy()[0][1:]
            # df_out.loc['std'] = np.add(df_out.loc['std'].to_numpy(), np.divide(add, 10))

            # MAX
            df = pd.read_csv(os.path.join(dim_path, subdir, 'results.csv'))
            prev_max = df_out.loc['max'].to_numpy()
            next_max = df.loc[df['desc'] == 'max'].to_numpy()[0][1:]
            for idx in range(0, len(prev_max)):
                if prev_max[idx] < next_max[idx]:
                    prev_max[idx] = next_max[idx]
            df_out.loc['max'] = prev_max
            # MIN
            df = pd.read_csv(os.path.join(dim_path, subdir, 'results.csv'))
            if flag == 1:
                prev_min = df_out.loc['min'].to_numpy()
                next_min = df.loc[df['desc'] == 'min'].to_numpy()[0][1:]
                for idx in range(0, len(prev_min)):
                    if prev_min[idx] > next_min[idx]:
                        prev_min[idx] = next_min[idx]
                df_out.loc['min'] = prev_min
            else:
                df_out.loc['min'] = df.loc[df['desc'] == 'min'].to_numpy()[0][1:]
                flag = 1

        x = np.reshape(STD, [10,17])
        x = x.astype('float')
        df_out.loc['std'] = x.std(0)

        # mean
        cycles_mean.append(df_out['cycles'].loc['mean'].astype(float))
        snitch_occ_mean.append(df_out['snitch_occupancy'].loc['mean'].astype(float))
        fpsubs_occ_mean.append(df_out['fpss_occupancy'].loc['mean'].astype(float))
        corecc_occ_mean.append(df_out['total_ipc'].loc['mean'].astype(float))
        # std_deviation
        cycles_std.append(df_out['cycles'].loc['std'].astype(float))
        snitch_occ_std.append(df_out['snitch_occupancy'].loc['std'].astype(float))
        fpsubs_occ_std.append(df_out['fpss_occupancy'].loc['std'].astype(float))
        corecc_occ_std.append(df_out['total_ipc'].loc['std'].astype(float))
    return pd.DataFrame({   'cycles_mean':cycles_mean,
                            'snitch_occ_mean':snitch_occ_mean,
                            'fpsubs_occ_mean':fpsubs_occ_mean,
                            'corecc_occ_mean':corecc_occ_mean,
                            'cycles_std':cycles_std,
                            'snitch_occ_std':snitch_occ_std,
                            'fpsubs_occ_std':fpsubs_occ_std,
                            'corecc_occ_std':corecc_occ_std})

directory = '/scratch2/mbertuletti/snitch/results_softmaxdense_mc_axis0'
dimensions = np.array([5, 10, 20, 30 , 40, 50]);
multi_core_dense = create_dataframe(directory, dimensions)

directory = '/scratch2/mbertuletti/snitch/results_softmax_mc_axis0'
dimensions = np.array([5, 10, 20, 30 , 40, 50]);
multi_core_0 = create_dataframe(directory, dimensions)

directory = '/scratch2/mbertuletti/snitch/results_softmax_mc_axis1'
dimensions = np.array([5, 10, 20, 30 , 40, 50]);
multi_core_1 = create_dataframe(directory, dimensions)

###############################################################################

directory = '/scratch2/mbertuletti/snitch/results_softmaxdense_sc_axis0'
dimensions = np.array([5, 10, 20, 30 , 40, 50]);
single_core_dense = create_dataframe_single(directory)
print(single_core_dense)

directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis0'
dimensions = np.array([5, 10, 20, 30 , 40, 50]);
single_core_0= create_dataframe_single(directory)

directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis1'
dimensions = np.array([5, 10, 20, 30 , 40, 50]);
single_core_1 = create_dataframe_single(directory)

###############################################################################

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
fig0, ax0 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('Cycles')
plt.title('SOFTMAX Multi-core')
dims = np.array([5, 10, 20, 30 , 40, 50]);
ax0.plot(dims, (multi_core_dense['cycles_mean']).to_numpy(), marker="o", linewidth=2.0)
ax0.plot(dims, (multi_core_0['cycles_mean']).to_numpy(), marker="o", linewidth=2.0)
ax0.plot(dims, (multi_core_1['cycles_mean']).to_numpy(), marker="o", linewidth=2.0)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax0.legend(['dense', 'axis 0', 'axis 1'], loc='best')
plt.grid(True)
plt.tight_layout()

# plot IPC:
fig1, ax1 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('IPC')
plt.title('Multi-core DENSE')
dims = np.array([5, 10, 20, 30 , 40, 50]);
width = 3
ax1.bar(dims, (multi_core_dense['snitch_occ_mean']).to_numpy(), width, color='tab:blue')
ax1.bar(dims, (multi_core_dense['fpsubs_occ_mean']).to_numpy(), width, bottom=(multi_core_dense['snitch_occ_mean']).to_numpy(), color='tab:red')
ax1.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(0, 1.2), yticks=np.arange(0, 1.2, 0.2))
ax1.legend(['Snitch IPC', 'FSS IPU'], loc='upper left')
plt.axhline(y = 1, color = 'r', linestyle = '--')
plt.grid(True)
plt.tight_layout()

# plot IPC:
fig2, ax2 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('IPC')
plt.title('Multi-core CRS0')
dims = np.array([5, 10, 20, 30 , 40, 50]);
width = 3
ax2.bar(dims, (multi_core_0['snitch_occ_mean']).to_numpy(), width, yerr=3*(multi_core_0['snitch_occ_std']).to_numpy(), color='tab:blue')
ax2.bar(dims, (multi_core_0['fpsubs_occ_mean']).to_numpy(), width, yerr=3*(multi_core_0['fpsubs_occ_std']).to_numpy(), bottom=(multi_core_0['snitch_occ_mean']).to_numpy(), color='tab:red')
ax2.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(0, 1.2), yticks=np.arange(0, 1.2, 0.2))
ax2.legend(['Snitch IPC', 'FSS IPU'], loc='upper left')
plt.axhline(y = 1, color = 'r', linestyle = '--')
plt.grid(True)
plt.tight_layout()

## plot IPC:
fig3, ax3 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('IPC')
plt.title('Multi-core CRS1')
dims = np.array([5, 10, 20, 30 , 40, 50]);
width = 3
ax3.bar(dims, (multi_core_1['snitch_occ_mean']).to_numpy(), width, yerr=3*(multi_core_1['snitch_occ_std']).to_numpy(), color='tab:blue')
ax3.bar(dims, (multi_core_1['fpsubs_occ_mean']).to_numpy(), width, yerr=3*(multi_core_1['fpsubs_occ_std']).to_numpy(), bottom=(multi_core_1['snitch_occ_mean']).to_numpy(), color='tab:red')
ax3.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
       ylim=(0, 1.2), yticks=np.arange(0, 1.2, 0.2))
ax3.legend(['Snitch IPC', 'FSS IPU'], loc='upper left')
plt.axhline(y = 1, color = 'r', linestyle = '--')
plt.grid(True)
plt.tight_layout()



# plot SPEEDUP:
fig4, ax4 = plt.subplots()
plt.xlabel('Input dimension')
plt.ylabel('Speed-up')
plt.title('SOFTMAX Speed-up')
dims = np.array([5, 10, 20, 30 , 40, 50]);
ax4.plot(dims, np.divide((single_core_dense['cycles_mean']).to_numpy(), (multi_core_dense['cycles_mean']).to_numpy()), marker="o", linewidth=2.0)
ax4.plot(dims, np.divide((single_core_0['cycles_mean']).to_numpy(), (multi_core_0['cycles_mean']).to_numpy()), marker="o", linewidth=2.0)
ax4.plot(dims, np.divide((single_core_1['cycles_mean']).to_numpy(), (multi_core_1['cycles_mean']).to_numpy()), marker="o", linewidth=2.0)
ax4.legend(['dense', 'axis 0', 'axis 1'], loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()
