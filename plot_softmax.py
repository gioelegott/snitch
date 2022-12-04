#!/usr/bin/env python3

import os
import argparse
import re

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

def create_dataframe(directory, stat, metric, **kwargs):
    os.chdir(directory)
    path = os.getcwd()
    df_out = pd.DataFrame()
    #loops over different dimensions
    for dimdir in os.listdir(path):
        dim_path = os.path.join(path, dimdir)
        #loops over different tests
        distribution = []
        for testdir in os.listdir(dim_path):
            df = pd.read_csv(os.path.join(dim_path, testdir, 'results.csv'))
            distribution.extend(df.loc[df['desc'] == stat, metric].to_numpy())
        df_out[dimdir] = distribution

    return df_out

def mean_confidence_interval(data, confidence):
    m = data.apply(np.mean, axis=0)
    se = data.apply(stats.sem)
    n = len(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, se, h

def mean_confidence_interval_ratio(data1, data2, confidence):
    m=[]
    se=[]
    h=[]
    for column in data1:
        div=[]
        for i in range(0, len(data1[column])):
            for j in range(0, len(data2[column])):
                x = data1.loc[i].at[column]/data2.loc[j].at[column]
                div.append(x)
        n = len(div)
        m.append(np.mean(div))
        se.append(stats.sem(div))
        h.append(stats.sem(div) * stats.t.ppf((1 + confidence) / 2., n-1))
    return m, se, h


directory = '/scratch2/mbertuletti/snitch/results_softmaxdense_sc_axis0'
sc_dense_cycles = create_dataframe(directory, 'mean', 'cycles')
sc_dense_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
sc_dense_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')

directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis0'
sc_csr0_cycles = create_dataframe(directory, 'mean', 'cycles')
sc_csr0_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
sc_csr0_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')

directory = '/scratch2/mbertuletti/snitch/results_softmax_sc_axis1'
sc_csr1_cycles = create_dataframe(directory, 'mean', 'cycles')
sc_csr1_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
sc_csr1_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')

directory = '/scratch2/mbertuletti/snitch/results_softmaxdense_mc_axis0'
mc_dense_cycles = create_dataframe(directory, 'mean', 'cycles')
mc_dense_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
mc_dense_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')

directory = '/scratch2/mbertuletti/snitch/results_softmax_mc_axis0'
mc_csr0_cycles = create_dataframe(directory, 'mean', 'cycles')
mc_csr0_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
mc_csr0_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')

directory = '/scratch2/mbertuletti/snitch/results_softmax_mc_axis1'
mc_csr1_cycles = create_dataframe(directory, 'mean', 'cycles')
mc_csr1_coreipc = create_dataframe(directory, 'mean', 'snitch_occupancy')
mc_csr1_fpssipc = create_dataframe(directory, 'mean', 'fpss_occupancy')

#############################################################################
# PLOT FEATURES

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fe = font_manager.FontEntry(
    fname='/home/mbertuletti/.local/share/fonts/helvetica/Helvetica.otf',
    name='FreeSans'
)
font_manager.fontManager.ttflist.insert(0, fe)
mpl.rcParams['font.family'] = fe.name


#############################################################################
# CYCLES PLOT

# plot cycles:
fig0 = plt.figure(figsize=(15, 6))
ax0= plt.subplot2grid((2, 2), (0, 0), rowspan=1)
ax1= plt.subplot2grid((2, 2), (1, 0), rowspan=1)
ax2= plt.subplot2grid((2, 2), (0, 1), rowspan=2)
dims = np.array([10, 20, 30 , 40, 50]);
cmap = mcp.gen_color(cmap="Spectral_r",n=6)

##LINES
#m, se, h = mean_confidence_interval(sc_dense_cycles, 0.95)
#l1, = ax0.plot(dims, m[1:], marker="o", linewidth=2.0, color=cmap[0])
#m, se, h = mean_confidence_interval(sc_csr0_cycles, 0.95)
#l2, = ax0.plot(dims, m[1:], marker="x", linewidth=2.0, color=cmap[1])
#plt.fill_between(dims, m[1:] - h[1:], m[1:] + h[1:], color='k', alpha=0.2)
#m, se, h = mean_confidence_interval(sc_csr1_cycles, 0.95)
#l3, = ax0.plot(dims, m[1:], marker="x", linewidth=2.0, color=cmap[2])
#plt.fill_between(dims, m[1:] - h[1:], m[1:] + h[1:], color='k', alpha=0.2)
#BARS
width = 2
m, se, h = mean_confidence_interval(sc_dense_cycles, 0.95)
l1 = ax0.bar(dims-width, m[1:], width, color=cmap[0], edgecolor='k')
m, se, h = mean_confidence_interval(sc_csr0_cycles, 0.95)
l2 = ax0.bar(dims, m[1:], width, color=cmap[1], edgecolor='k')
ax0.errorbar(dims, m[1:], 2*np.array(h[1:]), fmt='none', ecolor='r', elinewidth=2)
m, se, h = mean_confidence_interval(sc_csr1_cycles, 0.95)
l3 = ax0.bar(dims+width, m[1:], width, color=cmap[2], edgecolor='k')
ax0.errorbar(dims+width, m[1:], 2*np.array(h[1:]), fmt='none', ecolor='r', elinewidth=2)
ax0.legend([l1, l2, l3], ['dense', 'CRS axis-0', 'CRS axis-1'], loc='upper left', facecolor='white', framealpha=1)
ax0.set_xlabel('Input dimension')
ax0.set_ylabel('Cycles')
ax0.set_title('SOFTMAX Single-core')
ax0.set(ylim=(0, 300000), yticks=np.arange(0, 300000, 50000))
ax0.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)


ax0.grid(True)
plt.tight_layout()

##LINES
#m, se, h = mean_confidence_interval(mc_dense_cycles, 0.95)
#l1, = ax1.plot(dims, m[1:], marker="o", linewidth=2.0, color=cmap[0])
#m, se, h = mean_confidence_interval(mc_csr0_cycles, 0.95)
#l2, = ax1.plot(dims, m[1:], marker="x", linewidth=2.0, color=cmap[1])
#plt.fill_between(dims, m[1:] - h[1:], m[1:] + h[1:], color='k', alpha=0.2)
#m, se, h = mean_confidence_interval(mc_csr1_cycles, 0.95)
#l3, = ax1.plot(dims, m[1:], marker="x", linewidth=2.0, color=cmap[2])
#plt.fill_between(dims, m[1:] - h[1:], m[1:] + h[1:], color='k', alpha=0.2)
#BARS
width = 2
m, se, h = mean_confidence_interval(mc_dense_cycles, 0.95)
l1 = ax1.bar(dims-width, m[1:], width, color=cmap[0], edgecolor='k')
m, se, h = mean_confidence_interval(mc_csr0_cycles, 0.95)
l2 = ax1.bar(dims, m[1:], width, color=cmap[1], edgecolor='k')
ax1.errorbar(dims, m[1:], 2*np.array(h[1:]), fmt='none', ecolor='r', elinewidth=2)
m, se, h = mean_confidence_interval(mc_csr1_cycles, 0.95)
l3 = ax1.bar(dims+width, m[1:], width, color=cmap[2], edgecolor='k')
ax1.errorbar(dims+width, m[1:], 2*np.array(h[1:]), fmt='none', ecolor='r', elinewidth=2)
ax1.legend([l1, l2, l3], ['dense', 'CRS axis-0', 'CRS axis-1'], loc='upper left', facecolor='white', framealpha=1)
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Cycles')
ax1.set_title('SOFTMAX 8-cores')
ax1.set(ylim=(0, 50000), yticks=np.arange(0, 50000, 15000))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
ax1.grid(True)
plt.tight_layout()

##LINES
#m, se, h = mean_confidence_interval_ratio(sc_dense_cycles, mc_dense_cycles, 0.95)
#l1, = ax2.plot(dims, m[1:], marker="o", linewidth=2.0, color=cmap[0])
#m, se, h = mean_confidence_interval_ratio(sc_csr0_cycles, mc_csr0_cycles, 0.95)
#l2, = ax2.plot(dims, m[1:], marker="x", linewidth=2.0, color=cmap[1])
#plt.fill_between(dims, np.array(m[1:]) - np.array(h[1:]), m[1:] + np.array(h[1:]), color='k', alpha=0.2)
#m, se, h = mean_confidence_interval_ratio(sc_csr1_cycles, mc_csr1_cycles, 0.95)
#l3, = ax2.plot(dims, m[1:], marker="x", linewidth=2.0, color=cmap[2])
#plt.fill_between(dims, np.array(m[1:]) - np.array(h[1:]), np.array(m[1:]) + np.array(h[1:]), color='k', alpha=0.2)
#BARS
width = 2
m, se, h = mean_confidence_interval_ratio(sc_dense_cycles, mc_dense_cycles, 0.95)
l1 = ax2.bar(dims-width, m[1:], width, color=cmap[0], edgecolor='k')
m, se, h = mean_confidence_interval_ratio(sc_csr0_cycles, mc_csr0_cycles, 0.95)
l2 = ax2.bar(dims, m[1:], width, color=cmap[1], edgecolor='k')
ax2.errorbar(dims, m[1:], 2*np.array(h[1:]), fmt='none', ecolor='r', elinewidth=2)
m, se, h = mean_confidence_interval_ratio(sc_csr1_cycles, mc_csr1_cycles, 0.95)
l3 = ax2.bar(dims+width, m[1:], width, color=cmap[2], edgecolor='k')
ax2.errorbar(dims+width, m[1:], 2*np.array(h[1:]), fmt='none', ecolor='r', elinewidth=2)
ax2.legend([l1, l2, l3], ['dense', 'CRS axis-0', 'CRS axis-1'], loc='upper left', facecolor='white', framealpha=1)
ax2.set_xlabel('Input dimension')
ax2.set_ylabel('Speed-up')
ax2.set_title('SOFTMAX Speed-UP')
ax2.set(ylim=(0,10), yticks=np.arange(0, 10, 1))
ax2.text(42, 8.2, 'IDEAL Speed-UP', color='r', fontsize=MEDIUM_SIZE)
ax2.axhline(y = 8, color = 'r', linestyle = '--')
ax2.grid(True)
plt.tight_layout()

#############################################################################
# IPC PLOT

fig1 = plt.figure(figsize=(11, 6))
ax0= plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1= plt.subplot2grid((2, 2), (1, 0), colspan=2)
dims = np.array([10, 20, 30 , 40, 50]);
cmap = mcp.gen_color(cmap="RdBu",n=6)
cmap=cmap[2:4]
patterns = [ " ", "/" , ".", "*" ]

ax0.bar(dims-width, sc_dense_coreipc.apply(np.mean, axis=0)[1:], width,
        color=cmap[0], hatch=patterns[0], edgecolor='k')
ax0.bar(dims-width, sc_dense_fpssipc.apply(np.mean, axis=0)[1:], width,
        bottom=sc_dense_coreipc.apply(np.mean, axis=0)[1:],
        color=cmap[1], hatch=patterns[0], edgecolor='k')
ax0.bar(dims, sc_csr0_coreipc.apply(np.mean, axis=0)[1:], width,
        yerr=3*sc_csr0_coreipc.apply(np.std, axis=0)[1:],
        color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims, sc_csr0_fpssipc.apply(np.mean, axis=0)[1:], width,
        yerr=sc_csr0_fpssipc.apply(np.std, axis=0)[1:], bottom=sc_csr0_coreipc.apply(np.mean, axis=0)[1:],
        color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims+width, sc_csr1_coreipc.apply(np.mean, axis=0)[1:], width,
        yerr=3*sc_csr1_coreipc.apply(np.std, axis=0)[1:],
        color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax0.bar(dims+width, sc_csr1_fpssipc.apply(np.mean, axis=0)[1:], width,
        yerr=3*sc_csr1_fpssipc.apply(np.std, axis=0)[1:], bottom=sc_csr1_coreipc.apply(np.mean, axis=0)[1:],
        color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax0.legend( [plt.bar([0], [0], color=cmap[0], edgecolor='k'),
             plt.bar([0], [0], color=cmap[1], edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[1]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[2])],
            ["INT-core IPC", "FP-SS IPC","Dense", "CRS axis-0", "CRS axis-1"],
            loc='upper right', bbox_to_anchor=(1.3, 1), facecolor='white', framealpha=1)
ax0.set(ylim=(0,1.2), yticks=np.arange(0, 1.3, 0.2))
ax0.text(42, 1.05, 'IDEAL Occupation', color='r', fontsize=MEDIUM_SIZE)
ax0.axhline(y = 1, color = 'r', linestyle = '--')
ax0.set_xlabel('Input dimension')
ax0.set_ylabel('Cycles')
ax0.set_title('Occupation Single-core')
ax0.grid(True)
plt.tight_layout()

ax1.bar(dims-width, mc_dense_coreipc.apply(np.mean, axis=0)[1:], width,
        color=cmap[0], hatch=patterns[0], edgecolor='k')
ax1.bar(dims-width, mc_dense_fpssipc.apply(np.mean, axis=0)[1:], width,
        bottom=mc_dense_coreipc.apply(np.mean, axis=0)[1:],
        color=cmap[1], hatch=patterns[0], edgecolor='k')
ax1.bar(dims, mc_csr0_coreipc.apply(np.mean, axis=0)[1:], width,
        yerr=3*mc_csr0_coreipc.apply(np.std, axis=0)[1:],
        color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims, mc_csr0_fpssipc.apply(np.mean, axis=0)[1:], width,
        yerr=mc_csr0_fpssipc.apply(np.std, axis=0)[1:], bottom=mc_csr0_coreipc.apply(np.mean, axis=0)[1:],
        color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims+width, mc_csr1_coreipc.apply(np.mean, axis=0)[1:], width,
        yerr=3*mc_csr1_coreipc.apply(np.std, axis=0)[1:],
        color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
ax1.bar(dims+width, mc_csr1_fpssipc.apply(np.mean, axis=0)[1:], width,
        yerr=3*mc_csr1_fpssipc.apply(np.std, axis=0)[1:], bottom=mc_csr1_coreipc.apply(np.mean, axis=0)[1:],
        color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

ax1.legend( [plt.bar([0], [0], color=cmap[0], edgecolor='k'),
             plt.bar([0], [0], color=cmap[1], edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k'),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[1]),
             plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[2])],
            ["INT-core IPC", "FP-SS IPC","Dense", "CRS axis-0", "CRS axis-1"],
            loc='upper right', bbox_to_anchor=(1.3, 1), facecolor='white', framealpha=1)
ax1.set(ylim=(0,1.2), yticks=np.arange(0, 1.3, 0.2))
ax1.set(xlim=(5,55))
ax1.text(42, 1.05, 'IDEAL Occupation', color='r', fontsize=MEDIUM_SIZE)
ax1.axhline(y = 1, color = 'r', linestyle = '--')
ax1.set_xlabel('Input dimension')
ax1.set_ylabel('Cycles')
ax1.set_title('Occupation 8-cores')
ax1.grid(True)
plt.tight_layout()


















## plot IPC:
#fig1, ax1 = plt.subplots()
#plt.xlabel('Input dimension')
#plt.ylabel('IPC')
#plt.title('Single-core DENSE')
#dims = np.array([5, 10, 20, 30 , 40, 50]);
#width = 3
#ax1.bar(dims, (single_core_0dense['snitch_occ_mean']).to_numpy(), width, color='tab:blue')
#ax1.bar(dims, (single_core_0dense['fpsubs_occ_mean']).to_numpy(), width, bottom=(single_core_0dense['snitch_occ_mean']).to_numpy(), color='tab:red')
#ax1.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
#       ylim=(0, 1.2), yticks=np.arange(0, 1.2, 0.2))
#ax1.legend(['Snitch IPC', 'FSS IPU'], loc='upper left')
#plt.axhline(y = 1, color = 'r', linestyle = '--')
#plt.grid(True)
#plt.tight_layout()

## plot IPC:
#fig2, ax2 = plt.subplots()
#plt.xlabel('Input dimension')
#plt.ylabel('IPC')
#plt.title('Single-core CRS0')
#dims = np.array([5, 10, 20, 30 , 40, 50]);
#width = 3
#ax2.bar(dims, (single_core_0['snitch_occ_mean']).to_numpy(), width, yerr=3*(single_core_0['snitch_occ_std']).to_numpy(), color='tab:blue')
#ax2.bar(dims, (single_core_0['fpsubs_occ_mean']).to_numpy(), width, yerr=3*(single_core_0['fpsubs_occ_std']).to_numpy(), bottom=(single_core_0['snitch_occ_mean']).to_numpy(), color='tab:red')
#ax2.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
#       ylim=(0, 1.2), yticks=np.arange(0, 1.2, 0.2))
#ax2.legend(['Snitch IPC', 'FSS IPU'], loc='upper left')
#plt.axhline(y = 1, color = 'r', linestyle = '--')
#plt.grid(True)
#plt.tight_layout()

## plot IPC:
#fig3, ax3 = plt.subplots()
#plt.xlabel('Input dimension')
#plt.ylabel('IPC')
#plt.title('Single-core CRS1')
#dims = np.array([5, 10, 20, 30 , 40, 50]);
#width = 3
#ax3.bar(dims, (single_core_1['snitch_occ_mean']).to_numpy(), width, yerr=3*(single_core_1['snitch_occ_std']).to_numpy(), color='tab:blue')
#ax3.bar(dims, (single_core_1['fpsubs_occ_mean']).to_numpy(), width, yerr=3*(single_core_1['fpsubs_occ_std']).to_numpy(), bottom=(single_core_1['snitch_occ_mean']).to_numpy(), color='tab:red')
#ax3.set(xlim=(0, 60), xticks=np.arange(0, 60, 10),
#       ylim=(0, 1.2), yticks=np.arange(0, 1.2, 0.2))
#ax3.legend(['Snitch IPC', 'FSS IPU'], loc='upper left')
#plt.axhline(y = 1, color = 'r', linestyle = '--')
#plt.grid(True)
#plt.tight_layout()



plt.show()




