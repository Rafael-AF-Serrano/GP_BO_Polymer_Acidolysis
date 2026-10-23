# -*- coding: utf-8 -*-
"""
01_Experimental_Data_Overview.py creates descriptive plots of the experimental
datasets used for the polyurethane and Nylon 6 Bayesian-optimization studies.

For polyurethane, the figure contains:
    - Histograms of hydroxyl number (OH, Y1) and acid value (AV, Y2)
    - OH and AV as functions of temperature, reaction time and reactant ratio

For Nylon 6, the figure contains:
    - A histogram of molecular weight (Mw)
    - Mw as a function of temperature, reaction time and reactant ratio

All experimental observations are displayed. Lines connect the mean response
at each factor level, and the experimentally observed optimum is highlighted.

Sections:
    - Imports
    - Configuration
    - Auxiliary Functions
        . plotHistogram()
        . plotFactorResponse()
        . addPanelLabels()
    - Main Script
    - Polyurethane Data Overview
    - Nylon 6 Data Overview
    - Results Summary

Author: Rafael Serrano and Dinis Abranches
Last Edit: 2026-10-23
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import time

# Specific
import pandas
import numpy
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
databasesFolderPath=(r'/path/to/Databases')

# Figure display configuration
showFigures=True

# Histogram configuration
nBins_PU=4
nBins_Nylon=6

# Plot colors
color_OH='dodgerblue'
color_AV='orange'
color_MW='limegreen'
color_Optimum='tomato'
color_Data='silver'

# =============================================================================
# Auxiliary Functions
# =============================================================================

def plotHistogram(ax,Y,nBins,color,bestValue,bestLabel,title,yLabel):
    """
    plotHistogram() plots a horizontal response histogram and marks the
    experimentally observed optimum.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object
        Axes on which the histogram is plotted.
    Y : numpy array
        Experimental response values.
    nBins : int
        Number of histogram bins.
    color : string
        Color used for the histogram bars.
    bestValue : float
        Experimentally observed optimum to mark with a horizontal line.
    bestLabel : string
        Legend label used for the optimum line.
    title : string
        Title of the subplot.
    yLabel : string
        Label of the response axis.

    Returns
    -------
    None.

    """
    ax.hist(Y,bins=nBins,orientation='horizontal',color=color,alpha=0.75,
            edgecolor='black',linewidth=0.8)
    ax.axhline(bestValue,color=color_Optimum,linestyle='--',linewidth=2.5,
               label=bestLabel)
    ax.plot(numpy.zeros(Y.shape),Y,'|',color='black',markersize=7,
            markeredgewidth=0.8,clip_on=False)
    ax.set_title(title,weight='bold',fontsize=10)
    ax.set_xlabel('Frequency',fontsize=11)
    ax.set_ylabel(yLabel,fontsize=11)
    ax.grid(axis='x',linestyle=':',alpha=0.5)
    ax.legend(frameon=False,fontsize=9,loc='upper right',
    bbox_to_anchor=(0.98, 1.02))
    

def plotFactorResponse(ax,X,Y,bestIndex,color,xLabel,yLabel=None):
    """
    plotFactorResponse() plots all experimental observations, the mean
    response at each factor level and the experimentally observed optimum.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object
        Axes on which the response data are plotted.
    X : numpy array
        Experimental values of one variable.
    Y : numpy array
        Experimental response values corresponding to X.
    bestIndex : int
        Index of the experimentally observed optimum in X and Y.
    color : string
        Color used for the factor-level mean line.
    xLabel : string
        Label of the process-variable axis.
    yLabel : string or None, optional
        Label of the response axis. If None, no response-axis label is added.
        Default: None

    Returns
    -------
    None.

    """
    levels=numpy.sort(numpy.unique(X))
    levelMeans=numpy.array([Y[X==level].mean() for level in levels])
    ax.scatter(X,Y,s=38,color=color_Data,edgecolor='black',linewidth=0.6,
               zorder=2,label='Experimental data')
    ax.plot(levels,levelMeans,'-s',color=color,linewidth=1.6,markersize=5,
            markeredgecolor='black',markeredgewidth=0.5,zorder=3,
            label='Mean at each level')
    ax.scatter(X[bestIndex],Y[bestIndex],s=150,marker='*',
               color=color_Optimum,edgecolor='black',linewidth=0.7,zorder=4,
               label='Observed optimum')
    ax.set_xlabel(xLabel,fontsize=11)
    if yLabel is not None:
        ax.set_ylabel(yLabel,fontsize=11)
    ax.set_xticks(levels)
    ax.grid(linestyle=':',alpha=0.5)

def addPanelLabels(axes):
    """
    addPanelLabels() adds sequential alphabetical labels to a collection of
    subplot axes.

    Parameters
    ----------
    axes : array-like of matplotlib.axes.Axes objects
        Axes to label in row-major order.

    Returns
    -------
    None.

    """
    labels='abcdefghijkl'
    for i,ax in enumerate(numpy.asarray(axes).ravel()):
        ax.text(0.02,0.99,f'({labels[i]})',transform=ax.transAxes,
                ha='left',va='top',weight='bold',fontsize=9)

# =============================================================================
# Main script
# =============================================================================

# Initiate timer
ti=time.time()
# Load polyurethane data
Data_PU=pandas.read_csv(os.path.join(databasesFolderPath,
                                     'Data_Polyurethane.csv'))
# Retrieve temperature, time and ratio
T_PU=Data_PU.iloc[:,2].to_numpy()
t_PU=Data_PU.iloc[:,3].to_numpy()
R_PU=Data_PU.iloc[:,4].to_numpy()
# Get Hydroxyl number
Y1=Data_PU.iloc[:,6].to_numpy()
# Get Acid value
Y2=Data_PU.iloc[:,5].to_numpy()
# Define the observed polyurethane optima
indexBest_Y1=int(Y1.argmax())
indexBest_Y2=int(Y2.argmin())
# Load Nylon 6 data
# Try Latin-1 (ISO-8859-1) or Windows-1252
Data_Nylon = pandas.read_csv(
    os.path.join(databasesFolderPath, 'Data_Nylon.csv'), 
    encoding='latin1'
)
# Retrieve temperature, time and ratio
T_Nylon=Data_Nylon.iloc[:,1].to_numpy()
t_Nylon=Data_Nylon.iloc[:,2].to_numpy()
R_Nylon=Data_Nylon.iloc[:,3].to_numpy()
# Get molecular weight 
MW_Nylon=Data_Nylon['Molecular Weight'].to_numpy()
# Define the observed Nylon 6 optimum
indexBest_Nylon=int(MW_Nylon.argmin())

# =============================================================================
# Nylon 6 Data Overview
# =============================================================================

fig_Nylon,axes_Nylon=plt.subplots(1,4,figsize=(14,3.4))
# Plot histogram of MW values
plotHistogram(axes_Nylon[0],MW_Nylon,nBins_Nylon,color_MW,
              MW_Nylon[indexBest_Nylon],
              f'Minimum MW = {MW_Nylon[indexBest_Nylon]:.1f}',
              'Nylon 6 Mw distribution',
              'Molecular weight (g mol$^{-1}$)')
# Plot temperature data
plotFactorResponse(axes_Nylon[1],T_Nylon,MW_Nylon,indexBest_Nylon,color_MW,
                   'Temperature (°C)')
# Plot time data
plotFactorResponse(axes_Nylon[2],t_Nylon,MW_Nylon,indexBest_Nylon,color_MW,
                   'Reaction time (h)')
# Plot reactant ratio data
plotFactorResponse(axes_Nylon[3],R_Nylon,MW_Nylon,indexBest_Nylon,color_MW,
                   'Polymer to acid ratio')
for j,factorName in enumerate(['Temperature','Reaction time',
                               'Reactant ratio']):
    axes_Nylon[j+1].set_title(f'Mw vs {factorName.lower()}',weight='bold',
                              fontsize=11)
# Use one response-axis scale across the Nylon 6 panels
MW_range=MW_Nylon.max()-MW_Nylon.min()
for ax in axes_Nylon:
    ax.set_ylim(MW_Nylon.min()-0.03*MW_range,
                MW_Nylon.max()+0.12*MW_range)
handles,labels=axes_Nylon[1].get_legend_handles_labels()
fig_Nylon.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,1.03),
                 ncol=3,frameon=False,fontsize=11,markerscale=1.2)
addPanelLabels(axes_Nylon)
fig_Nylon.tight_layout(rect=(0,0,1,0.91))

# =============================================================================
# Polyurethane Data Overview
# =============================================================================

fig_PU,axes_PU=plt.subplots(2,4,figsize=(14,6.5))
# Plot histogram of OH values
plotHistogram(axes_PU[0,0],Y1,nBins_PU,color_OH,Y1[indexBest_Y1],
              f'Maximum OH = {Y1[indexBest_Y1]:.1f}',
              'OH distribution',
              'Hydroxyl number (mg KOH g$^{-1}$)')
# Plot temperature data
plotFactorResponse(axes_PU[0,1],T_PU,Y1,indexBest_Y1,color_OH,
                   'Temperature (°C)')
# Plot time data
plotFactorResponse(axes_PU[0,2],t_PU,Y1,indexBest_Y1,color_OH,
                   'Reaction time (h)')
# Plot reactant ratio data
plotFactorResponse(axes_PU[0,3],R_PU,Y1,indexBest_Y1,color_OH,
                   'Polymer to acid ratio')
# Plot histogram of AV values
plotHistogram(axes_PU[1,0],Y2,nBins_PU,color_AV,Y2[indexBest_Y2],
              f'Minimum AV = {Y2[indexBest_Y2]:.1f}',
              'AV distribution',
              'Acid value (mg KOH g$^{-1}$)')
# Plot temperature data
plotFactorResponse(axes_PU[1,1],T_PU,Y2,indexBest_Y2,color_AV,
                   'Temperature (°C)')
# Plot time data
plotFactorResponse(axes_PU[1,2],t_PU,Y2,indexBest_Y2,color_AV,
                   'Reaction time (h)')
# Plot reactant ratio data
plotFactorResponse(axes_PU[1,3],R_PU,Y2,indexBest_Y2,color_AV,
                   'Polymer to acid ratio')
# Add panel titles to the factor plots
for j,factorName in enumerate(['Temperature','Reaction time',
                               'Reactant ratio']):
    axes_PU[0,j+1].set_title(f'OH vs {factorName.lower()}',weight='bold',
                             fontsize=11)
    axes_PU[1,j+1].set_title(f'AV vs {factorName.lower()}',weight='bold',
                             fontsize=11)
# Use the same response-axis limits within each row
for ax in axes_PU[0,:]:
    ax.set_ylim(Y1.min()-0.8,Y1.max()+1.0)
for ax in axes_PU[1,:]:
    ax.set_ylim(Y2.min()-0.2,Y2.max()+0.25)
# Add one common legend for the factor plots
handles,labels=axes_PU[0,1].get_legend_handles_labels()
fig_PU.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,1.01),
              ncol=3,frameon=False,fontsize=11,markerscale=1.2)
addPanelLabels(axes_PU)
fig_PU.tight_layout(rect=(0,0,1,0.95))

# Show figures if requested
if showFigures:
    plt.show()
else:
    plt.close(fig_PU)
    plt.close(fig_Nylon)

# =============================================================================
# Results Summary
# =============================================================================

# Print summary of the experimentally observed optima
print(f'Polyurethane data points: {len(Data_PU)}\n'
      f'Maximum OH: {Y1[indexBest_Y1]:.1f} mg KOH g^-1 at '
      f'{T_PU[indexBest_Y1]:.0f} °C, {t_PU[indexBest_Y1]:.1f} h, '
      f'ratio {R_PU[indexBest_Y1]:.1f}\n'
      f'Minimum AV: {Y2[indexBest_Y2]:.1f} mg KOH g^-1 at '
      f'{T_PU[indexBest_Y2]:.0f} °C, {t_PU[indexBest_Y2]:.1f} h, '
      f'ratio {R_PU[indexBest_Y2]:.1f}\n')
print(f'Nylon 6 data points: {len(Data_Nylon)}\n'
      f'Minimum Mw: {MW_Nylon[indexBest_Nylon]:.1f} g mol^-1 at '
      f'{T_Nylon[indexBest_Nylon]:.0f} °C, '
      f'{t_Nylon[indexBest_Nylon]:.1f} h, '
      f'ratio {R_Nylon[indexBest_Nylon]:.1f}')

# Print elapsed time
tf=time.time()
elapsed=tf-ti
print(f'Time of execution: {elapsed:.2f} s')

