# -*- coding: utf-8 -*-
"""
SI_Bayesian_Optimization_Nylon_Multiple_Starting_Points.py performs sequential 
Bayesian optimization over the experimental conditions contained in 
Data_Nylon.csv using three distinct initial conditions. These are selected
at approximately the 25th, 50th and 75th percentiles of the negative molecular
weight (−MW) ranking. A Gaussian process regression model predicts molecular 
weight (MW), while the expected improvement acquisition function selects the 
next experiment from the remaining experimental candidates.

The experimental variables are: temperature, reaction time and reactant ratio
The target property is: molecular weight (MW)

Sections:
    - Imports
    - Configuration
    - Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
    - Main Script
    - Plots

Author: Rafael Serrano and Dinis Abranches
Last Edit: 2026-10-23
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import warnings
import time
warnings.filterwarnings("ignore",
                        message="pkg_resources is deprecated as an API")

# Specific
import pandas
import numpy
import matplotlib.pyplot as plt
import scipy
import gpflow
from sklearn import preprocessing

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
databasesFolderPath=(r'/path/to/Databases')

# Gaussian-Process (GP) configuration
gpConfig={'kernel':'RBF',
          'useWhiteKernel':False,
          'trainLikelihood':True}

# Bayesian-Optimization (BO) configuration
AF_max_stop=-1 # Negative value disables acquisition-based stopping
maxIter=100

# Define what property serves as the label
label='Mw'

# Parameters for feature and label normalization
featureNorm='Log+bStand' # None,Standardization,MinMax,LogStand,Log+bStand
labelNorm='Standardization' # None,Standardization,MinMax,LogStand,Log+bStand

# =============================================================================
# Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
                           zero; uses a small buffer,
                           (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Default: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='MinMax':
            skScaler=preprocessing.StandardScaler().fit(aux)
        else:
            skScaler=preprocessing.MinMaxScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand': inputArray=numpy.exp(inputArray)
        elif method=='Log+bStand': inputArray=numpy.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

def buildGP(X_Train,Y_Train,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential()
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,
                            noise_variance=10**-5)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           method='L-BFGS-B')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

# =============================================================================
# Main Script
# =============================================================================

for example in (1,2,3):
    # Initiate timer
    ti=time.time()
    # Load data
    Data=pandas.read_csv(os.path.join(databasesFolderPath,'Data_Nylon.csv'),
                         encoding='latin1')
    X_Avail=Data.iloc[:,1:4].to_numpy()
    Y_Avail=-Data.iloc[:,4].to_numpy().reshape(-1,1)
    # Define global maximum in the negative space
    negative_global_max=Y_Avail.max()
    # Rank conditions by negative Mw, from least to most favourable.
    ranked_points=numpy.argsort(Y_Avail[:,0],kind='stable')
    # Select condition at approximately the 25th, 50th or 75th percentile
    start_fraction={1:0.25,2:0.50,3:0.75}[example]
    percentile_label=f'{int(start_fraction*100)}th percentile of -Mw'
    print(f'\nExample {example}: {percentile_label}')
    initial_rank=int(numpy.ceil(start_fraction*len(ranked_points)))-1
    initial_point=int(ranked_points[initial_rank])
    # Print the initial condition for reporting in the Supporting Information.
    print(f'Initial condition:\n'
          f'Temperature: {X_Avail[initial_point,0]} °C\n'
          f'Time: {X_Avail[initial_point,1]} h\n'
          f'Ratio: {X_Avail[initial_point,2]}\n'
          f'Mw: {-Y_Avail[initial_point,0]:.1f} g mol^-1')
    # Define intial BO data point
    X_BO=X_Avail[initial_point,:].reshape(1,-1)
    Y_BO=Y_Avail[initial_point,:].reshape(-1,1)
    # Delete initial point from test set
    X_Avail=numpy.delete(X_Avail,initial_point,axis=0)
    Y_Avail=numpy.delete(Y_Avail,initial_point,axis=0)
    # Fit the feature scaler once on the complete known candidate set
    X_Domain=numpy.vstack((X_BO,X_Avail))
    if featureNorm is not None:
        _,skScaler_X=normalize(X_Domain,method=featureNorm)
    else:
        skScaler_X=None
    # Define standard distribution
    standard=scipy.stats.norm(loc=0,scale=1)
    # Define history containers
    AF_max_hist=[]
    Target_hist=[]
    print("Starting Bayesian Optimization Loop...")
    for n in range(maxIter):
        if X_Avail.shape[0]==0:
            print("Candidate pool exhausted. Stopping optimization.")
            break
        start2=time.time()
        # Fit feature scaler on known finite experimental domain
        if featureNorm is not None:
            X_BO_N,_=normalize(X_BO,method=featureNorm,
                               skScaler=skScaler_X)
            X_Avail_N,_=normalize(X_Avail,method=featureNorm,
                                  skScaler=skScaler_X)
        else:
            X_BO_N=X_BO
            X_Avail_N=X_Avail
        # Refit label normalization using the currently acquired observations
        if labelNorm is not None:
            Y_BO_N,skScaler_Y=normalize(Y_BO,method=labelNorm)
        else:
            Y_BO_N=Y_BO
        # Train GP
        model=buildGP(X_BO_N,Y_BO_N,gpConfig=gpConfig)
        model=model.posterior()
        # Get GP predictions for the remaining candidates
        Y_Avail_Pred_N,STD=gpPredict(model,X_Avail_N)
        # Get best observed value in the normalized maximization space
        best_N=Y_BO_N.max()
        delta=Y_Avail_Pred_N-best_N
        # Compute Expected Improvement
        AF=delta*standard.cdf(delta/STD)+STD*standard.pdf(delta/STD)
        # Get maximum acquisition value and its candidate index
        AF_max=AF.max()
        index=AF.argmax()
        # Check stopping criterion
        if (AF_max<AF_max_stop and n>0) or n==maxIter:
            print("Stopping criterion met.")
            break
        # Add the selected experimental observation to the BO dataset
        X_BO=numpy.append(X_BO,X_Avail[index,:].reshape(1,-1).copy(),axis=0)
        Y_BO=numpy.append(Y_BO,Y_Avail[index,:].reshape(1,-1).copy(),axis=0)
        # Remove the acquired observation from the remaining candidate pool
        X_Avail=numpy.delete(X_Avail,index,axis=0)
        Y_Avail=numpy.delete(Y_Avail,index,axis=0)
        # Get best after adding the observation
        current_best=Y_BO.max()
        # Convert back to physical molecular weight for printing
        print_best=-current_best
        # Print iteration details
        print('\nIteration #'+str(n+1))
        print(' - Current AF max: '+'{:.2f}'.format(AF_max))
        print(' - Current Best: '+'{:.2f}'.format(print_best))
        # Update history
        AF_max_hist.append(AF_max)
        Target_hist.append(current_best)

    # Get best hit
    X_Best=X_BO[Y_BO.argmax(),:].reshape(1,-1)
    Y_Best=-Y_BO.max().reshape(1,1)
    true_global_min=-negative_global_max
    # Check whether the database minimum was reached
    reached_min=(Y_Best[0,0]<=true_global_min)
    status_str='YES' if reached_min else f'NO (Minimum is {true_global_min})'
    print(f'\nOptimization Complete. \nBest Physical Parameters:\n'
          f'Temperature: {X_Best[0,0]} °C\n'
          f'Time: {X_Best[0,1]} hours\n'
          f'Ratio: {X_Best[0,2]}\n'
          f'Resulting {label}: {Y_Best[0,0]:.1f} g mol^-1\n'
          f'Reached Global Minimum: {status_str}')

    # =============================================================================
    # Plots
    # =============================================================================

    # Define the MW history, including the initial observation
    full_Target_hist=[Y_BO[0,0]]+Target_hist
    # AF is undefined before the loop starts
    full_AF_hist=[numpy.nan]+AF_max_hist
    # Convergence plot
    fig,ax1=plt.subplots(figsize=(6,4))
    # Define x-axis
    iterations=range(0,len(full_Target_hist))
    # Plot 1: Acquisition function (left axis, red)
    line1=ax1.plot(iterations,full_AF_hist,'--or',
                  label='Expected Improvement',linewidth=1.5,markersize=4)
    # Plot 2: Best observed molecular weight (right axis, green)
    ax2=ax1.twinx()
    full_Target_array=numpy.array(full_Target_hist)
    line2=ax2.plot(iterations,-full_Target_array,'-s',color='limegreen',
                  label=f'Best Observed {label}',linewidth=1.5,markersize=4)
    # Axis labels
    ax1.set_xlabel('BO Iteration',weight='bold',fontsize=11)
    ax1.set_ylabel('Max. Expected Improvement',weight='bold',fontsize=10,
                   color='red')
    ax2.set_ylabel('Molecular Weight (g mol$^{-1}$)',weight='bold',fontsize=10,
                   color='limegreen')
    ax1.tick_params(axis='y',colors='red')
    ax2.tick_params(axis='y',colors='limegreen')
    # Combined legend
    lines=line1+line2
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc='lower center',bbox_to_anchor=(0.5,1.02),
               ncol=2,frameon=False)
    plt.tight_layout()
    plt.show()

    # Print elapsed time in minutes and seconds
    tf=time.time()
    elapsed=tf-ti
    minutes=int(elapsed//60)
    seconds=elapsed%60
    print(f"Time of execution: {minutes} min {seconds:.2f} s")