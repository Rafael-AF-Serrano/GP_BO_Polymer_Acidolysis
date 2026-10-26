# -*- coding: utf-8 -*-
"""
03_Multi_Bayesian_Optimization_Polyurethane_EHVI.py performs sequential
multi-objective Bayesian optimization over the experimental conditions
contained in Data_Polyurethane.csv. It uses two Gaussian-process regression
models and exact two-objective expected hypervolume improvement (EHVI) to
optimize hydroxyl number (OH) and acid value (AV) simultaneously.

The objectives are:
    - Maximize hydroxyl number (OH)
    - Minimize acid value (AV), represented internally as −AV

The experimental variables are:
    Feature 0: Temperature
    Feature 1: Reaction time
    Feature 2: Reactant ratio
    Objective 0: Hydroxyl number
    Objective 1: Negative acid value

The remaining measured conditions form the retrospective BO candidate set.

Sections:
    - Imports
    - Configuration
    - Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
        . calculate_EHVI()
    - Main Script
    - Observed Pareto Analysis
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
EHVI_max_stop=-1 # A negative value disables acquisition-based stopping
maxIter=100

# Equal weights used only to select the poor initial condition
w_OH=0.5
w_AV=0.5

# EHVI reference-point configuration
referenceMargin=0.10

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

def calculate_EHVI(Y_Train_N1,Y_Train_N2,Y_Test_Pred_N1,STD_Y_Test_Pred_N1,
        Y_Test_Pred_N2,STD_Y_Test_Pred_N2,ref_point=None):
    """
    calculate_EHVI() calculates the exact expected hypervolume improvement
    (EHVI) for two objectives and returns the index of the best candidate.

    Parameters
    ----------
    Y_Train_N1, Y_Train_N2 : numpy arrays
        Observed values of the two objectives. Both objectives must be written
        as maximization objectives.
    Y_Test_Pred_N1, Y_Test_Pred_N2 : numpy arrays
        GP predictive means for the available candidates.
    STD_Y_Test_Pred_N1, STD_Y_Test_Pred_N2 : numpy arrays
        GP predictive standard deviations for the available candidates.
    ref_point : numpy array
        Fixed two-objective reference point dominated by all points in the
        current Pareto set.

    Returns
    -------
    index : int
        Index of the candidate with maximum EHVI.
    EHVI : numpy array
        EHVI calculated for every available candidate.

    Notes
    -----
    The calculation follows the exact two-objective formulation of Yang et al.
    and assumes independent Gaussian predictive distributions.
    """
    # Prepare the observed objective values
    Y1=numpy.asarray(Y_Train_N1,dtype=float).ravel()
    Y2=numpy.asarray(Y_Train_N2,dtype=float).ravel()
    if Y1.size==0 or Y1.size!=Y2.size:
        raise ValueError('Training objectives must have equal lengths.')
    Y=numpy.column_stack((Y1,Y2))
    # Create non-dominated set P
    dominated=numpy.array([
        numpy.any(numpy.all(Y>=p,axis=1) & numpy.any(Y>p,axis=1))
        for p in Y
    ],dtype=bool)
    P=numpy.unique(Y[~dominated],axis=0)
    P=P[numpy.argsort(P[:,0])]
    # Fixed reference point dominated by P
    if ref_point is None:
        raise ValueError('A fixed reference point must be provided.')
    r=numpy.asarray(ref_point,dtype=float).ravel()
    if r.size!=2 or not numpy.all(numpy.isfinite(r)):
        raise ValueError('The reference point must contain two finite values.')
    if numpy.any(P<=r):
        raise ValueError('The reference point must be dominated by every '
                         'point in the current Pareto set.')
    # Prepare the predictive distributions
    mu1=numpy.asarray(Y_Test_Pred_N1,dtype=float).ravel()
    mu2=numpy.asarray(Y_Test_Pred_N2,dtype=float).ravel()
    sigma1=numpy.asarray(STD_Y_Test_Pred_N1,dtype=float).ravel()
    sigma2=numpy.asarray(STD_Y_Test_Pred_N2,dtype=float).ravel()
    if mu1.size==0 or not (mu1.size==mu2.size==sigma1.size==sigma2.size):
        raise ValueError('Prediction arrays must have equal, nonzero lengths.')
    # Prevents division by zero
    sigma1=numpy.maximum(sigma1,1e-12)
    sigma2=numpy.maximum(sigma2,1e-12)
    # Creates standard normal distribution
    standard=scipy.stats.norm(loc=0,scale=1)
    # Eq. (19): Psi_infinity(a,b,mu,sigma)
    def psi_inf(a,b,mu,sigma):
        if numpy.isposinf(b):
            return numpy.zeros_like(mu,dtype=float)
        z=(b-mu)/sigma
        return sigma*standard.pdf(z)+(mu-a)*standard.sf(z)
    # Eq. (25): theta(l,u,mu,sigma)
    def theta(l,u,mu,sigma):
        if numpy.isposinf(u):
            return numpy.zeros_like(mu,dtype=float)
        z=(u-mu)/sigma
        return (u-l)*standard.sf(z)
    # Limits of the n+1 integration slices
    l1_values=numpy.r_[r[0],P[:,0]]
    u1_values=numpy.r_[P[:,0],numpy.inf]
    l2_values=numpy.r_[P[:,1],r[1]]
    EHVI=numpy.zeros_like(mu1,dtype=float)
    for l1,u1,l2 in zip(l1_values,u1_values,l2_values):
        psi_y2=psi_inf(l2,l2,mu2,sigma2)
        # Eqs. (31) and (32)
        term_1=(psi_inf(l1,l1,mu1,sigma1)
                 -psi_inf(l1,u1,mu1,sigma1))*psi_y2
        term_2=theta(l1,u1,mu1,sigma1)*psi_y2
        EHVI+=term_1+term_2
    # Remove negative values caused only by numerical round-off
    EHVI=numpy.maximum(EHVI,0.0)
    index=int(numpy.argmax(EHVI))
    return index,EHVI

# =============================================================================
# Main script
# =============================================================================

for example in (1,2,3):
    # Initiate timer
    ti=time.time()
    # Load data
    Data=pandas.read_csv(os.path.join(databasesFolderPath,
                                      'Data_Polyurethane.csv'))
    # Features: Temperature, Time, Ratio
    X_Exp=Data.iloc[:,2:5].to_numpy()
    # Labels (Objective: Maximize OH, Minimize AV)
    Y_Exp_OH=Data.iloc[:,6].to_numpy().reshape(-1,1)
    Y_Exp_AV=-Data.iloc[:,5].to_numpy().reshape(-1,1)
    # Preserve the complete finite experimental domain
    X_Domain=X_Exp.copy()
    Y_Domain_OH=Y_Exp_OH.copy()
    Y_Domain_AV=Y_Exp_AV.copy()
    # Define the true physical extrema in the experimental database
    true_global_max_OH=Data.iloc[:,6].max()
    true_global_min_AV=Data.iloc[:,5].min()
    # Define a fixed reference point for this retrospective database analysis
    ref_point_physical=numpy.array([
        Y_Domain_OH.min()
    -referenceMargin*numpy.maximum(numpy.ptp(Y_Domain_OH),1e-12),
        Y_Domain_AV.min()
        -referenceMargin*numpy.maximum(numpy.ptp(Y_Domain_AV),1e-12)])
    # Rank conditions using the existing equal-weight initialization score.
    Y_Initial_OH_N,__=normalize(Y_Exp_OH,method=labelNorm)
    Y_Initial_AV_N,__=normalize(Y_Exp_AV,method=labelNorm)
    initial_score=w_OH*Y_Initial_OH_N[:,0]+w_AV*Y_Initial_AV_N[:,0]
    ranked_points=numpy.argsort(initial_score,kind='stable')
    # Select an observed condition at approximately the
    # 25th, 50th or 75th percentile of the performance ranking.
    start_fraction={1:0.25,2:0.50,3:0.75}[example]
    percentile_label=f'{int(start_fraction*100)}th percentile of initial score'
    print(f'\nExample {example}: {percentile_label}')
    initial_rank=int(numpy.ceil(start_fraction*len(ranked_points)))-1
    initial_point=int(ranked_points[initial_rank])
    # Print the initial condition for reporting in the Supporting Information.
    print(f'Initial condition:\n'
          f'Temperature: {X_Exp[initial_point,0]} °C\n'
          f'Time: {X_Exp[initial_point,1]} h\n'
          f'Ratio: {X_Exp[initial_point,2]}\n'
          f'OH: {Y_Exp_OH[initial_point,0]:.1f} mg KOH g^-1\n'
          f'AV: {-Y_Exp_AV[initial_point,0]:.1f} mg KOH g^-1')
    # Define initial BO data
    X_BO=X_Exp[initial_point,:].reshape(1,-1)
    Y_BO_OH=Y_Exp_OH[initial_point,:].reshape(-1,1)
    Y_BO_AV=Y_Exp_AV[initial_point,:].reshape(-1,1)
    # Remove the same initial row from all available experimental arrays
    X_Exp=numpy.delete(X_Exp,initial_point,axis=0)
    Y_Exp_OH=numpy.delete(Y_Exp_OH,initial_point,axis=0)
    Y_Exp_AV=numpy.delete(Y_Exp_AV,initial_point,axis=0)
    print("Starting Pareto-Based Multi-Objective BO Loop...")
    # Initialize tracking histories
    EHVI_max_hist=[]
    best_OH_hist=[float(Y_BO_OH.max())]
    best_AV_hist=[float(-Y_BO_AV.max())]
    # Fit feature scaler on known finite experimental domain
    if featureNorm is not None:
        __,skScaler_X=normalize(X_Domain,method=featureNorm)
    else:
        skScaler_X=None
    print(f'Fixed EHVI reference point: OH={ref_point_physical[0]:.3f}, '
          f'AV={-ref_point_physical[1]:.3f}')
    for n in range(maxIter):
        if X_Exp.shape[0]==0:
            print('\nAll available experimental data have been sampled. '
                  'Stopping.')
            break
        if featureNorm is not None:
            X_BO_N,__=normalize(X_BO,method=featureNorm,skScaler=skScaler_X)
            X_Exp_N,__=normalize(X_Exp,method=featureNorm,skScaler=skScaler_X)
        else:
            X_BO_N=X_BO
            X_Exp_N=X_Exp
        # Refit response scalers using the currently acquired observations
        if labelNorm is not None:
            Y_BO_OH_N,skScaler_Y_OH=normalize(Y_BO_OH,method=labelNorm)
            Y_BO_AV_N,skScaler_Y_AV=normalize(Y_BO_AV,method=labelNorm)
            # Transform fixed physical reference with current response scalers
            ref_OH_N,__=normalize(
                numpy.array([[ref_point_physical[0]]]),method=labelNorm,
                skScaler=skScaler_Y_OH)
            ref_AV_N,__=normalize(
                numpy.array([[ref_point_physical[1]]]),method=labelNorm,
                skScaler=skScaler_Y_AV)
            ref_point_N=numpy.array([ref_OH_N[0,0],ref_AV_N[0,0]])
        else:
            Y_BO_OH_N=Y_BO_OH
            Y_BO_AV_N=Y_BO_AV
            ref_point_N=ref_point_physical.copy()
        # Train separate GPs for OH and AV
        model_OH=buildGP(X_BO_N,Y_BO_OH_N,gpConfig=gpConfig)
        model_AV=buildGP(X_BO_N,Y_BO_AV_N,gpConfig=gpConfig)
        # Predict on the remaining experimental candidates
        Y_Pred_OH_N,STD_OH=gpPredict(model_OH,X_Exp_N)
        Y_Pred_AV_N,STD_AV=gpPredict(model_AV,X_Exp_N)
        # Calculate two-objective Expected Hypervolume Improvement
        index_exp,EHVI=calculate_EHVI(Y_BO_OH_N,Y_BO_AV_N,Y_Pred_OH_N,STD_OH,
                                      Y_Pred_AV_N,STD_AV,ref_point=ref_point_N)
        EHVI_max=float(EHVI[index_exp])
        # Early stopping check; a negative threshold disables this criterion
        if EHVI_max<EHVI_max_stop and n>0:
            print('Stopping criterion met.')
            break
        # Append the true experimental data from the same row to the BO dataset
        X_BO=numpy.append(X_BO,X_Exp[index_exp,:].reshape(1,-1),axis=0)
        Y_BO_OH=numpy.append(Y_BO_OH,
                             Y_Exp_OH[index_exp,:].reshape(1,-1),axis=0)
        Y_BO_AV=numpy.append(Y_BO_AV,
                             Y_Exp_AV[index_exp,:].reshape(1,-1),axis=0)
        # Remove the same acquired row from the available experimental pool
        X_Exp=numpy.delete(X_Exp,index_exp,axis=0)
        Y_Exp_OH=numpy.delete(Y_Exp_OH,index_exp,axis=0)
        Y_Exp_AV=numpy.delete(Y_Exp_AV,index_exp,axis=0)
        # Update histories after adding the newly acquired observation
        EHVI_max_hist.append(EHVI_max)
        best_OH_hist.append(float(Y_BO_OH.max()))
        best_AV_hist.append(float(-Y_BO_AV.max()))
        # Print iteration details
        print(f'Iteration #{n+1} | Max EHVI: {EHVI_max:.3e} | '
              f'Best OH: {best_OH_hist[-1]:.3f} | '
              f'Best AV: {best_AV_hist[-1]:.3f}')

    # =========================================================================
    # Observed Pareto Analysis
    # =========================================================================

    # Combine the two observed maximization objectives [OH,-AV]
    Y_Combined_Domain=numpy.concatenate((Y_Domain_OH,Y_Domain_AV),axis=1)
    pareto_mask_observed=numpy.array([
        not numpy.any(numpy.all(Y_Combined_Domain>=point,axis=1) &
                      numpy.any(Y_Combined_Domain>point,axis=1))
        for point in Y_Combined_Domain
    ],dtype=bool)
    pareto_indices_observed=numpy.flatnonzero(pareto_mask_observed)
    if len(pareto_indices_observed)!=1:
        raise ValueError('Expected one unique observed Pareto condition, but '
                         f'found {len(pareto_indices_observed)}.')
    pareto_index_observed=int(pareto_indices_observed[0])

    # =========================================================================
    # Best Observed Results
    # =========================================================================

    # Find the sampled conditions with maximum OH and minimum AV
    index_best_OH=int(Y_BO_OH.argmax())
    index_best_AV=int(Y_BO_AV.argmax())
    print(f'\nBest Observed OH Result:\n'
          f'Temperature: {X_BO[index_best_OH,0]} °C\n'
          f'Time: {X_BO[index_best_OH,1]} hours\n'
          f'Ratio: {X_BO[index_best_OH,2]}\n'
          f'OH: {Y_BO_OH[index_best_OH,0]:.4f}\n'
          f'AV at this condition: {-Y_BO_AV[index_best_OH,0]:.4f}')
    print(f'\nBest Observed AV Result:\n'
          f'Temperature: {X_BO[index_best_AV,0]} °C\n'
          f'Time: {X_BO[index_best_AV,1]} hours\n'
          f'Ratio: {X_BO[index_best_AV,2]}\n'
          f'AV: {-Y_BO_AV[index_best_AV,0]:.4f}\n'
          f'OH at this condition: {Y_BO_OH[index_best_AV,0]:.4f}')
    # Check whether the individual database extrema were sampled
    reached_max_OH=numpy.isclose(best_OH_hist[-1],true_global_max_OH,
                                 rtol=1e-6,atol=1e-8)
    reached_min_AV=numpy.isclose(best_AV_hist[-1],true_global_min_AV,
                                 rtol=1e-6,atol=1e-8)
    status_OH='YES' if reached_max_OH else 'NO'
    status_AV='YES' if reached_min_AV else 'NO'
    # Identify iteration at which the observed Pareto condition was acquired
    pareto_condition=X_Domain[pareto_index_observed]
    pareto_condition_matches=numpy.all(numpy.isclose(
        X_BO,pareto_condition,rtol=1e-6,atol=1e-8),axis=1)
    pareto_recovery_iteration=(int(numpy.flatnonzero(
        pareto_condition_matches)[0]) if numpy.any(pareto_condition_matches)
        else None)
    pareto_recovery_text=(str(pareto_recovery_iteration)
                          if pareto_recovery_iteration is not None
                          else 'Not recovered')
    print(f'\nConvergence Check:\n'
          f'Reached Global Maximum OH: {status_OH}\n'
          f'Best OH Found: {best_OH_hist[-1]:.4f}\n'
          f'Database Maximum OH: {true_global_max_OH:.4f}\n'
          f'Reached Global Minimum AV: {status_AV}\n'
          f'Best AV Found: {best_AV_hist[-1]:.4f}\n'
          f'Database Minimum AV: {true_global_min_AV:.4f}\n'
          f'Iterations to Recover Database Pareto Point: '
          f'{pareto_recovery_text}')

    # =========================================================================
    # Plots
    # =========================================================================

    # Measured conditions and acquisition order
    fig,ax1=plt.subplots(figsize=(5,4))
    fig.suptitle(f'Example {example}: {percentile_label}')
    ax1.scatter(Y_Domain_OH,-Y_Domain_AV,c='lightgray',marker='o',s=60,
                edgecolors='dimgray',label='Measured conditions',zorder=1)
    # Numeration of points based on the order they were added
    for i in range(len(Y_BO_OH)):
        offset=(7,8) if i==0 else (5,5)
        va='top' if i==0 else 'bottom'
        ax1.annotate(str(i),(Y_BO_OH[i,0],-Y_BO_AV[i,0]),
                             textcoords='offset points',xytext=offset,
                             ha='left',va=va,fontsize=9,weight='bold',
                             color='black',zorder=3)
    # Mark unique observed Pareto condition and its first recovery iteration
    pareto_OH=Y_Domain_OH[pareto_index_observed,0]
    pareto_AV=-Y_Domain_AV[pareto_index_observed,0]
    ax1.scatter(pareto_OH,pareto_AV,c='gold',marker='*',s=140,
                edgecolors='black',label='Observed Pareto condition',zorder=4)
    ax1.set_xlabel('Hydroxyl Number (OH) [mg KOH g$^{-1}$]',weight='bold')
    ax1.set_ylabel('Acid Value (AV) [mg KOH g$^{-1}$]',weight='bold')
    ax1.grid(True,linestyle=':',alpha=0.6)
    leg=ax1.legend(loc='upper right',fontsize=8)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    # Convergence plots
    fig,(ax2,ax3)=plt.subplots(1,2,figsize=(12,5))
    # Objective histories include iteration zero; EHVI begins at iteration one
    objective_iterations=numpy.arange(len(best_OH_hist))
    EHVI_iterations=numpy.arange(1,len(EHVI_max_hist)+1)
    # Plot OH convergence
    line1=ax2.plot(EHVI_iterations,EHVI_max_hist,'--or',label='EHVI',
                   linewidth=1.5,markersize=4)
    ax2_twin=ax2.twinx()
    line2=ax2_twin.plot(objective_iterations,best_OH_hist,'-s',
                        color='dodgerblue',label='Best OH',linewidth=1.5,
                        markersize=4)
    line3=ax2_twin.axhline(true_global_max_OH,color='black',linestyle=':',
                           linewidth=1.5,
                           label='Database maximum')
    ax2.set_xlabel('BO Iteration',weight='bold')
    ax2.set_ylabel('Maximum EHVI',weight='bold',color='red')
    ax2_twin.set_ylabel('Hydroxyl Number (mg KOH g$^{-1}$)',weight='bold',
                        color='dodgerblue')
    ax2.tick_params(axis='y',colors='red')
    ax2_twin.tick_params(axis='y',colors='dodgerblue')
    lines_OH=line1+line2+[line3]
    labels_OH=[l.get_label() for l in lines_OH]
    ax2.legend(lines_OH,labels_OH,loc='lower center',
               bbox_to_anchor=(0.5,1.02),ncol=3,frameon=False,fontsize=9)
    ax2.set_title('Convergence to Maximum OH',weight='bold',pad=35)
    ax2.grid(True,linestyle=':',alpha=0.6)
    # Plot AV convergence
    line4=ax3.plot(EHVI_iterations,EHVI_max_hist,'--or',label='EHVI',
                   linewidth=1.5,markersize=4)
    ax3_twin=ax3.twinx()
    line5=ax3_twin.plot(objective_iterations,best_AV_hist,'-s',color='orange',
                        label='Best AV',linewidth=1.5,markersize=4)
    line6=ax3_twin.axhline(true_global_min_AV,color='black',linestyle=':',
                           linewidth=1.5,
                           label='Database minimum')
    ax3.set_xlabel('BO Iteration',weight='bold')
    ax3.set_ylabel('Maximum EHVI',weight='bold',color='red')
    ax3_twin.set_ylabel('Acid Value (mg KOH g$^{-1}$)',weight='bold',
                        color='orange')
    ax3.tick_params(axis='y',colors='red')
    ax3_twin.tick_params(axis='y',colors='orange')
    lines_AV=line4+line5+[line6]
    labels_AV=[l.get_label() for l in lines_AV]
    ax3.legend(lines_AV,labels_AV,loc='lower center',
               bbox_to_anchor=(0.5,1.02),ncol=3,frameon=False,fontsize=9)
    ax3.set_title('Convergence to Minimum AV',weight='bold',pad=35)
    ax3.grid(True,linestyle=':',alpha=0.6)
    fig.subplots_adjust(top=0.75,wspace=0.35)
    plt.show()

    # Print elapsed time in minutes and seconds
    tf=time.time()
    elapsed=tf-ti
    minutes=int(elapsed//60)
    seconds=elapsed%60
    print(f'\nTime of execution: {minutes} min {seconds:.2f} s')

