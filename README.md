# [gpsc] - Gaussian Process Spatial Clustering

Repo for "Gaussian Process Spatial Clustering." Here we also host the supplement to the paper as a PDF. 

# Installation 
Main dependencies - recommended to install with conda 

    1. numpy
    2. pandas
    3. matplotlib
    4. scikit-learn
    5. tabulate
    6. scikit-fuzzy
    7. geopandas (for real world application, optional)

More specifically: (this has only been tested on Mac, useage may vary)

    - conda create --name gpsc
    - conda activate gpsc

    - conda install numpy
    - conda install pandas
    - conda install matplotlib
    - conda install scikit-learn
    - conda install tabulate
    - conda install -c conda-forge scikit-fuzzy
    - conda install --channel conda-forge geopandas (optional)


# Structure
    - /algorithms/ - contains the actual GPSC algorithm (from pseudocode in paper), and code for GDBSCAN competitor algorithm 
    - /applications/ - contains code for the real world application (without data)
    - /simulations/ - contains scripts for the simulations from the main paper and supplement
    - /utils/ - contains two sets of helper functions for the simulations 

# Simulations 

One can run each script. ie /simulations/psb_simul_1_ball-linear.py after setting up the dependencies. These are as follows:

    - psb_simul_1_ball-linear.py - simulation 1 from main paper
    - psb_simul_2_ring.py - simulation 2 from main paper
    - psb_simul_3_lambda-30-FINAL-k3_noise=2.py - simulation 3 from the main paper

For the supplementary figures in order: 

    - psb_simul_3_lambda-30-FINAL-k3_noise=50.py
    - psb_simul_3_lambda-30-FINAL-k3_noise=100.py
    - psb_simul_3_lambda-30-FINAL-k3_noise=200.py
    - psb_simul_3_lambda-30-FINAL-k4_noise=2.py
    - psb_simul_3_lambda-30-FINAL-k5_noise=2.py
    - psb_simul_3_lambda-30-FINAL-k6_noise=2.py
    - psb_simul_3_lambda-30-FINAL-k3-SX-suppl.py



The main functions of each simulation script are set up to replicate the experiments from the paper and supplement with the exact parameter and seeds used. Running over all 50 random seeds may take some time to complete. It is recommended to set the 'visual' parameter to 'False' to skip the 
visualizations at each step if one chooses to do so for convenience.

# Application

As a disclaimer, at this time, we are unable to release the data and corresponding auxiliary files related to census tracts for our real world application. Hence, the code in the /applications/folder will not run as is, but the main code for clustering and plotting is provided for the reviewers. 

# Final Notes

The structure of the simulations scripts in the main paper may differ slightly from the supplemental scripts for the supplement, but functionally perform similar tasks, ie, replicating the clustering simulations and tuning the parameters. 

With regard to parameter tuning, one may set the 'tune' parameter of the simulation function in each script to True in order to re-tune parameters for any changes to the data generation. In order to just run on one iteration, it is recommended to simply allow the script to iterate over one 
random seed. In the two main paper simulation functions this is found at the top of the simulation helper function (not main function). For the four supplemental experiments, this is found within the main function. Finally, note that for parameter tuning it is possible that the re-tuned
parameters in some simulations may differ slightly to the parameters used for the simulation. This is possibly due to setting a wider range for parameter searches in later iterations of the code, but is unlikely to significantly affect the final scores as a significantly wide range for each parameter was originally employed as well. 
