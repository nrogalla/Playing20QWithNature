# Playing20QWithNature

This repository is accompanying the Bachelor's Thesis 'Playing 20Q with Nature - ChatGPT as guidance for interperting fMRI data'.

It is segmented into two sections which reflect the structure of the thesis.

In the Folder src.ChatGPTAnalysis, functionality used for section 2 of the thesis can be found. 
It is bisected into the main part and an extended version of the scipy.cluster.hierarchy.dendrogram method that allows for truncation by threshold and max cluster size and additionally offers functionality to exclude clusters of smaller size.
The Folder src.fMRIAnalysis reflects functionality used for section 3 of the thesis. 
It is further structured into a data, statistics and visualisation section. 
The data section contains functionality for extracting fMRI data from the given dataset and building representational distance matrices (RDM). It additionally specifies the distinctions that are tested in this thesis
The statistics section contains functionality to correlate fMRI rdms and model RDMs and to perform a permutation test to obtain a p-value. It also harbours functions to analyse the depencency on the noise_ceiling.
The visualisation section contains functions for displaying mutlidimensional scaling and plotting RDMs.

In the Folder 'additional material' one can find additional transcripts of the game played with the variations of using exactly 20 questions, using ChatGPT4 or supplying a list of possible object labels. 