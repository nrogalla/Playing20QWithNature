# Playing20QWithNature

This repository is accompanying the Bachelor's Thesis 'Playing 20Q with Nature - ChatGPT as guidance for interperting fMRI data'.

## Abstract: 

The features underlying human object representation are still largely unknown. While recent progress has been able to unveil important distinctions, they are not enough to explain the complexity of real world objects. This thesis investigates human object representation using ChatGPT's object representation as a guide. 14 distinctions important for ChatGPT object representation were determined by semantic clustering analysis of the questions ChatGPT poses during the game ’20 Questions’. These features are then investigated in fMRI response patterns from the human inferior temporal cortex (IT) by use of representational dissimilarity matrices (RDMs). Results were found to be dependent on voxel selection and distance metric. Still, they suggest the previously not investigated distinction ‘indoors’ to be a feature guiding object representation in human IT which is robust to voxel selection and distance metric dependencies found for other distinctions. Additionally, we were able to replicate findings of previous distinctions most of which surprisingly did not show the same level of robustness.  

## Structure

The folder 'notebooks' contains the analysis conducted and presents findings segmented into two notebooks which reflect the structure of the thesis (ChatGPT and fMRI analysis).

The folder'src' contains the main functionality.

In the Folder 'src.ChatGPTAnalysis', functionality used for section 2 of the thesis can be found. 
It is bisected into the main part and an extended version of the scipy.cluster.hierarchy.dendrogram method that allows for truncation by threshold and max cluster size and additionally offers functionality to exclude clusters of smaller size.

The Folder 'src.fMRIAnalysis' reflects functionality used for section 3 of the thesis. It is further structured into a data, statistics and visualisation section. 
The data section contains functionality for extracting fMRI data from the given dataset and building representational distance matrices (RDM). It additionally specifies the distinctions that are tested in this thesis
The statistics section contains functionality to correlate fMRI rdms and model RDMs and to perform a permutation test to obtain a p-value. It also harbours functions to analyse the depencency on the noise_ceiling.
The visualisation section contains functions for displaying mutlidimensional scaling and plotting RDMs.

The Folder 'data' contains the transcript containing the games '20 Questions' played with ChatGPT that underlie the main part of the analysis. It also contains the fMRI dataset from the THINGS Initiative (https://things-initiative.org/). 

In the Folder 'additional material' one can find additional transcripts of the game played with the variations of using exactly 20 questions, using ChatGPT4 or supplying a list of possible object labels. 
