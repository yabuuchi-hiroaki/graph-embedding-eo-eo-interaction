# Data and scripts for 'In vitro and in silico prediction of antibacterial interaction between essential oils via graph embedding approach'

Essential oils are known to possess antibacterial activity. However, it is difficult to foresee the effect of blending the oils because hundreds of compounds can be involved in synergistic and antagonistic interactions. This repository stores the data and Python scripts to generate results in the paper [^1].

<p align="center"><img src="https://github.com/yabuuchi-hiroaki/graph-embedding-eo-eo-interaction/blob/images/github_overview.jpg"></p>

## Installation and Dependencies

The Python scripts need following packages: stellargraph, scikit-learn

Please make sure to install all dependencies prior to running the code. 
The code presented here was implemented and tested in Anaconda ver.22.11.1 (Python ver.3.9.16).

## Usage
1. Download this repository.
2. Uncompress "data.zip" file to create "data" folder.
    - "data/pair_Sa" : synergistic/antagonistic/no interaction pairs of essential oils (= Supplementary Table S1 of the paper)
    - "data/pair_content_Sa" : chemical composition of the essential oils (corresoponding to Supplementary Table S2 of the paper)
3. Run a Python script "GraphEmbedEO_3classCV.py"
    - The script performs 10-fold cross-validation and calculates AUCs for synergistic-versus-rest and antagonistic-versus-rest classification.
4. Run a Python script "GraphEmbedEO_pred3class.py"
    - The script calculates output probability for given EO-EO pairs (described in "data/pred/pair" file) as an output file "output_prob.txt".

## References
[^1]: Yabuuchi H et al. In vitro and in silico prediction of antibacterial interaction between essential oils via graph embedding approach. 
Sci Rep. 2023 Nov 2;13(1):18947. doi: [10.1038/s41598-023-46377-5](https://doi.org/10.1038/s41598-023-46377-5).
