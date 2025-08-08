# fly_connectome

##### --------------------------------------- #####
- A repsoitory containging scripts to perform infomap community detection on the Drosophilia Connectome from flywire.ai (1)
- Contains a GUI to run Infomap community detection (2) adjust number of synapses to include (measure of degree), which classes of neurotransmitters, and number of internal trials
- Contains plotting GUI to visualize infomap communties across different hierarchical levels and options to adjust size based on neuron degree



Citation:
1. Dorkenwald, S., Matsliah, A., Sterling, A.R. et al. Neuronal wiring diagram of an adult brain. Nature 634, 124–138 (2024). https://doi.org/10.1038/s41586-024-07558-y
2. M. Rosvall, & C.T. Bergstrom, Maps of random walks on complex networks reveal community structure, Proc. Natl. Acad. Sci. U.S.A. 105 (4) 1118-1123, https://doi.org/10.1073/pnas.0706851105 (2008).



Installs:
infomap algorithm 
pip install infomap

Input files: /derivatives
connections_princeton.csv
