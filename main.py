import numpy as np
import tda_helper as tda
import rsr_helper as rsr
import examples as ex 

num_samples = 60 # num points from true data
num_noise = 20 # num points from noise
epsilon = 0.1 # for rsr alg P

#----------------------- RSR demo ---------------------------------------------

data = ex.make_data_plane(num_samples, num_noise) # generate synthetic data
dummy_labels = [np.zeros(len(data[0]))] # for initial visualization
ex.show_point_cloud(data, dummy_labels) # visualizing unlabelled point cloud
ss = rsr.RSR(data, epsilon) # initialize 
shrunk_subsp = ss.algP() # Run algorithm P
labels = ss.generate_labels(data, shrunk_subsp) # label points
ex.show_point_cloud(data, labels) # visualizing labelled point cloud

#----------------------- RSR with TDA demo ------------------------------------

R = 1 if num_noise == 0 else 2 # max homology dimension
data = ex.make_data_sphere(num_samples, num_noise) # generate synthetic data
dummy_labels = [np.zeros(len(data[0]))] # for initial visualization
ex.show_point_cloud(data, dummy_labels) # visualizing unlabelled point cloud

diagrams = tda.generate_persistence_diagrams(data[0], R) # use ripser to generate persistence diags
tda.draw_persistence_diagrams(diagrams) # displaying persistence diagrams

ss = rsr.RSR(data, epsilon) # initialize 
shrunk_subsp = ss.algP() # Run algorithm P
labels = ss.generate_labels(data, shrunk_subsp) # label points
ex.show_point_cloud(data, labels) # visualizing labelled point cloud

clean_data = data[0][labels[0]==1] # isolating the dominant label
clean_diagrams = tda.generate_persistence_diagrams(clean_data, R) # use ripser to generate cleaner persistence diags
tda.draw_persistence_diagrams(clean_diagrams) # displaying cleaner persistence diagrams

# ------------------- SRSR demo ------------------------------------------------

#coming soon.
