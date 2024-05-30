import numpy as np
import tda_helper as tda
import rsr_helper as rsr
import examples as ex 

num_samples = 49 # num points from true data
num_noise = 15 # num points from noise
epsilon = 0.1 # for rsr alg P
dummy_labels = np.zeros(num_samples + num_noise) # for initial visualization

#----------------------- RSR demo ---------------------------------------------

data = ex.make_data_plane(num_samples, num_noise) # generate synthetic data
ex.show_point_cloud(data, dummy_labels) # visualizing unlabelled point cloud
ss = rsr.RSR(data, epsilon) # initialize 
shrunk_subsp = ss.algP() # Run algorithm P
labels = ss.generate_labels(data, shrunk_subsp) # label points
ex.show_point_cloud(data, labels) # visualizing labelled point cloud

#----------------------- RSR with TDA demo ------------------------------------

R = 1 if num_noise == 0 else 2 # max homology dimension
data = ex.make_data_sphere(num_samples, num_noise) # generate synthetic data
ex.show_point_cloud(data, dummy_labels) # visualizing unlabelled point cloud

diagrams = tda.generate_persistence_diagrams(data[0], R) # use ripser to generate persistence diags
tda.draw_persistence_diagrams(diagrams) # displaying persistence diagrams

ss = rsr.RSR(data, epsilon) # initialize 
shrunk_subsp = ss.algP() # Run algorithm P
labels = ss.generate_labels(data, shrunk_subsp) # label points
ex.show_point_cloud(data, labels) # visualizing labelled point cloud

clean_data = data[0][labels==1] # isolating the dominant label
clean_diagrams = tda.generate_persistence_diagrams(clean_data, R) # use ripser to generate cleaner persistence diags
tda.draw_persistence_diagrams(clean_diagrams) # displaying cleaner persistence diagrams

# ------------------- SRSR demo ------------------------------------------------

data = ex.make_srsr_data(num_samples, num_noise)
ex.show_point_cloud(data, dummy_labels) # visualizing unlabelled point cloud

ss = rsr.RSR(data, 0.1)
shrunksub = ss.algP()
labels = ss.generate_labels(data, shrunksub)
ex.show_point_cloud(data, labels) # visualizing labelled point cloud

clean_data = [i[labels==1] for i in data] # isolating the dominant label
ex.show_point_cloud(clean_data, np.ones(len(clean_data[0]))) # visualizing labelled point cloud
# -------------------------------------------------------------------------------