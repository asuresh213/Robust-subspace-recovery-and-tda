import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import teaspoon as ts
from ripser import ripser
import teaspoon.TDA.Draw as Draw

# Setting random seed for uniformity across runs. 
np.random.seed(100)

# Generate persistence diagrams using ripser
def generate_persistence_diagrams(data, R=2):
    return ripser(data, R)['dgms']

# Helper function to draw persistence diagram outputs from ripser
def draw_persistence_diagrams(diagrams, R = 2):
    n=len(diagrams)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize = (20,5))
    for i in range(n):
        plt.sca(axes[i])
        plt.title('{}-dim homologies'.format(i))
        try:
            Draw.drawDgm(diagrams[i])
        except:
            print("dim {}-homology is empty".format(i))
    plt.show()

