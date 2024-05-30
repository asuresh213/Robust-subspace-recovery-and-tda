import numpy as np
import matplotlib.pyplot as plt

# Building the data for tda-rsr
def make_data_sphere(num_pure, num_noise):
    angles = [np.pi*(np.random.uniform(0, 2)) for _ in range(num_pure)]
    pure = np.array([[np.cos(angle), np.sin(angle), 0.0] for angle in angles])
    angle_noise = [[np.pi*(np.random.uniform(0, 2)), np.pi*(np.random.uniform(0, 1))] for _ in range(num_noise)]
    if(angle_noise):
        noise = np.array([[np.sin(l[0])*np.cos(l[1]), np.sin(l[0])*np.sin(l[1]), np.cos(l[0])] for l in angle_noise])
    else:
        noise = np.array([[0,0,0]])
    data = np.concatenate((pure, noise), axis = 0)
    return [data]

# Building data for rsr
def make_data_plane(num_pure, num_noise):
    step_pure = int(np.sqrt(num_pure))
    xp = np.linspace(-1,1,step_pure)
    yp = np.linspace(-1,1,step_pure)
    xp, yp = np.meshgrid(xp, yp)
    zp = 2*xp + 3*yp
    xp, yp, zp = xp.flatten(), yp.flatten(), zp.flatten()
    pure = list(zip(xp, yp, zp))
    pure = np.array([list(ele) for ele in pure])
    t = np.linspace(-1,1,num_noise)
    noise = np.array([t0*np.array([2,3,-1]) for t0 in t])
    data = np.concatenate((pure, noise), axis = 0)
    return [data]

# Building data for srsr
def make_srsr_data(num_pure, num_noise):
    if(int(np.sqrt(num_pure))**2 != num_pure):
        print("number of pure data points needs to be a square for this particular example to to work!")
        return None, None
    data_plane = make_data_plane(num_pure, num_noise)
    data_circ = make_data_sphere(num_pure, num_noise)
    data = [data_plane[0], data_circ[0]]
    return data



# Visualization helper
def show_point_cloud(data, labels):
    colors = ['red', 'blue']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for k in range(len(data)):
        for d0 in data[k][labels==0]:
            ax.scatter(*d0, color=colors[k], marker='o')
        for d0 in data[k][labels==1]:
            ax.scatter(*d0, color=colors[k], marker='^')
    plt.show()

