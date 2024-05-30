# (Simultaneous) Robust Subspace Recovery + Applications to TDA 

This project was done to supplement the final oral presentation portion of my comprehensive exams.

# Simultaneous Robust Subspace Recovery (SRSR) 

## Robust Subspace Recovery - overview 
Given a point cloud $\mathcal{X} \in \mathbb{R}^D$, the Robust Subspace Recovery (RSR) problem aims to determine a lower dimensional linear subspace structure in $\mathcal{X}$. Intuitively, the RSR problem aims to find a lower dimensional linear subspace $L$ that contains "enough" points of $\mathcal{X}$ so that by restricting to $\mathcal{X}\cap L \subseteq \mathcal{X}$, we do not drastically alter the stastical and topological properties of $\mathcal{X}$. One can understand RSR as an optimization problem over the Grasmannian $\mathrm{Gr}(D, \ell)$ parameterizing linear subspaces of $\mathrm{R}^D$ of a given fixed dimension $\ell$. Since the Grassmannian is a non-convex set, this optimization problem is very much non-trivial. Much progress has been made towards this problem, including some very powerful schemes such as Principal Component Analysis (PCA), that are used prominently in the data industry today. A more thorough overview of the RSR literature can be found [here](https://arxiv.org/pdf/1803.01013) and [here](https://jmlr.csail.mit.edu/papers/volume20/17-324/17-324.pdf). 

## Simultaneous RSR:

[This paper](https://arxiv.org/abs/2003.02962) outlines a quiver based approach for simultaneous RSR. We mean simultaneous in the following sense: Suppose we have point clouds $\mathcal{X}_1 \in \mathbb{R}^{d_1}, \dots, \mathcal{X}_m \in \mathbb{R}^{d_m}$ - which can now be concatenated into form a much higher dimensional point cloud $\mathcal{X} = [\mathcal{X}_1, \dots, \mathcal{X}_m] \in \mathbb{R}^D$ with $D = d_1+\dots+d_m$. The SRSR problem aims to find a sequence of linear subspaces $(L_1,\dots, L_m)$ such that each $L_i$ contains enough points of $\mathcal{X}_i$. More details can be found in the paper attached above.  


## The code: 
The `rsr_helper.py` contains an RSR class which initializes the setup `RSR.algP()` 
