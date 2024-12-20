# (Simultaneous) Robust Subspace Recovery + Applications to Topological Data Analysis 

This project was done to supplement the final oral presentation portion of my comprehensive exams. There is a more comprehensive outline and motivation for the project below - but we start with a TL;WR (Too Long; Wont Read).

# TL;WR

The goal of this project is two-fold:
  1. Given multiple point clouds $\mathcal{X}^i \in \mathbb{R}^{d_i}$, recover a tuple of linear subspaces $T_i \subseteq \mathbb{R}^{d_i}$ *simultaneously* in one fell sweep so that each $T_i$ contains "enough" points of $\mathcal{X}^i$. 

  2. Use the above recovery scheme in conjunction with topological data analysis, to first recover prominent subspaces contained in our data before doing homological computations - thereby eradicating phantom homologies that might arise otherwise due to higher dimenisonal noise. 

In [this demo](./_demo_SubspaceRecovery.py) we illustrate the interplay of these methods in eradicating phantom homologies. For the more important question as to why this is important and/or relevant, [this python notebook](./_demo_TopMLwithGiotto.ipynb) illustrates an instance of topological machine learning which levarages the usage of homological computations in conjunction with machine learning. In particular, this notebook illustrates how pre-processing our data via these topological methods could reduce the dimensionality of the input feature matrix for the classifier by an order of **ten** (for real world data, and **hundred** for synthetic data).

# Background:

### Robust Subspace Recovery - overview 
Given a point cloud $\mathcal{X} \in \mathbb{R}^D$, the Robust Subspace Recovery (RSR) problem aims to determine a lower dimensional linear subspace structure in $\mathcal{X}$. Intuitively, the RSR problem aims to find a lower dimensional linear subspace $L$ that contains "enough" points of $\mathcal{X}$ so that by restricting to $\mathcal{X}\cap L \subseteq \mathcal{X}$, we do not drastically alter the statistical and topological properties of $\mathcal{X}$. One can understand RSR as an optimization problem over the Grasmannian $\mathrm{Gr}(D, \ell)$ parameterizing linear subspaces of $\mathrm{R}^D$ of a given fixed dimension $\ell$. The Grassmannian being a non-convex set, makes this optimization problem highly non-trivial. Much progress has been made towards this problem, including some very powerful algorithms such as Principal Component Analysis (PCA), that are used prominently in the data industry today. A more thorough overview of the RSR literature can be found [here](https://arxiv.org/pdf/1803.01013) and [here](https://jmlr.csail.mit.edu/papers/volume20/17-324/17-324.pdf). 

### Simultaneous RSR:

[This key paper](https://arxiv.org/abs/2003.02962) outlines a quiver based approach for *Simultaneous* RSR. We mean simultaneous in the following sense: Given point clouds $\mathcal{X}_1 \in \mathbb{R}^{d_1}, \dots, \mathcal{X}_m \in \mathbb{R}^{d_m}$ - we can now concatenate them to form a much higher dimensional point cloud $\mathcal{X} = [\mathcal{X}_1, \dots, \mathcal{X}_m] \in \mathbb{R}^D$ with $D = d_1+\dots+d_m$. The SRSR problem aims to analyze the joint point cloud $\mathcal{X}$ to, in one fell sweep, find a sequence of linear subspaces $(L_1,\dots, L_m)$ such that each $L_i$ is an RSR solution for $\mathcal{X}_i$. More details about this quiver based approach can be found in the paper above.  

### TDA overview:

Topological Data Analysis (TDA) is an up and coming discipline in data science, and one of its celebrated accomplishments is the utilization of simplicial homology theory to study the so called "persistence homologies" of a dataset $\mathcal{X}$ that encode information about the "shape" of the data in question. Homology in algebraic topolgy, very roughly, deals with quantifying the holey-ness of a topological space. For instance, a data set sampled from a circle poses a prominent 1-dimensional "hole" in the middle of the data set (people in this discipline typically ascribe the dimension of the shape to the dimension of the hole). The surface of sphere has no zero dimensional or one dimensional "holes", but poses a prominent 2 dimensional "hole" in its void inside. [This book](https://www.cs.purdue.edu/homes/tamaldey/book/CTDAbook/CTDAbook.html) and [this book](https://www.amazon.com/Topological-Analysis-Applications-Gunnar-Carlsson/dp/1108838650) formally introduces the subject and all the required background from a data science perspective.

### Using (S)RSR to resolve phantom homologies:

Understandably, when the data $\mathcal{X}$ is corrupted by higher dimensional noise (as is the case in the premise of the (S)RSR problem), running TDA schemes on it could cause unwanted higher dimensional homologies to show up in our analysis, which could misinform our model. Therefore, one could create an (S)RSR-TDA pipeline, where we refine our data $\mathcal{X}$ by recovering a lower dimensional subspace that eradicates these higher dimensional corrputions, and then running TDA analysis on the truncated data. In addition to allowing for dimenison reduction, this also refines the output of our TDA scheme that is now unbothered by higher dimensional noise, thereby resolving the phantom homologies that would have shown up otherwise.  

### (Topological) Machine learning with Giotto 

Since persistence homologies capture topological information of the underlying data, it is worth thinking about ``vectorizing" these persistence diagrams, and using these vectorizations as input to a machine learning model. Often times, these vectorizations produce input feature matrices that have a considerable reduction in dimension/size, with comparable performance as that of contemporary machine learning models aiming to perform a similar classification/regression task by using the data $\mathcal{X}$ directly. 


# The code: 

The `srsr_helper.py` file contains an SRSR class which when given the point cloud $\mathcal{X}$ (written as a sequence of np arrays, one for each $\mathcal{X}_i \in \mathbb{R}^{d_i}$) initializes the set up for algorithm P presented in the key paper. Running `SRSR.algP()` returns a $\mathrm{corank}(B)$ shrunk subspace. The class function `SRSR.generate_labels()` takes the shrunk subspace (along with the original data) to generate labels that distinguish points contained in the lower dimensional subspaces $L_i$.

The `tda_helper.py` file contains methods that perform homology computations on the data. This is done primarily by utilizing a tda library [Teaspoon](https://github.com/teaspoontda/teaspoon) developed and maintained by [Liz Munch](https://elizabethmunch.com/) from Michigan State University. The particular teaspoon functions we use for persistence homology computations utilize the [RIPSER.py](https://ripser.scikit-tda.org/en/latest/) library which wraps around the "blazingly fast" C++ RIPSER package.

The `generate_data.py` file contains custom methods that generate synthetic data for demo purposes.

The `_demo_SubspaceRecovery.py` illustrates the recovery of lower dimensional subspaces using (S)RSR and using it to resolve phantom homologies in TDA.

The `_demo_TopMLwithGiotto.py` file illustrates the usage of the robust tda library [Giotto-tda](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html) to do homology computations on synthetically generated point clouds $\mathcal{X}$ corresponding to four different shapes. Subsequently these diagrams are vectorized via the so called `PersistenceEntropy` function. Finally these vectorizations are used as input features in random forest classifier, which classifies our synthetic data with a 100\% out-of-bag accuracy.

The `_demo_TopMLwithGiotto.ipynb` is a python notebook of the above file that can be run online. 

# Results:

### RSR demo
The first part of the demo illustrates the ability of the SRSR scheme in `srsr_helper.py` to perform RSR tasks. The data for this part of the demo consists of a points sampled from plane union a line. Of course, the plane here is the prominent subspace $L$ we would like to recover; which we promptly do, as illustrated below:

<p align="middle">
  <img src="./imgs/rsr_unlabelled.png" width="300" />
  <img src="./imgs/rsr_labelled.png" width="300" /> 
</p>

### RSR + TDA demo
The second part of the demo illustrates the use of RSR in conjunction with persistence homology computations. Illustrated below is the original data set, which is data sampled from the unit circle in the x-y plane, with synthetic noise data sampled from the unit ball in $\mathbb{R}^3$. The data, along with its persistence diagrams, is illustrated below:

<p align="middle">
  <img src="./imgs/rsrtda_unlabelled.png" width="300" />
  <img src="./imgs/noisy_homologies.png" width="400" /> 
</p>

A note on reading persistence diagrams: For dimensions $\geq 1$, a point away from the diagonal line indicates the potential existence of a "hole" of that dimension in our data, while its distance from the diagonal indicates the magnitude of that potential. Zero dimensional homology indicates the connectedness of our data. In our case - the singular red dot signifying a data point "at infinity" indicates the persistence of one connected component.

For the second part of this demo, we run our data first through the RSR machine, extracting the points contained in the RSR solution, and then generating the persistence diagrams from the clean data. This yields the following diagrams:

<p align="middle">
  <img src="./imgs/rsrtda_labelled.png" width="300" />
  <img src="./imgs/cleaner_homologies.png" width="400" /> 
</p>

Certainly, we see a clean dimension 1 homology representing the unit circle, and improved estimates on dimension 2 homologies. 

### SRSR demo

For the last part of this demo, we consider both families of data points (from parts 1 and 2) considered simultaneously. The SRSR scheme is able to simultaneously recover the correct lower dimensional subspaces, as illustrated below:

<p align="middle">
  <img src="./imgs/srsr_unlabelled.png" width="300" />
  <img src="./imgs/srsr_labelled.png" width="300" /> 
  <img src="./imgs/srsr_resolved.png" width="300" />
</p>


### Topological Machine Learning with Giotto

We sample point clouds from four different manifolds embedded in $\mathbb{r}^3$: a circle, a sphere, a torus and a plane. The machine learning pipeline is able to classify their corresponding persistence entropies with a 100\% out-of-bag accuracy. The whole process is outlined in the file [TopMLwithGiotto.pdf](/TopMLwithGiotto.pdf) which is an excerpt of my response to a question in doctoral comprehensive exam on Topological Data Analysis.

[This python notebook](/_demo_TopMLwithGiotto.ipynb) also contains examples involving 3-d point clouds obtained from real-world data.

# Usage and future work:

Download and run the demo files to reproduce the results listed above. Note: For illustration purposes, the num_pure variable in `demo.py` only admits a perfect squre number when generating srsr (demo 3) data. Following the same layout should allow for the usage of user-defined data.

This implementation of Algorithm P from the key paper involves constructing and (pseudo-)inverting a huge singular matrix $B$ whose dimensions scale linearly with the number of data points. The existing code already takes advantage of the sparsity of this matrix while constructing it. However, improvements can be made to avoid (pseudo-)inverting this matrix altogether -- thereby allowing for faster performance and reduced numerical errors. Future work might also include GPU support to further speed up the process. 
