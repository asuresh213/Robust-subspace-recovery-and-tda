import gtda 
import numpy as np
from generate_data import *
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from gtda.homology import WeakAlphaPersistence
from gtda.diagrams import PersistenceEntropy
import matplotlib.pyplot as plt
from gtda.plotting import plot_diagram
from sklearn.ensemble import RandomForestClassifier
from openml.datasets.functions import get_dataset
from gtda.diagrams import Amplitude, NumberOfPoints
from sklearn.pipeline import make_union
from gtda.pipeline import Pipeline 
import pandas as pd
import matplotlib.pyplot as plt

def make_point_clouds(n_samples_per_shape: int, n_points: int, noise: float):
    """Make point clouds for circles, spheres, and tori with random noise.
    """
    circle_point_clouds = [
        np.asarray(
            [
                [np.sin(t) + noise * (np.random.rand(1)[0] - 0.5), np.cos(t) + noise * (np.random.rand(1)[0] - 0.5), 0]
                for t in range((n_points ** 2))
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    # label circles with 0
    circle_labels = np.zeros(n_samples_per_shape)

    sphere_point_clouds = [
        np.asarray(
            [
                [
                    np.cos(s) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.cos(s) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    # label spheres with 1
    sphere_labels = np.ones(n_samples_per_shape)

    torus_point_clouds = [
        np.asarray(
            [
                [
                    (2 + np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (2 + np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    # label tori with 2
    torus_labels = 2 * np.ones(n_samples_per_shape)
    coefs = list(zip((1+6*np.random.rand(n_samples_per_shape)).astype(int), (1+6*np.random.rand(n_samples_per_shape)).astype(int)))
    print(coefs)
    plane_point_clouds = [
        np.asarray(
            [
                [
                    x,
                    y,
                    coef[0]*x + coef[1]*y,
                ]
                for x in np.linspace(-1, 1, n_points)
                for y in np.linspace(-1, 1, n_points)
            ]
        )
        for coef in coefs
    ]
    plane_labels = 3 * np.ones(n_samples_per_shape)

    point_clouds = np.concatenate((circle_point_clouds, sphere_point_clouds, torus_point_clouds, plane_point_clouds))
    labels = np.concatenate((circle_labels, sphere_labels, torus_labels, plane_labels))

    return point_clouds, labels

n_samples_per_class = 10
point_clouds, labels = make_point_clouds(n_samples_per_class, n_points = 10, noise=0.1)
print(point_clouds.shape)

plot_point_cloud(point_clouds[0]).show()
plot_point_cloud(point_clouds[11]).show()
plot_point_cloud(point_clouds[20]).show()
plot_point_cloud(point_clouds[-1]).show()

# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric = "euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=False
)

diagrams = persistence.fit_transform(point_clouds)
plot_diagram(diagrams[0]).show()
plot_diagram(diagrams[11]).show()
plot_diagram(diagrams[20]).show()
plot_diagram(diagrams[-3]).show()


persistence_entropy = PersistenceEntropy()

# calculate topological feature matrix
X = persistence_entropy.fit_transform(diagrams)
print(X[0], X[11], X[20], X[-1])
#print(X.shape)
plot_point_cloud(X).show()
rf = RandomForestClassifier(oob_score=True)
rf.fit(X, labels)
print(f"OOB score: {rf.oob_score:.3f}")

# Real life data
df = get_dataset('shapes').get_data(dataset_format='dataframe')[0]
df.head()
plot_point_cloud(df.query('target == "human_arms_out3"')[["x", "y", "z"]].values)
point_clouds = np.asarray(
    [
        df.query("target == @shape")[["x", "y", "z"]].values
        for shape in df["target"].unique()
    ]
)
point_clouds.shape
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)
persistence_diagrams = persistence.fit_transform(point_clouds)
persistence_entropy = PersistenceEntropy(normalize=True)
# Calculate topological feature matrix
X = persistence_entropy.fit_transform(persistence_diagrams)
# Visualise feature matrix
plot_point_cloud(X)
labs = np.zeros(40)
labs[10:20] = 1
labs[20:30] = 2
labs[30:] = 3

rf = RandomForestClassifier(oob_score=True, random_state=400)
rf.fit(X, labs)
print(rf.oob_score_)


# Improving the model
metrics = [
    {"metric": metric}
    for metric in ["heat", "betti", "bottleneck", "wasserstein", "landscape"]
]


feature_union = make_union(
    PersistenceEntropy(normalize=True),
    NumberOfPoints(),
    *[Amplitude(**metric) for metric in metrics]
)

p = Pipeline(
    [
        ("features", feature_union),
        ("rf", RandomForestClassifier(oob_score=True, random_state=42)),
    ]
)
p.fit(persistence_diagrams, labs)
p["rf"].oob_score_


# Finding the best features
feature_types = ["PE", "NP", "heat", "betti", "bottleneck", "wasserstein", "landscape"]
feature_names = [type + str(i) for type in feature_types for i in homology_dimensions]
importances = p["rf"].feature_importances_
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
fig.tight_layout()

