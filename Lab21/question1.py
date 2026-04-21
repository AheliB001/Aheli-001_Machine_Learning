import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("USArrests.csv")
df.set_index("State", inplace=True)

X = df.values
k = 3

np.random.seed(1)
centroids = X[np.random.choice(len(X), k, replace=False)]

for _ in range(100):

    distances = np.linalg.norm(X[:, None] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    new_centroids = np.array([
        X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
        for i in range(k)
    ])

    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200)
plt.xlabel("Murder")
plt.ylabel("Assault")
plt.title("K-Means (From Scratch)")
plt.show()