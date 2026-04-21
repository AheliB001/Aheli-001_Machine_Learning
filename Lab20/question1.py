import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===== LOAD LOCAL DATA =====
USArrests = pd.read_csv("USArrests.csv")
USArrests.set_index("State", inplace=True)

USArrests.head()
USArrests.mean()
USArrests.var()

scaler = StandardScaler(with_std=True, with_mean=True)
USArrests_scaled = scaler.fit_transform(USArrests)

pcaUS = PCA()
pcaUS.fit(USArrests_scaled)

scores = pcaUS.transform(USArrests_scaled)

print(scores.std(axis=0, ddof=1))
print(pcaUS.explained_variance_)

i, j = 0, 1

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:, i], scores[:, j])

ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             pcaUS.components_[i, k],
             pcaUS.components_[j, k])
    ax.text(pcaUS.components_[i, k],
            pcaUS.components_[j, k],
            USArrests.columns[k])

plt.show()

scale_arrow = s_ = 2

scores[:, 1] *= -1
pcaUS.components_[1] *= -1

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:, 0], scores[:, 1])

ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))

for k in range(pcaUS.components_.shape[1]):
    ax.arrow(0, 0,
             s_ * pcaUS.components_[i, k],
             s_ * pcaUS.components_[j, k])
    ax.text(s_ * pcaUS.components_[i, k],
            s_ * pcaUS.components_[j, k],
            USArrests.columns[k])

plt.show()

print(scores.std(axis=0, ddof=1))
print(pcaUS.explained_variance_)
print(pcaUS.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ticks = np.arange(pcaUS.n_components_) + 1

ax = axes[0]
ax.plot(ticks, pcaUS.explained_variance_ratio_, marker='o')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of Variance Explained')
ax.set_ylim([0,1])
ax.set_xticks(ticks)

ax = axes[1]
ax.plot(ticks, pcaUS.explained_variance_ratio_.cumsum(), marker='o')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Proportion of Variance Explained')
ax.set_ylim([0,1])
ax.set_xticks(ticks)

plt.show()

a = np.array([1,2,8,-3])
print(np.cumsum(a))