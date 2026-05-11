import numpy as np
from ISLP import load_data
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_nci():
    data = load_data('NCI60')
    X = data['data']
    y = np.array(data['labels']).ravel()
    return X, y

def reduce_hc(X, n_clusters=50):
    X_T = X.T
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X_T)
    X_new = []
    for i in range(n_clusters):
        cluster_features = X[:, labels == i]
        cluster_mean = cluster_features.mean(axis=1)
        X_new.append(cluster_mean)
    return np.array(X_new).T, labels

def reduce_hc_test(X, labels, n_clusters=50):
    X_new = []
    for i in range(n_clusters):
        cluster_features = X[:, labels == i]
        cluster_mean = cluster_features.mean(axis=1)
        X_new.append(cluster_mean)
    return np.array(X_new).T

def reduce_pca_train(X, n_components=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return scaler, pca, X_pca

def reduce_pca_test(X, scaler, pca):
    X_scaled = scaler.transform(X)
    return pca.transform(X_scaled)

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    X, y = load_nci()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Hierarchical Clustering
    X_train_hc, labels = reduce_hc(X_train)
    X_test_hc = reduce_hc_test(X_test, labels)
    model_hc = train_model(X_train_hc, y_train)
    acc_hc = evaluate(model_hc, X_test_hc, y_test)

    # PCA
    scaler, pca, X_train_pca = reduce_pca_train(X_train)
    X_test_pca = reduce_pca_test(
        X_test,
        scaler,
        pca
    )
    model_pca = train_model(X_train_pca, y_train)
    acc_pca = evaluate(model_pca, X_test_pca, y_test)

    print(f"Accuracy with Hierarchical Clustering: {acc_hc:.4f}")
    print(f"Accuracy with PCA: {acc_pca:.4f}")