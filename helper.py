import itertools
import timeit
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
from sklearn.metrics import silhouette_samples, pairwise_distances
from matplotlib.patches import Ellipse
from kmodes.kmodes import KModes
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict

plt.style.use('seaborn-whitegrid')

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    homogeneity_score
from sklearn.metrics import silhouette_score as sil_score


def run_kmeans(X, y, title):
    kclusters = list(np.arange(2, 50, 2))
    sil_scores = []
    train_times = []

    for k in kclusters:
        start_time = timeit.default_timer()
        km = KMeans(n_clusters=k, n_init=10, random_state=100, n_jobs=-1).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)
        sil_scores.append(sil_score(X, km.labels_))

    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, sil_scores)
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Analysis for KMeans: ' + title)
    plt.show()


def evaluate_kmeans(km, X, y):
    start_time = timeit.default_timer()
    km.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    y_mode_vote = cluster_predictions(y, km.labels_)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(km.n_iter_))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()


def elbow_function(X):
    # calculate distortion for a range of number of cluster
    sse = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random', max_iter=13, random_state=0
        )
        km.fit(X)
        sse.append(km.inertia_)

    # plot
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Distances(Distortion)')
    plt.show()


def cluster_predictions(Y, clusterLabels):
    # assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target

    return pred


def pairwiseDistCorr(X1, X2):
    assert X1.shape[0] == X2.shape[0]

    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def run_EM(X, y, title):
    kdist = list(np.arange(2, 100, 5))
    sil_scores = []
    train_times = []
    aic_scores = []
    bic_scores = []

    for k in kdist:
        start_time = timeit.default_timer()
        em = EM(n_components=k, covariance_type='spherical', random_state=100).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)

        labels = em.predict(X)
        sil_scores.append(sil_score(X, labels))
        # y_mode_vote = cluster_predictions(y, labels)
        # f1_scores.append(f1_score(y, y_mode_vote))
        # homo_scores.append(homogeneity_score(y, labels))
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))

    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Analysis for EM: ' + title)
    plt.show()

    # plot model AIC and BIC
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, aic_scores, label='AIC')
    ax.plot(kdist, bic_scores, label='BIC')
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('Model Complexity Score')
    plt.title('EM Model Complexity: ' + title)
    plt.legend(loc="best")
    plt.show()


def evaluate_EM(em, X, y):
    start_time = timeit.default_timer()
    em.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    labels = em.predict(X)
    y_mode_vote = cluster_predictions(y, labels)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(em.n_iter_))
    print("Log-likelihood Lower Bound: {:.2f}".format(em.lower_bound_))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    plt.show()


def run_PCA(X, title):
    pca = PCA(random_state=5)
    pca.fit(X)
    cum_var = np.cumsum(pca)

    cov_mat = np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    # calculate cumulative sum of explained variances
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # plot explained variances
    plt.bar(range(1, len(X.columns) + 1), var_exp, alpha=0.5,
            align='center', label='individual explained variance')
    plt.step(range(1, len(X.columns) + 1), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'm-')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Eigenvalues', color='m')
    ax1.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("PCA Eigenvalues: " + title)
    fig.tight_layout()
    plt.show()


def run_ICA(X, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    ica = ICA(random_state=5)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.show()


def run_RCA(X, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)

    for i, dim in product(range(10), dims):
        rp = RCA(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()

    fig, ax1 = plt.subplots()
    ax1.plot(dims, mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims, std_recon, 'm-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='m')
    ax2.tick_params('y', colors='m')
    plt.grid(False)

    plt.title("Random Components for 5 Restarts: " + title)
    fig.tight_layout()
    plt.show()


def run_RFC(X, y, df_original):
    rfc = RFC(n_estimators=500, min_samples_leaf=round(len(X) * .01), random_state=5, n_jobs=-1)
    imp = rfc.fit(X, y).feature_importances_
    imp = pd.DataFrame(imp, columns=['Feature Importance'], index=df_original.columns[2::])
    imp.sort_values(by=['Feature Importance'], inplace=True, ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum'] <= 0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def kmeans_silhoutte_analysis(X):
    for i, k in enumerate([2, 3, 4]):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Run the Kmeans algorithm
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        # Get silhouette samples
        silhouette_vals = silhouette_samples(X, labels)

        # Silhouette plot
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        avg_score = np.mean(silhouette_vals)
        ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
        ax1.set_yticks([])
        ax1.set_xlim([-0.1, 1])
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster labels')
        ax1.set_title('Silhouette plot for the various clusters', y=1.02)

        # Scatter plot of data colored with labels
        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
        ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
        ax2.set_xlim([-2, 2])
        ax2.set_xlim([-2, 2])
        ax2.set_xlabel('Eruption time in mins')
        ax2.set_ylabel('Waiting time to next eruption')
        ax2.set_title('Visualization of clustered data', y=1.02)
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.suptitle(f'Silhouette analysis using k = {k}',
                     fontsize=16, fontweight='semibold', y=1.05)


def plot_EM(em, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = em.fit(X).predict(X)
    if label:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / em.weights_.max()
    for pos, covar, w in zip(em.means_, em.covariances_, em.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt([covariance, covariance])

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
