from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import classification_report
from datetime import datetime
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.mixture import GaussianMixture as EM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string

from sklearn.tree import DecisionTreeClassifier

from helper import evaluate_kmeans, run_kmeans, elbow_function, run_EM, evaluate_EM, \
    kmeans_silhoutte_analysis, plot_EM, run_PCA, run_ICA, run_RCA
from my_encoder import my_encoder

credit_data = pd.read_csv('/Users/marshongreen/Documents/GATech ML(7641)/Datasets/dataset_31_credit-g.csv')


def credit_risk_data():
    data_X = credit_data.drop(['credit_amount', 'other_parties', 'purpose', 'own_telephone', 'foreign_worker'], axis=1)
    data_y = credit_data[['class']]

    features_to_encode = ['personal_status', 'checking_status', 'credit_history', 'savings_status', 'employment',
                          'property_magnitude', 'other_payment_plans', 'housing', 'job', 'class']
    enc = my_encoder()
    enc.fit(data_X, features_to_encode)
    X_train = enc.transform(data_X)
    # X_test = enc.transform(X_test)

    run_PCA(X_train, "Credit Data")
    run_ICA(X_train, "Credit Data")
    run_RCA(X_train, "Credit Data")

    pca_credit = PCA(n_components=3, random_state=5).fit_transform(X_train)
    ica_credit = ICA(n_components=2, random_state=5).fit_transform(X_train)
    rca_credit = RCA(n_components=29, random_state=5).fit_transform(X_train)

    run_kmeans(pca_credit, X_train, "KMEANS")
    run_kmeans(ica_credit, X_train, "KMEANS")
    run_kmeans(rca_credit, X_train, "KMEANS")

    run_EM(pca_credit, X_train, 'PCA Credit Risk Data')
    run_EM(ica_credit, X_train, 'ICA Credit Risk Data')
    run_EM(rca_credit, X_train, 'RCA Credit Risk Data')

    km = KMeans(n_clusters=3, random_state=0)
    y_km = km.fit_predict(X_train)

    score = silhouette_score(X_train, km.labels_, metric='euclidean')
    print('Silhouetter Score: %.3f' % score)

    # kmeans_silhoutte_analysis(X_train)

    elbow_function(X_train)
    run_kmeans(X_train, y_km, "KMEANS")

    em = EM(n_components=2, covariance_type='spherical', random_state=100)
    y_em = em.fit_predict(X_train)
    plot_EM(em, X_train)
    run_EM(X_train, y_em, "EM")
    # evaluate_EM(em, X_train, y_em)

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


credit_risk_data()
