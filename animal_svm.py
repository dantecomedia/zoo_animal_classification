#importing dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import svm

#loading datasets
data=pd.read_csv("zoo.csv")
X=data.iloc[1:,1:17]
y=data.iloc[1:,17]

clf = svm.LinearSVC() #gamma='scale', decision_function_shape='ovo')

#cross validation 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
clf.fit(X_train, y_train)

from sklearn.decomposition import PCA
pca = PCA(n_components=1).fit(X_train)
pca_2d = pca.transform(X_train)
plt.scatter(pca_2d,y_train,s=None, c=y_train, marker=None, cmap='rainbow', norm=None, vmin=None, vmax=None, alpha=None)

y_predict=clf.predict(X_test)

from sklearn import metrics 
pscore = metrics.accuracy_score(y_test,y_predict)

X = data.iloc[1:,17]
y = y_predict

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)
