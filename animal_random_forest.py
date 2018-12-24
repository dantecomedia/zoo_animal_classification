#importing dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn


#loading datasets
data=pd.read_csv("zoo.csv")
X=data.iloc[1:,1:17]
y=data.iloc[1:,17]



#encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Encoding
le=LabelEncoder()
ohe=OneHotEncoder()

label_y=le.fit_transform(y)
onehotencoder = OneHotEncoder() #categorical_features = [0])
onhe_x = onehotencoder.fit_transform(X).toarray() 

""" from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier().fit(X,y)  """

#cross validation 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=30, random_state=0)
classifier.fit(X_train,y_train)

train_pred=classifier.predict(X_train)
pred=classifier.predict(X_test)


from sklearn import metrics 
pscore = metrics.accuracy_score(y_test, pred)






 







    









    







