import os
import numpy as np
from sklearn import model_selection
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics

data =pd.read_csv('car.data',names=['buying','maint','doors','persons','lug_boot','safety','class'])
data['class'],class_names = pd.factorize(data['class'])
data['buying'],a = pd.factorize(data['buying'])
data['maint'],b= pd.factorize(data['maint'])
data['doors'],c = pd.factorize(data['doors'])
data['persons'],d = pd.factorize(data['persons'])
data['lug_boot'],e = pd.factorize(data['lug_boot'])
data['safety'],f= pd.factorize(data['safety'])



X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


k=int(input("please enter tree depth : "))
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=k, random_state=0)
dtree.fit(X_train, y_train)


y_pred = dtree.predict(X_test)


count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy:'+ str(accuracy))
feature_names = X.columns
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,
                                class_names=class_names)
treegraph = graphviz.Source(dot_data)
treegraph.view()


#Sara Mazaheri
#Decision Tree
#Data Mining Project
