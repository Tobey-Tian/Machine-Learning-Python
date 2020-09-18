import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load data set
alcohol = pd.read_csv("alcohol.csv")
# Remove squaredal terms - these terms i
alcohol.drop(['agesq','beertaxsq','cigtaxsq','ethanolsq','educsq'],axis=1,inplace=True)
alcohol.head()
count0, count1 = alcohol.abuse.value_counts()
aclass0 = alcohol[alcohol.abuse==0]
aclass1 = alcohol[alcohol.abuse==1]

# under sampling
# Generate sample of class 0 types matching number of class 1 types
under0 = aclass0.sample(count1)
alcoholus = pd.concat([under0,aclass1],axis=0)
# set up data, and check if balanced
y = alcoholus.abuse
X = alcoholus.iloc[:,2:33]
print(np.mean(y))
X.head()

# 1. Logistic Regression
nmc = 100
Clist = [10**c for c in np.arange(-5,10, dtype=float)]
fullmodel = make_pipeline(StandardScaler(),LogisticRegression(solver='lbfgs'))
param_grid={'logisticregression__C':Clist,'logisticregression__max_iter':[3000]}
cvf=ShuffleSplit(test_size=0.25,n_splits=nmc)
grid_search1=GridSearchCV(fullmodel,param_grid,cv=cvf,return_train_score=True)
grid_search1.fit(X,y)
results1 = pd.DataFrame(grid_search1.cv_results_)
print(results1[['rank_test_score','mean_test_score','param_logisticregression__C']])

plt.semilogx(results1['param_logisticregression__C'],results1['mean_test_score'])
plt.ylabel("Mean Test Score")
plt.xlabel("C")
plt.grid()

# 2. KNN classification
nmc = 100
neighbors = [10, 50, 100, 200, 300, 400, 500]
fullmodel = make_pipeline(StandardScaler(),KNeighborsClassifier())
param_grid={'kneighborsclassifier__n_neighbors':neighbors}
cvf=ShuffleSplit(test_size=0.25,n_splits=nmc)
grid_search2=GridSearchCV(fullmodel,param_grid,cv=cvf,return_train_score=True)
grid_search2.fit(X,y)
results2 = pd.DataFrame(grid_search2.cv_results_)
print(results2[['rank_test_score','mean_test_score','param_kneighborsclassifier__n_neighbors']])

plt.semilogx(results2['param_kneighborsclassifier__n_neighbors'],results2['mean_test_score'])
plt.ylabel("Mean Test Score")
plt.xlabel("n_neighbors")
plt.grid()

# 3. Random Forest Classifier
nmc = 100
param_grid={'max_features':[1,2,3,4,5,6,7,8,9,10],'max_depth':[2,4,6,8,10],'n_estimators':[100]}
cvf=ShuffleSplit(test_size=0.25,n_splits=nmc)
grid_search3=GridSearchCV(RandomForestClassifier(),param_grid,cv=cvf,return_train_score=True)
grid_search3.fit(X,y)
results3 = pd.DataFrame(grid_search3.cv_results_)
print(results3[['rank_test_score','mean_test_score','param_max_features','param_max_depth']])

# Random Forest does not need rescaling, Random Forests are based on tree partitioning algorithms.
# There's no analogue to a coefficient one obtain in general regression strategies, which would depend 
# on the units of the independent variables. Instead, one obtain a collection of partition rules, 
# basically a decision given a threshold, and this shouldn't change with scaling
