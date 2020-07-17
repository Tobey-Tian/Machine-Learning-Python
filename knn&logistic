import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def covidData(Nsamp):
    # Set seed so we all have the same data
    np.random.seed(11)
    # age as a uniform value [0,1]
    age = np.random.uniform(low=0.0,high=1.0,size=Nsamp)
    # Smoke?
    smoke = np.random.randint(low=0,high=2,size=Nsamp)
    # number of smokers and nonsmokers
    nSmoke = np.sum(smoke)
    nnSmoke = Nsamp - nSmoke
    death = np.zeros(Nsamp,dtype=int)
    # Death prob = 0.5 if you don't smoke
    death[smoke==0] = (np.random.uniform(size=nnSmoke)<0.5)
    # If you do smoke, then it depends quadratically on age
    death[smoke==1] = (np.random.uniform(size=(nSmoke))<3./2.*age[smoke==1]**2)
    # stack predictor variables in matrix
    xPredict = np.stack((age,smoke),axis=1)
    return xPredict, death

def MCtraintest(nmc,X,y,modelObj,testFrac):
    trainScore = np.zeros(nmc)
    testScore  = np.zeros(nmc)
    for i in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testFrac)
        modelObj.fit(X_train,y_train)
        trainScore[i] = modelObj.score(X_train,y_train)
        testScore[i]  = modelObj.score(X_test,y_test)
    return trainScore,testScore
    
# 1.Generate a sample with 100 data points using covidData. What is the mean for the death rate overall, for smokers, and for nonsmokers.

xPredict,death = covidData(100)
df = pd.DataFrame({'age': xPredict[:, 0], 'smoke': xPredict[:, 1],'death': death})
df

death_mean=df['death'].mean()
print('The mean for overall death rate is {}'.format(round(death_mean,2)))

death_smoke = df[df['smoke']==1]['death'].mean()
print('The mean for smoker death rate is {}'.format(round(death_smoke,4)))

death_nsmoke = df[df['smoke']==0]['death'].mean()
print('The mean for non smoker death rate is {}'.format(round(death_nsmoke,4)))


# 2.First set up code to evaluate a nearest neighbor classification system. 
# Plot the training and test set mean score (accuracy) from a 250 length monte-carlo simulation. 
# What looks like the optimal neighbor size for test data?

train = []
test = []
neighbors_settings = range(1, 31)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    train.append(MCtraintest(250,xPredict,death,clf,0.25)[0].mean().round(4))
    test.append(MCtraintest(250,xPredict,death,clf,0.25)[1].mean().round(4))
    
print(train)
print(test)

plt.plot(neighbors_settings,train, label="training accuracy")
plt.plot(neighbors_settings,test,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# From the plot, it shows that the optimal neighbor size for test data is 5
    
# 3.What is the score for the training sample with one neighbor?
clf = KNeighborsClassifier(n_neighbors=1)
trainscore,testscore=MCtraintest(250,xPredict,death,clf,0.25)
print(trainscore)

# The score for training sample with one neighbor is 1

# 4. Find the mean training and test classification accuracy for a logistic regression on the same data. 
# Report the mean for training and test set scores.

clf2 = LogisticRegression(solver='lbfgs')
train_score,test_score=MCtraintest(250,xPredict,death,clf2,0.25)
train_mean = np.mean(train_score).round(4)
test_mean = np.mean(test_score).round(4)

print('train mean for logistic regression is {}; test mean is {}'.format(train_mean,test_mean))


