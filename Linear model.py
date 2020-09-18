import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# function to make random classes
# X random with 20 components
# prob(y=1) depends on X[:,0]
# X[:,1] = X[:,0]**2
# X[:,2] = X[:,0]**3
# X[:,3:20] = noise
def make_Class(n):
    X = np.random.uniform(low=0.,high=1.0,size=(n,20))
    # Modify 1 and 2 to functions of 0
    for j in range(1,3):
        X[:,j] = X[:,0]**(j+1)
    # prob(y=1) is sum of the x components (0 and 1)
    p = X[:,0]
    y = 1*(np.random.uniform(low=0.,high=1.,size=n)<p)
    return X,y

# Q1. First using a sample size of 50, set up a monte-carlo run with test set size of 0.25 times the full sample, 
# and 250 iterations.Draw a new sampel from make_Class at each iteration. Find the difference between the training and test model accuracy.

nmc = 250
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
inbias = np.zeros(nmc)
for i in range(nmc):
    X,y = make_Class(50)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    model = LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs')
    model.fit(X_train,y_train)
    trainScore[i] = model.score(X_train,y_train)
    testScore[i]  = model.score(X_test,y_test)
    inbias[i]=trainScore[i]-testScore[i]

print(np.mean(trainScore))
print(np.mean(testScore))
print(np.mean(inbias))
print('The mean of in sample bias is {}'.format(np.mean(inbias).round(4)))

# Q2. Now sweep through a sequence of C’s given by np.power(10,range(-5,10)). plot this out in semi-log format using plt.semilog() from matplotlib.
def MCtraintest(nmc,modelObj,testFrac):
    trainScore = np.zeros(nmc)
    testScore  = np.zeros(nmc)
    for i in range(nmc):
        X,y = make_Class(50)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testFrac)
        modelObj.fit(X_train,y_train)
        trainScore[i] = modelObj.score(X_train,y_train)
        testScore[i]  = modelObj.score(X_test,y_test)
    return trainScore,testScore

train = []
test = []
Cset = [10**c for c in np.arange(-5,10, dtype=float)]
for c in Cset:
    modelObj = LogisticRegression(C=c,max_iter=2000)
    train.append(MCtraintest(250,modelObj,0.25)[0].mean())
    test.append(MCtraintest(250,modelObj,0.25)[1].mean())

plt.semilogx(Cset,train,label="train accuracy")
plt.semilogx(Cset,test,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("C")
plt.legend()


# Q3. Repeat this entire experiment for the linear support vector classifier, but change the range of C to Clist, but again plot in semilog form.
Clist = [25,10,5,1,0.1,0.01,0.001,0.0001]
trains = []
tests = []
for c in Clist:
    modelObj = LinearSVC(C=c,max_iter=2000)
    trains.append(MCtraintest(250,modelObj,0.25)[0].mean())
    tests.append(MCtraintest(250,modelObj,0.25)[1].mean())
        
plt.semilogx(Clist,trains,label="train accuracy")
plt.semilogx(Clist,tests,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("C")
plt.legend()
