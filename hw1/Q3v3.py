import numpy as np
import math
import matplotlib.pyplot as plt;
def sig(X):
    return 1/ (1+  math.e**(-X));
def df(theta,trainX,trainY):
    total=np.zeros(dim);
    for X, Y in zip(trainX,trainY):
        temp=sig((np.dot(X,theta)))-Y;
        total+=temp*X;
    print("df: ");
    print(total);
    return total;
def test(theta,testX,testY):
    print("Beginning Testing: ")
    TP=0;TN=0;FP=0;FN=0;
    testResult=sig((np.dot(testX,theta)));
    testResult=np.where(testResult>=0.5,1,0);
    print(testResult);
    for X, Y in zip(testResult,testY):
        if X==1 and Y==1:
            TP=TP+1;
        if X==0 and Y==0:
            TN=TN+1;
        if X==1 and Y==0:
            FP=FP+1;
        if X==0 and Y==1:
            FN=FN+1;
    print("TP %d TN %d FP %d FN %d"%(TP,TN,FP,FN))
dim=3;
theta=np.array([-1,1.5,0.5]);
trainX=np.array([[1,0.346,0.78],[1,0.303,0.439],[1,0.358,0.729],[1,0.602,0.863],[1,0.790,0.753],[1,0.611,0.965]]);
trainY=np.array([0,0,0,1,1,1]);
theta=theta-0.1*df(theta,trainX,trainY);
testX=np.array([[1,0.959,0.382],[1,0.75,0.306],[1,0.395,0.76],[1,0.823,0.764],[1,0.761,0.874],[1,0.844,0.435]])
testY=np.array([0,0,0,1,1,1]);
test(theta,testX,testY);

