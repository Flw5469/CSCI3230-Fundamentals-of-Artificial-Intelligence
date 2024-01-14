import numpy as np
import math
import matplotlib.pyplot as plt;
dimension=3;#parameter+1
sampleSize=5;
alpha = 0.1 # Learning rate
training_time=1;
def sig(X):
    return 1/ (1+  math.e**(-X));

def prob(theta,inputX):
    result=np.empty(0);
    for X in inputX:
        temp=(sig((np.dot(X,theta)).sum()))
        #print(X,end="");
        #print(" 's Y value is: ")
        #print(temp);
        result=np.concatenate((result,[temp]));
    print("prob is: ");
    print(result);
    print("theta is: ")
    print(theta);
    return result;

def df(theta,trainX,trainY):
    total=np.zeros(dimension);
    for X, Y in zip(trainX,trainY):
        temp=sig((np.dot(X,theta)))-Y;
        total+=temp*X;
    print("df: ");
    print(total);
    """
    p=prob(theta,trainX); 
    error=1;
    for P,Y in zip(p,trainY):
        error*=(Y)*(P)+(1-Y)*((1-P));
    print("any error chance: ")
    print(1-error);
    """
    return total;

def render(theta,inputX,inputY):
    """
    for X,Y in zip(inputX,inputY):
        plt.scatter(X[1],Y,c='r');
    
    plotx= np.linspace(-1, 1, 100);
    ploty = prob(theta,plotx);
    plt.plot(plotx,ploty);
    plt.axhline(y = 0.5, color = 'b', label = 'axvline - full height')
    plt.show();
    """

def test(theta,testX,testY):
    print("Beginning Testing: ")
    TP=0;TN=0;FP=0;FN=0;
    testResult=prob(theta,testX);
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
# Initialization
theta=np.array([-1,1.5,0.5]);
#theta=np.array([-77.7968,164.7524,-1.5495]);
trainX=np.array([[1,0.346,0.78],[1,0.303,0.439],[1,0.358,0.729],[1,0.602,0.863],[1,0.790,0.753],[1,0.611,0.965]]);
trainY=np.array([0,0,0,1,1,1]);
render(theta,trainX,trainY);
for i in range(training_time):
    print("Enter the %dth train: " %(i+1));
    temp=df(theta,trainX,trainY);
    theta=theta-alpha*temp;
    print("new theta is: ");
    print(theta);
render(theta,trainX,trainY);
test(theta,trainX,trainY)
testX=np.array([[1,0.959,0.382],[1,0.75,0.306],[1,0.395,0.76],[1,0.823,0.764],[1,0.761,0.874],[1,0.844,0.435]])
testY=np.array([0,0,0,1,1,1]);
render(theta,testX,testY);
test(theta,testX,testY);
#theta=np.array([-77.7968	,164.7524	,-1.5495	]);



