import numpy as np
import math
import matplotlib.pyplot as plt;
dimension=3;#parameter+1
sampleSize=5;
alpha = 0.1 # Learning rate

def sig(X):
    return 1/ (1+  math.e**(X*-1));

def prob(theta,inputX):
    result=np.empty(0);
    for X in inputX:
        temp=(sig((theta*X).sum()))
        #print(X,end="");
        #print(" 's Y value is: ")
        #print(temp);
        result=np.concatenate((result,[temp]));
    return result;

def df(theta,trainX,trainY):
    total=np.zeros(dimension);
    for X, Y in zip(trainX,trainY):
        temp=sig((X*theta))-Y;
        total+=temp*X;
    print("df: ");
    print(total);
    p=prob(theta,trainX);
    error=0;
    for P,Y in zip(p,trainY):
        error+=(-Y)*math.log(P)-(1-Y)*math.log((1-P));
    print("error: ")
    print(error);
    return total;

def render(theta,inputX,inputY):
    for X,Y in zip(inputX,inputY):
        if (Y==1):
            plt.scatter(X[1],X[2],c='g');
        if (Y==0):
            plt.scatter(X[1],X[2],c='r');
    plotx= np.linspace(-1, 1, 100);
    ploty = prob(theta,plotx);
    plt.plot(plotx,ploty);
    #plt.axvline(x = 0.5, color = 'b', label = 'axvline - full height')
    plt.show();
    

# Initialization
theta=np.array([-1,1.5,0.5]);
trainX=np.array([[1,0.346,0.78],[1,0.303,0.439],[1,0.358,0.729],[1,0.602,0.863],[1,0.790,0.753],[1,0.611,0.965]]);
trainY=np.array([0,0,0,1,1,1]);
render(theta,trainX,trainY);
for i in range(1):
    temp=df(theta,trainX,trainY);
    theta=theta-alpha*temp;
render(theta,trainX,trainY);
testX=np.array([[1,0.959,0.382],[1,0.75,0.306],[1,0.395,0.76],[1,0.823,0.764],[1,0.761,0.874],[1,0.844,0.435]])
testY=np.array([0,0,0,1,1,1]);
render(theta,testX,testY);




