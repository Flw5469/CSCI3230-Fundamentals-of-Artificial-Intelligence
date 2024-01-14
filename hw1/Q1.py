import numpy as np;
import matplotlib.pyplot as plt;

def OLS(XT,y):
    temp=np.matmul(XT,XT.T);
    temp=np.linalg.inv(temp);
    temp=np.matmul(temp,XT);
    temp=np.matmul(temp,y);
    print("coefficient is: ");
    print(temp);
    return temp;

def render(x1,y,Coefficient):
    plt.scatter(x1,y);
    plotx= np.linspace(0, 10, 100);
    ploty = np.polyval(Coefficient, plotx)
    plt.plot(plotx,ploty);
    plt.show();

def test(testx,testy,CoefficientInReverse):
    print(CoefficientInReverse);
    predictArray=np.zeros(sample_size,dtype=float);
    multiple=np.ones(sample_size,dtype=float);
    
    #construct theta0*1, theta1*X, theta2*X^2
    for coefficient in CoefficientInReverse:
        predictArray+=coefficient*multiple;
        multiple*=testx;
    

    print("predict is: ");
    print(predictArray)
    predictArray=predictArray-testy;
    predictArray=predictArray*predictArray;
    error=predictArray.sum();
    print("error is: ");
    print(error);

sample_size=10;
x1 = np.array([5.86,1.34,3.65,4.69,4.13,5.87,7.91,5.57,7.3,7.89]);
y = np.array([0.74,1.18,0.51,-0.48,-0.07,0.37,1.35,0.3,1.64,1.75]);
x2= x1*x1;
x3=x2*x1;
x4=x3*x1;
XT=np.vstack(([1,1,1,1,1,1,1,1,1,1],x1,x2,x3,x4));
print(XT);
render(x1,y,OLS(XT,y)[::-1])


testx=([5.8,0.57,4.3,6.55,0.82,3.72,5.8,3.26,6.75,4.77]);
testy=([0.93,1.87,-0.06,1.6,1.22,0.9,0.93,1.53,1.73,-0.51]);
test(testx,testy,OLS(XT,y));


