import numpy as np;
import matplotlib.pyplot as plt;
def render(x1,y,Coefficient):
    plt.scatter(x1,y);
    plotx= np.linspace(0, 10, 100);
    ploty = np.polyval(Coefficient, plotx)
    plt.plot(plotx,ploty);
    plt.show();

x1 = np.array([5.86,1.34,3.65,4.69,4.13,5.87,7.91,5.57,7.3,7.89]);
y = np.array([0.74,1.18,0.51,-0.48,-0.07,0.37,1.35,0.3,1.64,1.75]);