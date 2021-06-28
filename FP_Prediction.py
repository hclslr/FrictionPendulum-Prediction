"""

Cilsalar, H. (2021). "Prediction model for base shear increase due to vertical ground shaking in friction pendulum isolated structures"


Base shear, maximum isolator displacement and residual displacement increase prediction due to vertical ground acceleration


"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import load_model
import tkinter as tk
model = load_model("FP_PredictionMoldel.h5")

root = tk.Tk()
def plotFragilityVR():
    from scipy.stats import lognorm
    data = pd.read_csv('vrData.txt' , sep =',')
    data = data[data['stdVal'] > 0]  
    data = data[data['meanVal'] > 0]  
    data = data[data['u_stdVal'] > 0]  
    data = data[data['u_meanVal'] > 0]  
    data = data[data['ur_stdVal'] > 0.065]  
    data = data[data['ur_meanVal'] > 0]  
    data = data[data['stdVal'] < 0.06]
    data = data[data['u_stdVal'] < 0.075]  
    data = data[data['ur_stdVal'] < 0.22]
    data = data.reset_index()
    data = data.drop(columns = ['index'])    
    xData=data[['Teff','xi','muFast','alpha']]
    yData=data.drop(['Teff','xi','muFast','alpha'], axis = 1)
    TVal = float(T.get())
    xiVal = float(xi.get())
    muFastVal = float(muFast.get())
    alphaVal = float(alpha.get())
    model = load_model("FP_PredictionMoldel.h5")
    xNew = [[TVal/xData['Teff'].max(),xiVal/xData['xi'].max(),muFastVal/xData['muFast'].max(),alphaVal/xData['alpha'].max()]]
    predictions2  = model.predict(xNew)
    predictions2 *= [yData['stdVal'].max(),yData['meanVal'].max(),yData['u_stdVal'].max(),yData['u_meanVal'].max(),yData['ur_stdVal'].max(),yData['ur_meanVal'].max()]
    vr_dist = lognorm(predictions2[0][0], predictions2[0][1])
    plt.figure(dpi=150)
    x=np.linspace(0.9,1.25,50)
    plt.plot(x,vr_dist.cdf(x), label = 'Predicted')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.xlabel('VR')
    plt.axis([0.9,1.25,0,1])
    plt.ylabel('Probability of non-exceedance')
    plt.show()
    return [x,vr_dist.cdf(x)]


def plotFragilityUR():
    from scipy.stats import lognorm
    data = pd.read_csv('vrData.txt' , sep =',')
    data = data[data['stdVal'] > 0]  
    data = data[data['meanVal'] > 0]  
    data = data[data['u_stdVal'] > 0]  
    data = data[data['u_meanVal'] > 0]  
    data = data[data['ur_stdVal'] > 0.065]  
    data = data[data['ur_meanVal'] > 0]  
    data = data[data['stdVal'] < 0.06]
    data = data[data['u_stdVal'] < 0.075]  
    data = data[data['ur_stdVal'] < 0.22]  
    data = data.reset_index()
    data = data.drop(columns = ['index'])    
    xData=data[['Teff','xi','muFast','alpha']]
    yData=data.drop(['Teff','xi','muFast','alpha'], axis = 1)
    TVal = float(T.get())
    xiVal = float(xi.get())
    muFastVal = float(muFast.get())
    alphaVal = float(alpha.get())
    model = load_model("FP_PredictionMoldel.h5")
    xNew = [[TVal/xData['Teff'].max(),xiVal/xData['xi'].max(),muFastVal/xData['muFast'].max(),alphaVal/xData['alpha'].max()]]
    predictions2  = model.predict(xNew)
    predictions2 *= [yData['stdVal'].max(),yData['meanVal'].max(),yData['u_stdVal'].max(),yData['u_meanVal'].max(),yData['ur_stdVal'].max(),yData['ur_meanVal'].max()]
    print(predictions2)
    ur_dist = lognorm(predictions2[0][2], predictions2[0][3])
    plt.figure(dpi=150)
    xur=np.linspace(0.75,1.5,50)
    plt.plot(xur,ur_dist.cdf(xur), label = 'Predicted')
    plt.grid(True)
    plt.legend(frameon=False)
    plt.xlabel('UR')
    plt.axis([0.9,1.25,0,1])
    plt.show()
    return [xur,ur_dist.cdf(xur)]
    

def plotFragilityURr():
    from scipy.stats import lognorm
    data = pd.read_csv('vrData.txt' , sep =',')
    data = data[data['stdVal'] > 0]  
    data = data[data['meanVal'] > 0]  
    data = data[data['u_stdVal'] > 0]  
    data = data[data['u_meanVal'] > 0]  
    data = data[data['ur_stdVal'] > 0.065]  
    data = data[data['ur_meanVal'] > 0]  
    data = data[data['stdVal'] < 0.06]
    data = data[data['u_stdVal'] < 0.075]  
    data = data[data['ur_stdVal'] < 0.22]
    data = data.reset_index()
    data = data.drop(columns = ['index'])    
    xData=data[['Teff','xi','muFast','alpha']]
    yData=data.drop(['Teff','xi','muFast','alpha'], axis = 1)
    TVal = float(T.get())
    xiVal = float(xi.get())
    muFastVal = float(muFast.get())
    alphaVal = float(alpha.get())
    model = load_model("FP_PredictionMoldel.h5")
    xNew = [[TVal/xData['Teff'].max(),xiVal/xData['xi'].max(),muFastVal/xData['muFast'].max(),alphaVal/xData['alpha'].max()]]
    predictions2  = model.predict(xNew)
    predictions2 *= [yData['stdVal'].max(),yData['meanVal'].max(),yData['u_stdVal'].max(),yData['u_meanVal'].max(),yData['ur_stdVal'].max(),yData['ur_meanVal'].max()]
    urr_dist = lognorm(predictions2[0][4], predictions2[0][5])
    plt.figure(dpi=150)
    xurr=np.linspace(0.75,1.5,50)
    plt.plot(xurr,urr_dist.cdf(xurr), label = 'Predicted')
    plt.legend(frameon=False)
    plt.xlabel('$UR_r$')
    plt.show()
    return [xurr,urr_dist.cdf(xurr)]

def printResult(datapair):
    plt.close()
    textWindow=tk.Tk()
    textWindow.title("Probability Results")
    text1=tk.Text(textWindow,height=50, width=125)
    text1.insert(tk.INSERT, f'x,CDFx\n')
    for i in range(0,len(datapair[0])):
         text1.insert(tk.INSERT, f'{datapair[0][i]:.4f},{datapair[1][i]:.4f}\n')
    text1.pack()
def printModelInfo():
    print("Model information is given below")
    print("___________________________________")
    model = load_model("FP_PredictionMoldel.h5")
    text = model.summary()
    json_string = model.to_json()
    print(json_string)
    print(model.weights())
    print("End of model information")
    print("___________________________________")

    
myLabel1 = tk.Label(root,text = 'Prediction model for base shear increase due to \
 vertical ground shaking in friction pendulum isolated structures \n by Huseyin Cilsalar, PhD \n \
 Yozgat Bozok Univesity,Department of Civil Engineering\n \
 huseyin.cilsalar@bozok.edu.tr')




myLabel2 = tk.Label(root,text = 'Please enter structural properties to the boxes below')
Tlabel= tk.Label(root,text = 'Teff')
xilabel= tk.Label(root,text = f'\N{GREEK SMALL LETTER XI}')
muFastlabel= tk.Label(root,text = f'\N{GREEK SMALL LETTER MU} (Fast)')
alphalabel= tk.Label(root,text = f'\N{GREEK SMALL LETTER ALPHA}')
root.title("Probability Curve Prediction")


myLabel1.grid(row=0, column = 0 , columnspan=4)
myLabel2.grid(row=1, column = 0 , columnspan=4)
Tlabel.grid(row=2, column = 0)
xilabel.grid(row=2, column = 1)
muFastlabel.grid(row=2, column = 2)
alphalabel.grid(row=2, column = 3)

T = tk.Entry(root,width=15,borderwidth=5)
xi = tk.Entry(root,width=15,borderwidth=5)
muFast = tk.Entry(root,width=15,borderwidth=5)
alpha = tk.Entry(root,width=15,borderwidth=5)


button_vr = tk.Button(root,text = 'Plot and Print Probability Curve of VR', padx = 30, pady=10, command= lambda:printResult(plotFragilityVR()))
button_u = tk.Button(root,text = 'Plot and Print Probability Curve of UR', padx = 30, pady=10, command= lambda:printResult(plotFragilityUR()))
button_ur = tk.Button(root,text = 'Plot  and Print Probability Curve of URr', padx = 30, pady=10, command= lambda:printResult(plotFragilityURr()))
button_showModel = tk.Button(root,text = 'Show Model Information', padx = 30, pady=10, command= printModelInfo)



T.grid(row=3,column=0)
xi.grid(row=3,column=1)
muFast.grid(row=3,column=2)
alpha.grid(row=3,column=3)
button_vr.grid(row=4,column=0)
button_u.grid(row=4,column=1)
button_ur.grid(row=4,column=2)
button_showModel.grid(row=4,column=3)


root.mainloop()


exit()



