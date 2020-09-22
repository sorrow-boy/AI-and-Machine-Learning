import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

#______________ in this model co2 and consumption in car dataset is used _________#

# working with pandas dataframe
def preprocess(data,noise=None,normalize=False):
    if noise != None:
        data = data.fillna(0)
    if normalize:
        data = sklearn.preprocessing.StandardScaler(data)
    return data

# defining the multiple_linear_model
def train_model(x_train,y_train,x_test,y_test):
    multiple_linear_model = sklearn.linear_model.LinearRegression().fit(x_train,y_train)
    if x_test!=None:
        y_pred = multiple_linear_model.predict(x_test)
        report = sklearn.metrics.classification_report(y_test,y_pred,labels = None) # labels attribute is to set which targets indecis 
        #to include in report.
        print(report)
        
        # plotting model
        plt.plot(x_test['volume'],y_pred,'g.')
        plt.plot(x_test['volume'],y_test,'ro')
        plt.plot(x_test['consumption'],y_pred,'g-')
        plt.plot(x_test['consumption'],y_test,'r-')
        plt.show()
    print('coefficients of model: ',linear_model.coeff_,'\nintercept of linear model: ',linear_model.intercept_)
    return multiple_linear_model

# main function 
def main():
    
    #___________________# 
    # data input field
    # or select data from file.
    #_____X_______________#
    
    data=pd.read_csv('data/car_fuel.csv')
    x_train=data[['consumption','volume']]
    x_train['consumption']=preprocess(x_train['consumption'],normalize=True)
    y_train=data['co2']
    linear_model = train_model(x_train = x_train,y_train = y_train)
    
    # here we are with experienced linear model.
    
    
if __name__ == '__main__':
    main()
    # call to main function

    
    
    
    
    
    
    
    
    
    
    
    
    