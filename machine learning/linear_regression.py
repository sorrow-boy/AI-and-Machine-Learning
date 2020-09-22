# importing modules for data preparetion and model preparation
import numpy as np
import pandas
import sklearn
import matplotlib.pyplot as plt

#__________________________________________


# preprocessing the numpy array data.
def preprocess(data,normalize = False,remove_noise = False,clip=(None,None):

    if normalize: # normalizing data to have mean 0 and variance 1.
        data = sklearn.preprocessing.StandardScaler(data
        
    if remove_noise: # customizing this value with data respect.
        pass
        
    if None not in clip: # bounding each element of data between clip[0] and clip[1]
        data = (data-min(data))J/max(data)
    
    return data
        
# defining the model training function.
def train_model(x_train,y_train,x_test = None,y_test = None):
    linear_model = sklearn.linear_model.LinearRegression().fit(X,y)
    # predicting and visualizing data
    # preparing the classification report
    
    if x_test!=None:
        y_pred = linear_model.predict(x_test)
        report = sklearn.metrics.classification_report(y_test,y_pred,labels = None) # labels attribute is to set which targets indecis 
        #to include in report.
        print(report)
        
        # plotting model
        plt.plot(x_train,y_train,'g.')
        plt.plot(x_test,y_test,'ro')
        plt.show()
    print('coefficients of model: ',linear_model.coeff_,'\nintercept of linear model: ',linear_model.intercept_)
    return linear_model
    
    
# defining the main fuction.
def main():

    #___________________# 
    # data input field
    # or select data from file.
    #_____X_______________#
    
    x_train = np.array(range(10)+np.random.randn((10))) # creating data for training the model
    y_train = np.array(range(10))
    x_train=preprocess(x_train,normalize=True)
    
    linear_model = train_model(x_train = x_train,y_train = y_train)
    
    # here we are with experienced linear model.
    
    
if __name__ == '__main__':
    main()
    # call to main function








