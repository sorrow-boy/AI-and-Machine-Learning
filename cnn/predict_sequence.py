import numpy as np
import tensorflow as tf
from tensorflow import keras as k
import matplotlib.pyplot as plt

# creating the basic model and training items
def create_model(input_shape):
    model = k.models.Sequential()
    model.add(k.layers.Conv2D(5+,3,padding='same',input_shape = input_shape))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(1,activation='softmax'))
    
    return model
    
def train_model(model,x_train,y_train,x_teat = None,y_test = None,epochs = 1,batch_size = 32):
    if x_test == None:
        validation = None
    else:
        validation=(x_test,y_test)
    
    model.compile('adam',loss='mse',metrics=['accuracy']
    history = model.fit(x_train,y_train,epochs = epochs,batch_size = batch_size,validation_set = validation)
    
    print(model.score)
    plt.figure()
    plt.subplot(211)
    plt.plot(epcohs,history['loss'],'g.',label='loss')
    plt.plot(epochs,history['val_loss'],'r.',label='val_loss')
    plt.title('loss graph')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(epochs,history['accuracy'],'g_',label='accuracy')
    plt.plot(epochs,history['val_accuracy'],'r_',label='val_accuracy')
    plt.title('accuracy graph')
    plt.legend()
    
    return model


def main():
    #________________
    # taking the data input
    # or take data from file
    
    #_________________________
    
    data = np.random.randint(1,10000,shape=(20000,6))
    x_train = data[:,:-1]                     # shape=(20000,5)
    x_train = np.expand_dims(x_train,axis=-1) # shape=(20000,5,1)
    y_train = data[:,-1]                      # shape=(20000,1)
    
    data = np.random.randint(1,10000,shape=(2000,6))
    x_test = data[:,:-1]                      # shape=(20000,5)
    x_test = np.expand_dims(x_test,axis=-1)   # shape=(20000,5,1)
    y_test = data[:,-1]                       # shape=(20000,1)
    
    model = create_model(input_shape=(5,1))
    
    print(model.summary())                  # printingh the models architecture
    
    model = train_model(x_train,y_train,x_test = x_test,y_test = y_test)
    
    
    # model.predict(x_test) gives y_pred
    # sklearn.metircs.classification_report(y_test,y_pred)
  

# call main function  
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    