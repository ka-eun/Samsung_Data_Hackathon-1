from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    '''
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    '''

    input, output, _ = preprocessing()

    model = Sequential()
    model.add(Dense(100, input_dim=len(input[0]), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(output[0]), activation='relu'))


    n=len(input)

    input_test=np.array(input[:int(n/5)])
    input_train=np.array(input[int(n/5):n-int(n/5)])
    input_validation=np.array(input[n-int(n/5):])

    n = len(output)
    output_test = np.array(output[:int(n / 5)])
    output_train = np.array(output[int(n / 5):n - int(n / 5)])
    output_validation = np.array(output[n - int(n / 5):])

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()


    hist = model.fit(input_train,output_train,
                     epochs=1000,batch_size=32,
                     validation_data=(input_validation,output_validation),
                     verbose=2)

    scores = model.evaluate(input_test,output_test,verbose=2)
    print(scores)








