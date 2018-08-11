from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import concatenate
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


if __name__ == "__main__":
    _inputs, outputs, _ = preprocessing()

    n = len(outputs)

    # 한 row의 elem 개수만큼 append
    N = len(_inputs[0])

    inputs = []
    for _ in range(N):
        inputs.append([])

    #
    for _input in _inputs:
        for i, elem in enumerate(_input):
            inputs[i].append(elem)

    # inputs: attribute에 따라 생성한 list

    models = []
    _models = []

    inputs_by_attr = []

    n = len(inputs)  # 속성의 개수
    m = len(inputs[0])

    for r, row in enumerate(inputs):
        inputs_by_attr.append([])
        for _ in range(3):
            inputs_by_attr[r].append([])
        tmp = []
        count = 0
        for i, elem in enumerate(row):
            tmp.append(elem)
            if i<int(m/5):
                inputs_by_attr[r][0].append(elem)
            elif i<int(m-(m/5)):
                inputs_by_attr[r][1].append(elem)
            else:
                inputs_by_attr[r][2].append(elem)

    pprint(inputs_by_attr)

'''

    for input in inputs:
        model = Input(shape=(len(input[0]),))

        models.append(model)

    x = concatenate(models)
    x = Dense(64, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    x = Dense(len(outputs[0]), activation='relu')(x)


    model = Model(inputs=models, outputs=x)

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()

    hist = model.fit([np.array(i) for i in inputs], np.array(outputs),
                     epochs=10, batch_size=32,
                     verbose=2)




    """
    model = Sequential()
    model.add(Dense(100, input_dim=len(input_test[0]), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(output_test[0]), activation='relu'))

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()

    hist = model.fit(input_train, output_train,
                     epochs=10, batch_size=32,
                     validation_data=(input_validation, output_validation),
                     verbose=2)

    scores = model.evaluate(input_test, output_test, verbose=2)
    print('complete: %s = %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    print(model.predict(input_test))
    """
'''
