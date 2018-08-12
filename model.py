from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import concatenate, BatchNormalization
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


if __name__ == "__main__":
    _inputs, outputs, _ = preprocessing()

    inputs = []
    for _ in range(len(_inputs[0])):
        inputs.append([])

    for _input in _inputs:
        for i, elem in enumerate(_input):
            inputs[i].append(elem)

    # inputs: attribute에 따라 생성한 list




    inputs_by_attr = []

    # train, test, validation
    for _ in range(3):
        inputs_by_attr.append([])

    for i, col in enumerate(inputs):
        n = len(col)
        inputs_by_attr[0].append(col[:int(n / 5)])  # test
        inputs_by_attr[1].append(col[int(n / 5):n - int(n / 5)])  # train
        inputs_by_attr[2].append(col[n - int(n / 5):])  # validation

    n = len(outputs)
    output_test = (outputs[:int(n / 5)])
    output_train = (outputs[int(n / 5):n - int(n / 5)])
    output_validation = (outputs[n - int(n / 5):])


    models = []

    for input in inputs:
        model = Input(shape=(len(input[0]),))
        # model = Dense(32)(_model)

        models.append(model)

    x = concatenate(models)
    # x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)

    """
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    """

    x = Dense(len(outputs[0]), activation='relu')(x)

    model = Model(inputs=models, outputs=x)

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()

    hist = model.fit([np.array(i) for i in inputs_by_attr[1]], np.array(output_train),
                     epochs=1000, batch_size=64,
                     validation_data=([np.array(i) for i in inputs_by_attr[2]], np.array(output_validation)),
                     verbose=2)

    scores = model.evaluate([np.array(i) for i in inputs_by_attr[0]], np.array(output_test), verbose=2)
    print('complete: %s = %.2f%%' % (model.metrics_names[1], scores[1] * 100))
