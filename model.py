from preprocessing import preprocessing, shuffleList, deleteColumn
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import concatenate, BatchNormalization
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pprint import pprint


# np.random.seed(SEED)
# random.seed(SEED)


def separateSet(_inputs, _outputs):
    # inputs: attribute에 따라 생성한 list
    inputs = []
    for _ in range(len(_inputs[0])):
        inputs.append([])

    for _input in _inputs:
        for i, elem in enumerate(_input):
            inputs[i].append(elem)

    # train, test, validation
    inputs_by_attr = []

    for _ in range(3):
        inputs_by_attr.append([])

    for i, col in enumerate(inputs):
        n = len(col)
        inputs_by_attr[0].append(col[:int(n / 10)])  # test
        inputs_by_attr[1].append(col[int(n / 10):n - int(n / 10)])  # train
        inputs_by_attr[2].append(col[n - int(n / 10):])  # validation

    n = len(_outputs)
    output_test = (_outputs[:int(n / 10)])
    output_train = (_outputs[int(n / 10):n - int(n / 10)])
    output_validation = (_outputs[n - int(n / 10):])

    return inputs, _outputs,\
           inputs_by_attr[0], inputs_by_attr[1], inputs_by_attr[2],\
           output_test, output_train, output_validation


def createModel(inputs):
    # inputs
    models = []
    for input in inputs:
        model = Input(shape=(len(input[0]),))
        models.append(model)

    # more layers for each one-hot encoding vector
    _models = []
    for model in models:
        """
        """
        # x = Dense(200, kernel_initializer='he_normal')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        # collect refined model
        _models.append(model)

    # merge
    x = concatenate(_models)

    x = Dense(200, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(len(outputs[0]))(x)
    x = Activation('relu')(x)

    return Model(inputs=models, outputs=x)


if __name__ == "__main__":
    _inputs, _outputs, _ = preprocessing()

    # delete columns
    _inputs = deleteColumn(_inputs, [7, 9, 12])
    _outputs = deleteColumn(_outputs, [2])

    _inputs, _outputs = shuffleList(_inputs, _outputs)
    inputs, outputs, input_test, input_train, input_val, output_test, output_train, output_val\
        = separateSet(_inputs, _outputs)

    """
    create model
    """
    model = createModel(inputs)

    """
    training & test
    """
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

    model.summary()

    # train
    hist = model.fit([np.array(i) for i in input_train], np.array(output_train),
                     epochs=100, batch_size=8192,
                     validation_data=([np.array(i) for i in input_val], np.array(output_val)),
                     verbose=2)

    # test
    scores = model.evaluate([np.array(i) for i in input_test], np.array(output_test), verbose=2)
    print('complete: %s = %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    # predict
    _preds = model.predict([np.array(i) for i in input_test])

    preds = []
    for i, _pred in enumerate(_preds):
        preds.append([])
        for val in _pred:
            preds[i].append(int(max(0, round(val))))

    print(preds)
