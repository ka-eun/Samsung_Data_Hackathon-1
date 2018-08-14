from preprocessing import preprocessing, shuffleList, deleteColumn
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import concatenate, Input, BatchNormalization
from keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import operator
from pprint import pprint


SEED = 448
np.random.seed(SEED)
tf.set_random_seed(SEED)
# random.seed(SEED)


def plot_hist(hist):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()


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
        inputs_by_attr[1].append(col[int(n / 10):n - int(n / 5)])  # train
        inputs_by_attr[2].append(col[n - int(n / 5):])  # validation

    n = len(_outputs)
    output_test = (_outputs[:int(n / 10)])
    output_train = (_outputs[int(n / 10):n - int(n / 5)])
    output_validation = (_outputs[n - int(n / 5):])

    return inputs, _outputs,\
           inputs_by_attr[0], inputs_by_attr[1], inputs_by_attr[2],\
           output_test, output_train, output_validation


def majority(votes):
    # value
    # rate
    return sorted(Counter(votes).items(), key=operator.itemgetter(1), reverse=True)[0][0],\
           (sorted(Counter(votes).items(), key=operator.itemgetter(1), reverse=True)[0][1] / len(votes))


def compare_lists(a, b):
    return sum([1 if a[i] == b[i] else 0 for i in range(len(a))])


# TO DO: ensemble
def createModel(inputs, rand=0):
    # inputs
    models = []
    for input in inputs:
        model = Input(shape=(len(input[0]),))
        models.append(model)

    # more layers for each one-hot encoding vector
    _models = []
    for i, model in enumerate(models):
        """
        model = Dense(len(inputs[i][0]), kernel_initializer='he_normal')(model)
        model = BatchNormalization()(model)
        model = Activation('elu')(model)
        """

        # collect refined model
        _models.append(model)

    # merge
    x = concatenate(_models)

    """
    random parts for ensemble
    """
    x = Dense(200, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    # x = Dropout(0.2)(x)

    # output
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

    # for ensemble
    num_models = 10
    models = []
    for i in range(num_models):
        """
        create model
        """
        model = createModel(inputs)
        models.append(model)

    scores = []
    predicts = []
    for j, model in enumerate(models):
        """
        training
        """
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        # model.summary()

        # early stopping
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0)

        # train
        hist = model.fit([np.array(i) for i in input_train], np.array(output_train),
                         epochs=100, batch_size=pow(2, 13),
                         validation_data=([np.array(i) for i in input_val], np.array(output_val)),
                         callbacks=[early_stopping],
                         verbose=0)

        # plot_hist(hist)

        """
        test
        """
        score = model.evaluate([np.array(i) for i in input_test], np.array(output_test), verbose=0)
        # print('complete: %s = %.2f%%' % (model.metrics_names[1], score[1] * 100))
        scores.append(score)

        # better than random(0.5)
        # TO DO: threshold
        if score[1] <= 0.5:
            print('fail: model', j)

        else:
            """
            predict
            """
            _preds = model.predict([np.array(i) for i in input_test])

            preds = []
            for i, _pred in enumerate(_preds):
                preds.append([])
                for val in _pred:
                    preds[i].append(int(max(0, round(val))))

            predicts.append(preds)

            print('complete: model %d: %.2f%%'
                  % (j, sum([compare_lists(preds[i], output_test[i])
                             for i in range(len(preds))], 0.0) / (len(preds) * len(preds[0])) * 100))

    """
    ensemble
    """
    # scores
    print([score[1] * 100 for score in scores])

    # TO DO: select top N model (optional)

    # majority rule
    _collect = []
    for _ in range(len(predicts[0])):
        _collect.append([])

    for predict in predicts:
        # print(predict)
        for i, row in enumerate(predict):
            _collect[i].append(row)

    # print(_collect)

    collect = []
    for i, rows in enumerate(_collect):
        tmp = [[] for _ in range(len(rows[0]))]
        for row in rows:
            for j, elem in enumerate(row):
                tmp[j].append(elem)

        collect.append(tmp)

    # print(collect)

    results = []
    rates = []
    for row in collect:
        _results = []
        _rates = []
        for votes in row:
            val, rate = majority(votes)
            _results.append(val)
            _rates.append(rate)

        results.append(_results)
        rates.append(_rates)

    print(rates)
    print(results)
    print(output_test)

    """
    evaluate
    """
    print('finally: %.2f%%'
          % (sum([compare_lists(results[i], output_test[i])
                  for i in range(len(results))], 0.0) / (len(results) * len(results[0])) * 100))
