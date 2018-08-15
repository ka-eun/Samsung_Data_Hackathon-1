from preprocessing import preprocessing, shuffleList, deleteColumn
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import concatenate, Input, BatchNormalization, PReLU
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import operator
import random
from math import exp
from pprint import pprint


# SEED = 448
# np.random.seed(SEED)
# tf.set_random_seed(SEED)
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


#리스트 inputs을 test,train,validation 세 리스트로 분화하는 함수
def separateSet(_inputs, _outputs):
    # inputs: attribute에 따라 생성한 list
    inputs = []
    for _ in range(len(_inputs[0])):
        #inputs의 원소로 attr의 개수만큼의 리스트를 만듬
        inputs.append([])

    for _input in _inputs:
        for i, elem in enumerate(_input):
            #_inputs 속성별로 재분류해서 inputs에
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


def majority(votes):
    #만들어진 모델들에 대해서 다수결 투표하는 함수
    # 투표 개수를 세어서 표결수가 큰 순서대로 정렬한 후 가장 많은 표를 받은 키값
    # 투표 개수를 세어서 표결수가 큰 순서대로 정렬한 후 가장 많은 표를 받은 키값의 표결수를 투표자 수로 나눔. 즉 비율
    return sorted(Counter(votes).items(), key=operator.itemgetter(1), reverse=True)[0][0],\
           (sorted(Counter(votes).items(), key=operator.itemgetter(1), reverse=True)[0][1] / len(votes))


def compare_lists(a, b):
    return sum([1 if a[i] == b[i] else 0 for i in range(len(a))])


def evaluate_lists(n, m):
    res = 0.0

    for i, _ in enumerate(n):
        for j, _ in enumerate(n[i]):
            res += exp(-1 * pow(n[i][j] - m[i][j], 2))

    return res / (len(n) * len(n[0]))


def createModel(inputs):
    # inputs
    models = []

    # inputs의 열 별로
    for input in inputs:
        # 행은 각 속성의 분류 개수(예:주간,아간은 2)이고 열은 개수에 맞춰서 설정된 행렬
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
        for _ in range(random.randrange(0, 2)):
            rand = random.randrange(3)

            # relu, elu
            if rand == 0:
                #relu에 맞는 웨이트 설정법
                #배치 정규화:입력값이 너무 차이가 나지 않게 입력값 정규화해서 넘겨줌(매 층마다 정규화)
                model = Dense(round(len(inputs[i][0])*(random.random()+1.0)), kernel_initializer='he_normal')(model)  # default node number = 200
                model = BatchNormalization()(model)
                activation = random.choice(['elu', 'relu'])
                model = Activation(activation, name=activation+'_'+str(i))(model)

            # PReLU
            elif rand == 1:
                model = Dense(round(len(inputs[i][0])*(random.random()+1.0)), kernel_initializer='he_normal')(model)  # default node number = 200
                model = BatchNormalization()(model)
                model = PReLU()(model)

            # tanh
            else:
                model = Dense(round(len(inputs[i][0])*(random.random()+1.0)), kernel_initializer='glorot_normal')(model)  # default node number = 200
                model = BatchNormalization()(model)
                model = Activation('tanh', name='tanh_'+str(i))(model)

        # collect refined model
        _models.append(model)

    # merge
    x = concatenate(_models)

    """
    random parts for ensemble
    """
    tot = i

    for i in range(random.randrange(1, 3)):
        rand = random.randrange(3)

        # relu, elu
        if rand == 0:
            x = Dense(random.randrange(100, 400), kernel_initializer='he_normal')(x)  # default node number = 200
            x = BatchNormalization()(x)
            activation = random.choice(['elu', 'relu'])
            x = Activation(activation, name=activation+'_'+str(i+tot+1))(x)

        # PReLU
        elif rand == 1:
            x = Dense(random.randrange(100, 400), kernel_initializer='he_normal')(x)  # default node number = 200
            x = BatchNormalization()(x)
            x = PReLU()(x)

        # tanh
        else:
            x = Dense(random.randrange(100, 400), kernel_initializer='glorot_normal')(x)  # default node number = 200
            x = BatchNormalization()(x)
            x = Activation('tanh', name='tanh_'+str(i+tot+1))(x)

    # x = Dropout(0.2)(x)

    # output
    x = Dense(len(outputs[0]))(x)
    x = Activation('relu', name='relu_'+str(i+tot+2))(x)

    return Model(inputs=models, outputs=x)


if __name__ == "__main__":
    _inputs, _outputs, _ = preprocessing()  # 범주형 데이터 리스트, 사람 수 데이터 리스트, 벡터화 dictionary

    # delete columns
    _inputs = deleteColumn(_inputs, [3, 4, 7])  # 범주형 데이터에서 발생지시군구, 사고유형대분류, 도로형태대분류 column 제외
    _outputs = deleteColumn(_outputs, [1])  # 사람 수 데이터에서 사상자 수 column을 제외

    _inputs, _outputs = shuffleList(_inputs, _outputs)  # 데이터 리스트가 고루 섞이도록 _inputs와 _outputs를 함께 섞음
    inputs, outputs, input_test, input_train, input_val, output_test, output_train, output_val\
        = separateSet(_inputs, _outputs)  # 범주형 데이터와 사람 수 데이터를 각각 test, train, validate를 위해 분류

    # for ensemble model
    num_models = 20
    models = []

    # model의 개수만큼 model 생성
    for i in range(num_models):
        """
        create model
        """
        model = createModel(inputs)
        models.append(model)

    scores = []
    predicts = []

    # 각 model에 대한 training
    for j, model in enumerate(models):
        """
        training
        """
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        # model.summary()

        # early stopping
        # val_acc값이 5번 동안 향상되지 않으면 해당 model의 학습을 중단
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=2)

        # train
        # model의 학습 이력 정보로 train의 loss와 accuracy, val의 loss와 accuracy 값을 받음
        hist = model.fit([np.array(i) for i in input_train], np.array(output_train),
                         epochs=100, batch_size=pow(2, 13),
                         validation_data=([np.array(i) for i in input_val], np.array(output_val)),
                         callbacks=[early_stopping],
                         verbose=2)

        # plot_hist(hist)

        """
        test
        """
        # model의 성능 평가
        score = model.evaluate([np.array(i) for i in input_test], np.array(output_test), verbose=0)
        # print('complete: %s = %.2f%%' % (model.metrics_names[1], score[1] * 100))

        """
        predict
        """
        _preds = model.predict([np.array(i) for i in input_test])

        preds = []
        for i, _pred in enumerate(_preds):
            preds.append([])
            for val in _pred:
                # 반올림된 예측값이 0보다 클 경우 preds 리스트에 추가, 음수일경우 0을 추가
                preds[i].append(int(max(0, round(val))))

        # 성능 평가
        res = evaluate_lists(preds, output_test)

        # Threshold
        if res <= 0.9:
            print('fail    : model %d: %.2f%%' % (j, res * 100))
        else:
            scores.append(score)
            predicts.append(preds)
            plot_model(model, to_file='./models/model_plot_'+str(j)+'.png', show_shapes=True, show_layer_names=True)

            # 예측값과 실제output값을 비교한 compare_lists를 해당 리스트의 row길이, 개수만큼 나누어 정확도를 구함
            print('complete: model %d: %.2f%%' % (j, res * 100))

    """
    ensemble
    """
    # 각 model의 score 출력
    print([score[1] * 100 for score in scores])

    # TO DO: select top N model (optional)

    # majority rule
    _collect = []
    for _ in range(len(predicts[0])):
        _collect.append([])  # 예측 데이터의 종류만큼 빈 리스트 생성

    for predict in predicts:
        # print(predict)
        for i, row in enumerate(predict):
            _collect[i].append(row)  # 예측 데이터의 종류만큼 예측값들을 리스트에 추가

    # print(_collect)

    # collect에 예측 데이터의 종류만큼 빈 리스트를 만들고 각 리스트에 해당 종류의 예측값을 저장
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
            # 각 속성의 예측값 중 majority인 값과 그 비율을 받음
            val, rate = majority(votes)
            _results.append(val)
            _rates.append(rate)

        results.append(_results)
        rates.append(_rates)

    print(rates)
    # print(results)
    # print(output_test)

    """
    evaluate
    """
    print('finally: %.2f%%' % (evaluate_lists(results, output_test) * 100))
