from preprocessing import regression_preprocessing
from ml_utils import shuffleList, deleteColumn, separate_set_one_output
from ml_utils import majority, norm_list, plot_hist, value_key_map, oneHotEncoding
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import concatenate, Input, BatchNormalization
from keras.models import Model
from keras.utils.vis_utils import plot_model
import numpy as np
import os
import random
import csv
from math import exp


def compare_lists(a, b):
    return sum([1 if a[i] == b[i] else 0 for i in range(len(a))])


def evaluate_lists(n, m):
    res = 0.0

    for i, _ in enumerate(n):
        for j, _ in enumerate(n[i]):
            res += exp(-1 * pow(n[i][j] - m[i][j], 2))

    return res / (len(n) * len(n[0]))


def createModel(inputs, outputs):
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

        # 0~1 layer
        for _ in range(random.randrange(0, 2)):
            model = Dense(max(1, round(len(inputs[i][0]) * (random.random() + 0.5))), kernel_initializer='he_normal')(model)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            model = Dropout(0.5)(model)

        # collect refined model
        _models.append(model)

    # merge
    x = concatenate(_models)

    # 1~2 layers
    for _ in range(random.randrange(1, 3)):
        x = Dense(random.randrange(100, 400), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

    # output
    x = Dense(len(outputs[0]))(x)
    o1 = Activation('relu', name='num')(x)

    return Model(inputs=models, outputs=o1)


def main_14():
    if not os.path.isdir('./outputs'):
        os.mkdir('./outputs')

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    """
    preprocessing
    """
    _inputs, _outputs, input_dict = regression_preprocessing()  # 범주형 데이터 리스트, 사람 수 데이터 리스트, 벡터화 dictionary

    # delete columns
    _outputs = deleteColumn(_outputs, [1])  # 사람 수 데이터에서 사상자 수 column을 제외

    _inputs, _outputs = shuffleList(_inputs, _outputs)  # 데이터 리스트가 고루 섞이도록 _inputs와 _outputs를 함께 섞음
    inputs, outputs, input_test, input_train, input_val, output_test, output_train, output_val \
        = separate_set_one_output(_inputs, _outputs)  # 범주형 데이터와 사람 수 데이터를 각각 test, train, validate를 위해 분류

    # ensemble
    num_models = 11  # 11
    cnt = 0

    models = []
    while cnt < num_models:
        """
        create model
        """
        model = createModel(inputs, outputs)

        """
        training
        """
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        # model.summary()

        # early stopping
        # val_loss값이 10번 동안 개선되지 않으면 해당 model의 학습을 중단
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

        # train
        # model의 학습 이력 정보로 train의 loss와 accuracy, val의 loss와 accuracy 값을 받음
        hist = model.fit([np.array(i) for i in input_train], np.array(output_train),
                         epochs=1000, batch_size=pow(2, 13),
                         validation_data=([np.array(i) for i in input_val], np.array(output_val)),
                         callbacks=[early_stopping],
                         verbose=0)

        try:
            # plot_hist(hist)
            pass
        except:
            pass

        """
        test
        """
        # model의 성능 평가
        # score = model.evaluate([np.array(i) for i in input_test], np.array(output_test), verbose=0)
        # print('complete: %s = %.2f%%' % (model.metrics_names[1], score[1] * 100))

        _preds = model.predict([np.array(i) for i in input_test])

        preds = []
        for i, _pred in enumerate(_preds):
            preds.append([])
            for val in _pred:
                # 반올림된 예측값이 0보다 클 경우 preds 리스트에 추가, 음수일경우 0을 추가
                preds[i].append(int(max(0, round(val))))

        # 성능 평가
        res = evaluate_lists(preds, output_test)

        # TH = 0.90
        if res < 0.90:
            print('fail    : model %d: %.2f%%' % (cnt, res * 100))
            continue

        # PASS
        print('complete: model %d: %.2f%%' % (cnt, res * 100))

        """
        save model
        """
        model.save('./models/14_model_' + str(cnt) + '.h5')

        try:
            plot_model(model, to_file='./models/14_model_' + str(cnt) + '.png', show_shapes=True, show_layer_names=True)
        except:
            pass

        models.append(model)

        # model의 개수가 10개가 될 때
        cnt += 1

    """
    predict
    """
    f1 = open('./outputs/13_output.csv', 'r')
    f2 = open('./outputs/14_output.csv', 'w', newline='')
    r1 = csv.reader(f1)
    r2 = csv.writer(f2)

    for i, row in enumerate(r1):
        # blank
        if ((0 < i) & (i < 11)) | ((30 < i) & (i < 36)) | (45 < i):
            """
            preprocessing
            """
            # copy
            _row = row[:]

            # C~G열 제외
            for k, j in enumerate([2, 3, 4, 5, 6]):
                del(_row[j-k])

            input_pred = []
            for j, elem in enumerate(_row):
                try:
                    if isinstance(elem, int):
                        input_pred.append(elem)
                    else:
                        input_pred.append(input_dict[j][elem])
                except:
                    input_pred.append('')

            """
            predict
            """
            preds = []
            for k, model in enumerate(models):
                _pred = list(model.predict([np.array([j]) for j in input_pred])[0])

                preds.append([])
                for val in _pred:
                    # 반올림된 예측값이 0보다 클 경우 preds 리스트에 추가, 음수일경우 0을 추가
                    preds[k].append(int(max(0, round(val))))

            collect = []
            for j in range(len(preds[0])):
                collect.append([])

                for k in range(len(preds)):
                    collect[j].append(preds[k][j])

            # majority
            val1, rate1 = majority(collect[0])
            print('save    : row  %2d: 1_val = %.2f%%' % (i + 1, rate1 * 100))

            val2, rate2 = majority(collect[1])
            print('save    : row  %2d: 2_val = %.2f%%' % (i + 1, rate2 * 100))

            val3, rate3 = majority(collect[2])
            print('save    : row  %2d: 3_val = %.2f%%' % (i + 1, rate3 * 100))

            val4, rate4 = majority(collect[3])
            print('save    : row  %2d: 4_val = %.2f%%' % (i + 1, rate4 * 100))

            """
            write
            """
            if row[2] == '':
                row[2] = val1

            if row[4] == '':
                row[4] = val2

            if row[5] == '':
                row[5] = val3

            if row[6] == '':
                row[6] = val4

            r2.writerow(row)

        else:
            r2.writerow(row)

    f1.close()
    f2.close()


if __name__ == "__main__":
    main_14()
