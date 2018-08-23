from preprocessing import category_H_and_I_preprocessing
from ml_utils import shuffleList, deleteColumn, separate_set_multiple_outputs
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

    # 1 layers
    x = Dense(random.randrange(300, 500), kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Dropout(0.5)(x)

    # output 1
    o1 = Dense(random.randrange(300, 500), kernel_initializer='glorot_normal')(x)
    o1 = BatchNormalization()(o1)
    o1 = Activation('tanh')(o1)
    o1 = Dropout(0.5)(o1)

    o1 = Dense(len(outputs[0][0]))(o1)
    o1 = Activation('softmax', name='H')(o1)

    # output 2
    o2 = Dense(random.randrange(300, 500), kernel_initializer='glorot_normal')(x)
    o2 = BatchNormalization()(o2)
    o2 = Activation('tanh')(o2)
    o2 = Dropout(0.5)(o2)

    o2 = Dense(len(outputs[1][0]))(o2)
    o2 = Activation('softmax', name='I')(o2)

    return Model(inputs=models, outputs=[o1, o2])


def main_9():
    if not os.path.isdir('./outputs'):
        os.mkdir('./outputs')

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    """
    preprocessing
    """
    # 범주형 데이터 리스트, 사람 수 데이터 리스트, 벡터화 dictionary
    _inputs, _outputs, input_dict, output_dict = category_H_and_I_preprocessing()

    # delete columns
    _inputs = deleteColumn(_inputs, [3])  # 사상자 수 제외

    # make list
    tmp = []
    for _input in _inputs:
        ttmp = []
        nums = []
        for i, elem in enumerate(_input):
            if i < 2:
                ttmp.append(elem)
            elif i < 6:
                nums.append(elem)
            elif i == 6:
                ttmp.append(norm_list(nums))
                ttmp.append(elem)
            else:
                ttmp.append(elem)

        tmp.append(ttmp)

    _inputs, _outputs = shuffleList(tmp, _outputs)  # 데이터 리스트가 고루 섞이도록 _inputs와 _outputs를 함께 섞음
    inputs, outputs, input_test, input_train, input_val, output_test, output_train, output_val \
        = separate_set_multiple_outputs(_inputs, _outputs)  # 범주형 데이터와 사람 수 데이터를 각각 test, train, validate를 위해 분류

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
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        # model.summary()

        # early stopping
        # val_loss값이 10번 동안 개선되지 않으면 해당 model의 학습을 중단
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

        # train
        # model의 학습 이력 정보로 train의 loss와 accuracy, val의 loss와 accuracy 값을 받음
        hist = model.fit([np.array(i) for i in input_train], [np.array(i) for i in output_train],
                         epochs=1000, batch_size=pow(2, 13),
                         validation_data=([np.array(i) for i in input_val], [np.array(i) for i in output_val]),
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
        score = model.evaluate([np.array(i) for i in input_test], [np.array(i) for i in output_test], verbose=0)

        # TH = 0.72
        if (score[3] < 0.19) | (score[4] < 0.02):
            print('fail    : model %d: %s = %.2f%%' % (cnt, model.metrics_names[3], score[3] * 100))
            print('fail    : model %d: %s = %.2f%%' % (cnt, model.metrics_names[4], score[4] * 100))
            continue

        # PASS
        print('complete: model %d: %s = %.2f%%' % (cnt, model.metrics_names[3], score[3] * 100))
        print('complete: model %d: %s = %.2f%%' % (cnt, model.metrics_names[4], score[4] * 100))

        """
        save model
        """
        model.save('./models/9_model_' + str(cnt) + '.h5')

        try:
            plot_model(model, to_file='./models/9_model_' + str(cnt) + '.png', show_shapes=True, show_layer_names=True)
        except:
            pass

        models.append(model)

        # model의 개수가 10개가 될 때
        cnt += 1

    """
    predict
    """
    f1 = open('./outputs/8_output.csv', 'r')
    f2 = open('./outputs/9_output.csv', 'w', newline='')
    r1 = csv.reader(f1)
    r2 = csv.writer(f2)

    for i, row in enumerate(r1):
        # blank
        if (35 < i) & (i < 41):
            """
            preprocessing
            """
            # copy
            _row = row[:]

            # H, I열 제외
            for k, j in enumerate([7, 8]):
                del(_row[j-k])

            for j, elem in enumerate(_row):
                if 1 < j & j < 7:
                    _row[j] = int(elem)

            _input_pred = []
            for j, elem in enumerate(_row):
                try:
                    if isinstance(elem, int):
                        _input_pred.append(elem)
                    else:
                        _input_pred.append(input_dict[j][elem])
                except:
                    _input_pred.append('')

            # _inputs = deleteColumn(_inputs, [3, 12, 13, 14])  # 사상자 수,  제외
            for k, j in enumerate([3]):
                del (_input_pred[j - k])

            # make list
            input_pred = []
            nums = []
            for j, elem in enumerate(_input_pred):
                if j < 2:
                    input_pred.append(elem)
                elif j < 6:
                    nums.append(elem)
                elif j == 6:
                    input_pred.append(norm_list(nums))
                    input_pred.append(elem)
                else:
                    input_pred.append(elem)

            """
            predict
            """
            p1 = []
            p2 = []
            for model in models:
                _pred = model.predict([np.array([j]) for j in input_pred])
                _p1 = list(_pred[0][0])
                _p2 = list(_pred[1][0])
                p1.append(value_key_map(output_dict, 0, oneHotEncoding(len(_p1), _p1.index(max(_p1)))))
                p2.append(value_key_map(output_dict, 1, oneHotEncoding(len(_p2), _p2.index(max(_p2)))))

                # _pred = list(model.predict([np.array([j]) for j in input_pred])[0])
                # pred.append(value_key_map(output_dict, 0, oneHotEncoding(len(_pred), _pred.index(max(_pred)))))

            # majority
            val1, rate1 = majority(p1)
            val2, rate2 = majority(p2)
            print('save    : row  %2d: H_val = %.2f%%' % (i + 1, rate1 * 100))
            print('save    : row  %2d: I_val = %.2f%%' % (i + 1, rate2 * 100))

            """
            write
            """
            row[7] = val1
            row[8] = val2
            r2.writerow(row)

        else:
            r2.writerow(row)

    f1.close()
    f2.close()


if __name__ == "__main__":
    main_9()
