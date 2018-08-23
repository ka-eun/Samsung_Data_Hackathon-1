from preprocessing import category_P_preprocessing
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

    # 1~2 layers
    for _ in range(random.randrange(1, 3)):
        x = Dense(random.randrange(300, 500), kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(0.5)(x)

    # output 1
    x = Dense(len(outputs[0][0]))(x)
    o1 = Activation('softmax', name='P')(x)

    """
    # output 2
    o2 = Dense(random.randrange(300, 500), kernel_initializer='glorot_normal')(x)
    o2 = BatchNormalization()(o2)
    o2 = Activation('tanh')(o2)
    o2 = Dropout(0.5)(o2)

    o2 = Dense(len(outputs[1][0]))(o2)
    o2 = Activation('softmax', name='L')(o2)
    """

    # return Model(inputs=models, outputs=[o1, o2])
    return Model(inputs=models, outputs=[o1])


def main_7():
    if not os.path.isdir('./outputs'):
        os.mkdir('./outputs')

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    """
    preprocessing
    """
    # 범주형 데이터 리스트, 사람 수 데이터 리스트, 벡터화 dictionary
    _inputs, _outputs, input_dict, output_dict = category_P_preprocessing()

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
        if score[1] < 0.77:
            print('fail    : model %d: %s = %.2f%%' % (cnt, model.metrics_names[1], score[1] * 100))
            continue

        # PASS
        print('complete: model %d: %s = %.2f%%' % (cnt, model.metrics_names[1], score[1] * 100))

        """
        save model
        """
        model.save('./models/7_model_' + str(cnt) + '.h5')

        try:
            plot_model(model, to_file='./models/7_model_' + str(cnt) + '.png', show_shapes=True, show_layer_names=True)
        except:
            pass

        models.append(model)

        # model의 개수가 10개가 될 때
        cnt += 1

    """
    predict
    """
    f1 = open('./outputs/6_output.csv', 'r')
    f2 = open('./outputs/7_output.csv', 'w', newline='')
    r1 = csv.reader(f1)
    r2 = csv.writer(f2)

    for i, row in enumerate(r1):
        # blank
        if (23 < i & i < 31):
            """
            preprocessing
            """
            # copy
            _row = row[:]

            # P열 제외
            for k, j in enumerate([15]):
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
            pred = []
            for model in models:
                _pred = list(model.predict([np.array([j]) for j in input_pred])[0])
                pred.append(value_key_map(output_dict, 0, oneHotEncoding(len(_pred), _pred.index(max(_pred)))))

            # majority
            val, rate = majority(pred)
            print('save    : row  %2d: val = %.2f%%' % (i + 1, rate * 100))

            """
            write
            """
            row[15] = val
            r2.writerow(row)

        else:
            r2.writerow(row)

    f1.close()
    f2.close()


if __name__ == "__main__":
    main_7()
