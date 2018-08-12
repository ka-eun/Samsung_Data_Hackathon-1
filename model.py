from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import concatenate, BatchNormalization
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import random

SEED = 448
  #리스트 inputs을 test,train,validation 세 리스트로 분화하는 함수
def separateSet(_inputs, _outputs):
    # inputs: attribute에 따라 생성한 list
    inputs = []
    for _ in range(len(_inputs[0])):  #inputs의 원소로 attr의 개수만큼의 리스트를 만듬
        inputs.append([])

    for _input in _inputs:
        for i, elem in enumerate(_input):  #_inputs 속성별로 재분류해서 inputs에
            inputs[i].append(elem)

    # train, test, validation
    inputs_by_attr = []

    for _ in range(3):
        inputs_by_attr.append([])

    for i, col in enumerate(inputs):
        n = len(col)
        inputs_by_attr[0].append(col[:int(n / 5)])  # test
        inputs_by_attr[1].append(col[int(n / 5):n - int(n / 5)])  # train
        inputs_by_attr[2].append(col[n - int(n / 5):])  # validation

    n = len(_outputs)
    output_test = (_outputs[:int(n / 5)])
    output_train = (_outputs[int(n / 5):n - int(n / 5)])
    output_validation = (_outputs[n - int(n / 5):])

    return inputs, _outputs,\
           inputs_by_attr[0], inputs_by_attr[1], inputs_by_attr[2],\
           output_test, output_train, output_validation
  #파라미터로 받은 리스트의 원소를 랜덤하게 재배치하는 함수
def shuffleList(_input, _output):
    random.Random(SEED).shuffle(_input)
    random.Random(SEED).shuffle(_output)
    return _input, _output


if __name__ == "__main__":
    _inputs, _outputs, _ = preprocessing()
    _inputs, _outputs = shuffleList(_inputs, _outputs)
    inputs, outputs, input_test, input_train, input_val, output_test, output_train, output_val = separateSet(_inputs, _outputs)

    """
    """
    models = []

    for input in inputs: #inputs의 열 별로
        model = Input(shape=(len(input[0]),))  #행은 각 속성의 분류 개수(예:주간,아간은 2)이고 열은 개수에 맞춰서 설정된 행렬
        # model = Dense(32)(_model)

        models.append(model)

    x = concatenate(models)  #하나의 행으로 합침
    # x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)  #노드가 64개이고 활성화 함수로 ReLu를 사용한 fully connected layer

    """
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    """

    x = Dense(len(outputs[0]), activation='relu')(x)

    model = Model(inputs=models, outputs=x)

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])  #모델 컴파일
    model.summary()

    hist = model.fit([np.array(i) for i in input_train], np.array(output_train),
                     epochs=10, batch_size=64,
                     validation_data=([np.array(i) for i in input_val], np.array(output_val)),
                     verbose=2)  #모델 학슴

    scores = model.evaluate([np.array(i) for i in input_test], np.array(output_test), verbose=2)  #인풋은 input_test의 각 원소를 numpy형식으로 하여 리스트로 묶어서 보냄
    print('complete: %s = %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    # predict
    print(model.predict([np.array(i) for i in input_test]))  #학습한 모델에 input_test를 입력한 결과를 출력
