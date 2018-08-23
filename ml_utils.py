from collections import Counter
import operator
import random
import matplotlib.pyplot as plt


# 클래스를 orthonormal vector화하는 함수
# 서로 직교하며 크기가 1로 같은 orthonormal vector를 사용하여 input data를 수치화함
def oneHotEncoding(length, index):
    vector = []
    for _ in range(length):  # 크기가 length인 벡터 생성
        vector.append(0)
    vector[index] = 1
    return vector


def shuffleList(_input, _output):
    SEED = 448

    random.Random(SEED).shuffle(_input)
    random.Random(SEED).shuffle(_output)
    return _input, _output


# 각 row에서 인자로 받은 index를 제외시킨 리스트를 리턴하는 함수
def deleteColumn(rows, stopIdx):
    tmp = []
    for row in rows:
        refined = []
        # 인자로 받은 stopIdx에 해당 index가 있으면 그 값을 제외한 리스트를 tmp에 append
        for i, elem in enumerate(row):
            if i not in stopIdx:
                refined.append(elem)
        tmp.append(refined)

    return tmp


def norm_list(input):
    return [1 for _ in range(len(input))] if min(input) == max(input) else [(i - min(input)) / (max(input) - min(input)) for i in input]


def majority(votes):
    # 만들어진 모델들에 대해서 다수결 투표하는 함수
    # 투표 개수를 세어서 표결수가 큰 순서대로 정렬한 후 가장 많은 표를 받은 키값
    # 투표 개수를 세어서 표결수가 큰 순서대로 정렬한 후 가장 많은 표를 받은 키값의 표결수를 투표자 수로 나눔. 즉 비율
    return sorted(Counter(votes).items(), key=operator.itemgetter(1), reverse=True)[0][0], \
           (sorted(Counter(votes).items(), key=operator.itemgetter(1), reverse=True)[0][1] / len(votes))


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
def separate_set_one_output(_inputs, _outputs):
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

    return inputs, _outputs, \
           inputs_by_attr[0], inputs_by_attr[1], inputs_by_attr[2], \
           output_test, output_train, output_validation


#리스트 inputs을 test,train,validation 세 리스트로 분화하는 함수
def separate_set_multiple_outputs(_inputs, _outputs):
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

    """
    """
    outputs = []
    for _ in range(len(_outputs[0])):
        outputs.append([])

    for _output in _outputs:
        for i, elem in enumerate(_output):
            outputs[i].append(elem)

    # train, test, validation
    outputs_by_attr = []

    for _ in range(3):
        outputs_by_attr.append([])

    for i, col in enumerate(outputs):
        n = len(col)
        outputs_by_attr[0].append(col[:int(n / 10)])  # test
        outputs_by_attr[1].append(col[int(n / 10):n - int(n / 10)])  # train
        outputs_by_attr[2].append(col[n - int(n / 10):])  # validation

    """
    return
    """
    return inputs, outputs, \
           inputs_by_attr[0], inputs_by_attr[1], inputs_by_attr[2], \
           outputs_by_attr[0], outputs_by_attr[1], outputs_by_attr[2]


# 사전 value로 key 찾기
def value_key_map(dicts, col, val):
    for key, elem in dicts[col].items():
        if elem == val:
            return key
