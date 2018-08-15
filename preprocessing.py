import csv  # comma로 분리된 데이터를 읽고 쓰기 위한 모듈
from pprint import pprint
import random


def shuffleList(_input, _output):
    SEED = 448

    random.Random(SEED).shuffle(_input)
    random.Random(SEED).shuffle(_output)
    return _input, _output


# 클래스를 orthonormal vector화하는 함수
# 서로 직교하며 크기가 1로 같은 orthonormal vector를 사용하여 input data를 수치화함
def oneHotEncoding(length, index):
    vector = []
    for _ in range(length):  # 크기가 length인 벡터 생성
        vector.append(0)
    vector[index] = 1
    return vector


def preprocessing(file_train='./Kor_Train_교통사망사고정보(12.1~17.6).csv', file_test='./test_kor.csv'):
    f = open(file_train, 'r')
    r = csv.reader(f)

    f2 = open(file_test)
    r2 = csv.reader(f2)

    attr_test = []
    for row in r2:  # test_kor를 열 별로
        for elem in row:  # test_kor를 열의 원소 별로
            attr_test.append(elem)
        break

    attr_train = []
    for row in r:  # 교통사망정보(트레이닝.csv) 열 별로
        for elem in row:  # 교통사망정보 열의 원소 별로
            attr_train.append(elem)
        break

    _input_train = []  # 사람 수 데이타를 제외한 인풋 정보
    output_train = []  # 사람 수 데이터(사망자, 사상자, 중상자,  경상자,부상신고자의 수)

    # 위에서 row로 반복문을 돌려서 맨처음 row인 자료 정보 분류는 불포함하여 반복문을 돌림
    for row in r:  # 교통사망정보 열 별로(맨 처음 row였던 자료 분류는 불포함)
        tmp = []
        for i, elem in enumerate(row):  # 교통사망정보 열만큼 돌림
            if attr_train[i] in attr_test:  # 교통사망정보 열의 원소 중 test_kor와 일치하는 원소만 append
                tmp.append(elem)

        input = []
        output = []

        for i, elem in enumerate(tmp):  # 인덱스로 인풋 아웃풋 분류(2<=인풋<=7)
            if i < 2:
                input.append(elem)
            elif i < 7:
                output.append(int(elem))
            else:
                input.append(elem)

        _input_train.append(input)
        output_train.append(output)

    f.close()
    f2.close()

    """
    """
    tmp = []
    for i in range(len(_input_train[0])):
        tmp.append([])  # _input_train의 attribute 개수만큼 list 생성

    for row in _input_train:
        for i, elem in enumerate(row):
            tmp[i].append(elem)  # 각 attribute에 해당하는 element를 list형태로 tmp에 저장

    input_count = []
    input_set = []
    for i, c in enumerate(tmp):
        if i == (len(tmp) - 2):  # default file_train의 마지막 두 attribute의 element가 일부 겹치므로 union set으로 통일
            input_count.append(len(set(c) | set(tmp[i + 1])))
            input_set.append(set(c) | set(tmp[i + 1]))
            break
        else:
            input_count.append(len(set(c)))  # 각 attribute가 가지는 중복되지 않는 element의 수를 input_count list에 저장
            input_set.append(set(c))  # tmp의 각 리스트를 set으로 변환하여 input_set에 저장

    dic_list = []
    for _ in range(len(input_set)):  # file_train의 attribute 개수만큼 dictionary 생성
        dic_list.append({})

    for i, B in enumerate(input_set):
        for j, b in enumerate(B):
            dic_list[i][b] = oneHotEncoding(input_count[i],
                                            j)  # i번째 attribute에 해당되는 dictionary에서 키값을 b로 하는 벡터화된 value를 맵핑

    dic_list.append(dic_list[len(dic_list) - 1])  # input_count와 크기를 맞춰주기 위해 마지막번째 attribute에 병합된 set을 추가

    """
    """
    input_train = []  # 사람 수 데이타를 제외한 input을 벡터화한 list
    for i, row in enumerate(_input_train):
        input_train.append([])
        for j, elem in enumerate(row):
            input_train[i].append(dic_list[j][elem])

    return input_train, output_train, dic_list


def deleteColumn(rows, stopIdx):
    tmp = []
    for row in rows:
        refined = []
        for i, elem in enumerate(row):
            if i not in stopIdx:
                refined.append(elem)
        tmp.append(refined)

    return tmp


if __name__ == "__main__":
    train_input, train_output, dicts = preprocessing()
    print(train_input)
    print(train_output)
    print(dicts)
