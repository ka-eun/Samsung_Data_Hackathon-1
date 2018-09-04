import csv
from operator import eq
import os
from pprint import pprint


def map_cities(file_train='./train_kor.csv'):
    f1 = open(file_train, 'r')
    r = csv.reader(f1)
    f2 = open('cities.csv', 'w', newline='')
    wr = csv.writer(f2)

    big_city = []
    small_city=[]

    for j, row in enumerate(r):
        if j!=0:
            for i, elem in enumerate(row):
                if i==7:
                    big_city.append(elem)

    big_city = list(set(big_city))

    dic = {}
    for i in range(len(big_city)):
        small_city.append([])
        dic[big_city[i]] = small_city[i]

    f1.close();
    f1 = open(file_train, 'r')
    r = csv.reader(f1)

    for j, row in enumerate(r):
        if j != 0:
            dic[row[7]].append(row[8])

    for i, elem in enumerate(big_city):
        dic[elem] = list(set(dic[elem]))
        # dic[elem].insert(0, elem)
        # wr.writerow(dic[elem])

    return  dic
    # print(big_city)
    # print(dic)

    f1.close()
    f2.close()


def find_row7(_row):  # 발생지시도 찾는 함수
    dic = map_cities('./train_kor.csv')

    tmp = []
    for i, elem in enumerate (dic.keys()):
        tmp.extend(dic[elem])
    '''
    print(tmp)

    while True:
        pass
    '''
    count = {}
    for i, elem in enumerate(list(set(tmp))):
        _count = 0
        for j, _elem in enumerate(tmp):
            if eq(elem, _elem):
                _count += 1
        count[elem] = _count

    # print(count)
    for i, elem in enumerate(dic.keys()):
        try:
            if (_row[8] in dic[elem]) & (count[_row[8]] == 1):
                _row[7] = elem
        except:
            pass

    return _row


def find_row8(_row):  # 발생지시군구 찾는 함수
    if eq(_row[7], "세종"):
        _row[8] = "세종"

    return _row


def find_row9(_row):  # 사고유형_대분류 찾는 함수
    noCarList = [" ", "열차", "없음"]
    if eq(_row[10], "주/정차차량 충돌") | eq(_row[10], "도로이탈") | eq(_row[10], "공작물충돌") | eq(_row[10], "전도전복") | \
            eq(_row[10], "전도") | eq(_row[10], "전복") | eq(_row[15], "없음"):
        _row[9] = "차량단독"
    elif eq(_row[15], "보행자") | eq(_row[10], "차도통행중") | eq(_row[10], "길가장자리구역통행중") | eq(_row[10], "보도통행중") | eq(_row[10], "횡단중"):
        _row[9] = "차대사람"
    elif eq(_row[10], "경보기무시") | eq(_row[10], "직전진행") | eq(_row[10], "차단기돌파") | \
            eq(_row[12], "건널목") | eq(_row[13], "건널목") | eq(_row[14], "열차") | eq(_row[15], "열차") :
        _row[9] = "건널목"
    elif eq(_row[10], "후진중충돌") | eq(_row[10], "측면충돌") | eq(_row[10], "측면직각충돌") | eq(_row[10], "정면충돌") | eq(_row[10], "추돌") | \
        (_row[15] not in noCarList) :
        _row[9] = "차대차"

    # _row[10]이 '기타'이고 _row[15]가 빈칸인 경우 알 수 없음

    return _row


def find_row12(_row):  # 도로형태_대분류 찾는 함수
    if eq(_row[13], "기타단일로") | eq(_row[13], "터널안") | eq(_row[13], "횡단보도상") | eq(_row[13], "횡단보도부근") | \
            eq(_row[13], "교량위") | eq(_row[13], "지하차도(도로)내"):
        _row[12] = "단일로"
    elif eq(_row[13], "교차로부근") | eq(_row[13], "교차로내") | eq(_row[13], "교차로횡단보도내"):
        _row[12] = "교차로"
    elif eq(_row[13], "주차장"):
        _row[12] = "주차장"
    elif eq(_row[13], "기타"):
        _row[12] = "기타"
    elif eq(_row[13], "기타/불명"):
        _row[12] = "기타/불명"
    elif eq(_row[13], "불명"):
        _row[12] = "불명"
    elif eq(_row[13], "지하도로내"):
        _row[12] = "지하도로내"
    elif eq(_row[13], "건널목") | eq(_row[9], "건널목"):
        _row[12] = "건널목"
    elif eq(_row[13], "고가도로위"):  # 0.934의 확률로 고가도로위에 해당, 이외에는 단일로에 해당
        _row[12] = "고가도로위"

    return _row


def find_row13(_row):  # 도로형태 찾는 함수
    if eq(_row[12], "건널목"):
        _row[13] = "건널목"
    elif eq(_row[12], "지하도로내"):
        _row[13] = "지하도로내"
    elif eq(_row[12], "기타"):
        _row[13] = "기타"
    elif eq(_row[12], "기타/불명"):
        _row[13] = "기타/불명"

    return _row


def find_row15(_row):  # 당사자종별_2당_대분류 찾는 함수
    if eq(_row[9], "차대사람") | eq(_row[10], "보도통행중") | eq(_row[10], "횡단중"):
        _row[15] = "보행자"
    elif eq(_row[9], "차량단독"):
        _row[15] = "없음"

    return _row



def find_row(_row):
    _row = find_row7(_row)
    _row = find_row8(_row)
    _row = find_row9(_row)
    _row = find_row15(_row)
    _row = find_row12(_row)
    _row = find_row13(_row)
    return _row


def main_1():
    if not os.path.isdir('./outputs'):
        os.mkdir('./outputs')

    f = open('./test_kor.csv', 'r')
    f2 = open('./outputs/1_output.csv', 'w', newline='')
    r = csv.reader(f)
    wr = csv.writer(f2)


    for row in r:
        tmp = []
        for i, elem in enumerate(row):
            if i < 2:
                tmp.append(elem)
            elif i > 6:
                tmp.append(elem)

        # tmp의 어느 한 요소라도 비어있으면 빈 리스트를 삽입하고 다음 row를 읽음
        if '' in tmp:
            row = find_row(row)
            wr.writerow(row)
        else:
            wr.writerow(row)

    f.close()
    f2.close()


if __name__ == "__main__":
    main_1()

