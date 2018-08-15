import csv
from operator import eq
from pprint import pprint


def find_row9(_row):  # 사고유형_대분류 찾는 함수
    if eq(_row[15], "없음"):
        _row[9] = "차량단독"
    elif eq(_row[15], "보행자") | eq(_row[10], "보도통행중") | eq(_row[10], "횡단중"):
        _row[9] = "차대사람"
    elif eq(_row[14], "열차") | eq(_row[15], "열차") | eq(_row[12], "건널목") | eq(_row[13], "건널목"):
        _row[9] = "건널목"
    else:
        _row[9] = "차대차"

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
    _row = find_row9(_row)
    _row = find_row15(_row)
    _row = find_row12(_row)
    _row = find_row13(_row)
    return _row


if __name__ == "__main__":
    f = open('./test_kor.csv', 'r')
    f2 = open('./outputTest_kor.csv', 'w', newline='')
    r = csv.reader(f)
    wr = csv.writer(f2)

    for row in r:
        tmp = []
        for i, elem in enumerate(row):
            if i<2:
                tmp.append(elem)
            elif i>6:
                tmp.append(elem)

        if '' in tmp:  # tmp의 어느 한 요소라도 비어있으면 빈 리스트를 삽입하고 다음 row를 읽음
            row = find_row(row)
            wr.writerow(row)
        else:
            wr.writerow(row)

    f.close()
    f2.close()
