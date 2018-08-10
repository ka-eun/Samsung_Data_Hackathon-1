import csv

f=open('C:\Users\user\Downloads\dataset_kor\교통사망사고정보\Kor_Train_교통사망사고정보(12.1~17.6).csv')
r=csv.reader(f)

f2=open('C:\Users\user\Downloads\test_kor.csv')
r2=csv.reader(f2)

attr_test = []
for row in r2:  #test_kor를 열 별로
    for elem in row:    #test_kor를 열의 원소 별로
        attr_test.append(elem)
    break

attr_train = []
for row in r:   #교통사망정보(트레이닝.csv) 열 별로
    for elem in row:    #교통사망정보 열의 원소 별로
        attr_train.append(elem)
    break

input_train = []    #사람 수 데이타를 제외한 인풋 정보
output_train = []   #사람 수 데이터(사망자, 사상자, 중상자,  경상자,부상신고자의 수)

#위에서 row로 반복문을 돌려서 맨처음 row인 자료 정보 분류는 불포함하여 반복문을 돌림
for row in r:   #교통사망정보 열 별로(맨 처음 row였던 자료 분류는 불포함)
    tmp = []
    for i, elem in enumerate(row):#교통사망정보 열만큼 돌림
        if attr_train[i] in attr_test:#교통사망정보 열의 원소 중 test_kor와 일치하는 원소만 append
            tmp.append(elem)

    input = []
    output = []

    for i, elem in enumerate(tmp): #인덱스로 인풋 아웃풋 분류(2<=인풋<=7)
        if i < 2:
            input.append(elem)
        elif i < 7:
            output.append(elem)
        else:
            input.append(elem)

    input_train.append(input)
    output_train.append(output)

f.close()
f2.close()
