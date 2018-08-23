import csv
from pprint import pprint


def final():
    f1 = open('./outputs/15_output.csv', 'r')
    f2 = open('./result_kor.csv', 'r')
    r1 = csv.reader(f1)
    r2 = csv.reader(f2)

    res = []
    for row in r2:
        res.append(row)

    """
    좌표화
    """
    dicts = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5,\
             'F':6, 'G':7, 'H':8, 'I':9, 'J':10,\
             'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16}

    cord = []
    for i, row in enumerate(res):
        if i == 0:
            continue

        r = int(row[0]) - 1
        c = dicts[row[1]] - 1

        cord.append([r, c])

    """
    행렬화
    """
    table = []
    for row in r1:
        table.append(row)

    f1.close()
    f2.close()

    """
    calculation
    """
    cnt = 1
    for rc in cord:
        res[cnt][2] = table[rc[0]][rc[1]]
        cnt += 1

    """
    write
    """
    f3 = open('./result_kor.csv', 'w', newline='')
    r3 = csv.writer(f3)

    for row in res:
        r3.writerow(row)

    f3.close()


if __name__ == "__main__":
    final()
