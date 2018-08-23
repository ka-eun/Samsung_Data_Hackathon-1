import csv
import os


def main_15():
    if not os.path.isdir('./outputs'):
        os.mkdir('./outputs')

    f1 = open('./outputs/14_output.csv', 'r')
    f2 = open('./outputs/15_output.csv', 'w', newline='')
    r1 = csv.reader(f1)
    r2 = csv.writer(f2)

    for i, row in enumerate(r1):
        if i > 0:
            for j, elem in enumerate(row):
                if (j == 3) & (elem == ''):
                    row[j] = str(int(row[2]) + int(row[4]) + int(row[5]) + int(row[6]))

        r2.writerow(row)

    f1.close()
    f2.close()


if __name__ == "__main__":
    main_15()
