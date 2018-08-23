from rule_base import main_1
from model_L_ensemble import main_2
from model_K_ensemble import main_3
from model_M_ensemble import main_4
from model_N_ensemble import main_5
from model_O_ensemble import main_6
from model_P_ensemble import main_7
from model_B_ensemble import main_8
from model_H_and_I_ensemble import main_9
from model_N2_ensemble import main_10
from model_O2_ensemble import main_11
from model_A_ensemble import main_12
from model_I_ensemble import main_13
from model_regression_ensemble import main_14
from rule_base_2 import main_15
from result import final
import shutil
import os


if __name__ == "__main__":
    if os.path.exists('./test.csv'):
        shutil.copy('./test.csv', './test_kor.csv')

    if os.path.exists('./result.csv'):
        shutil.copy('./result.csv', './result_kor.csv')

    main_1()
    main_2()
    main_3()
    main_4()
    main_5()
    main_6()
    main_7()
    main_8()
    main_9()
    main_10()
    main_11()
    main_12()
    main_13()
    main_14()
    main_15()
    final()

    if os.path.exists('./test.csv'):
        os.remove('./test.csv')
        shutil.copy('./test_kor.csv', './test.csv')
        os.remove('./test_kor.csv')

    if os.path.exists('./result.csv'):
        os.remove('./result.csv')
        shutil.copy('./result_kor.csv', './result.csv')
        os.remove('./result_kor.csv')
