import datetime
from datetime import date
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from ytm import ytm, recent_future_coupons, accured_interest
x = Symbol('x')

def dirty_price(today, row):
    clean = row['Close']
    coupon = row['Coupon']
    maturity = date.fromisoformat(row['Maturity'])
    prev = recent_future_coupons(today, maturity)[0]
    dirty = clean + accured_interest(today, coupon, prev)
    return dirty


dates = ['2020-01-15',
    '2020-01-14',
    '2020-01-13',
    '2020-01-10',
    '2020-01-09',
    '2020-01-08',
    '2020-01-07',
    '2020-01-06',
    '2020-01-03',
    '2020-01-02'
         ]
data = pd.read_csv('RevelantBonds.csv')
lines = []
all_ytm = np.zeros((5, 10))
j=1
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []


for d in dates:
    today_str = d
    data_today = data[data['Date']==today_str]
    today = date.fromisoformat(today_str)
    data_today['Dirty Price'] = data_today.apply(
        lambda row: dirty_price(today, row), axis=1)

    bonds = []
    for index, row in data_today.iterrows():
        bonds += [
            ( row['Coupon'], row['Close'], date.fromisoformat(row['Maturity']) )
        ]

    list_mat, list_ytm = ytm(bonds=bonds, today=today)
    #plt.plot(list_mat, list_ytm, label=today_str)
    for i in [1, 2, 3, 4, 5]:
        all_ytm[i-1][j-1] = np.interp(x=i, xp=list_mat, fp=list_ytm)
    j += 1

for j in range(1,10):
    X1 += [math.log(all_ytm[1 - 1][j + 1 - 1] / all_ytm[1 - 1][j - 1])]
    X2 += [math.log(all_ytm[2 - 1][j + 1 - 1] / all_ytm[2 - 1][j - 1])]
    X3 += [math.log(all_ytm[3 - 1][j + 1 - 1] / all_ytm[3 - 1][j - 1])]
    X4 += [math.log(all_ytm[4 - 1][j + 1 - 1] / all_ytm[4 - 1][j - 1])]
    X5 += [math.log(all_ytm[5 - 1][j + 1 - 1] / all_ytm[5 - 1][j - 1])]

cov_matrix_1 = np.cov([X1, X2, X3, X4, X5])
w1, v1 = np.linalg.eig(cov_matrix_1)

# QUESTION 1 PLOT
# plt.legend()
# plt.ylim(1.5, 2.5)
# plt.xlabel('Years')
# plt.ylabel('YTM (%)')
# plt.title('5YR Par Curve for Jan02-Jan15')
# plt.show()
