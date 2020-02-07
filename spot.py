import datetime
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from ytm import ytm, recent_future_coupons, accured_interest
from test import dirty_price
x = Symbol('x')
import math


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
all_f = np.zeros((4, 10))
Y1 = []
Y2 = []
Y3 = []
Y4 = []
j2=1

def time_diff(day1, day2):
    return abs((day1-day2).days / 365)

for d in dates:
    r = []
    today_str = d
    data_today = data[data['Date'] == today_str]
    today = date.fromisoformat(today_str)
    data_today['Dirty Price'] = data_today.apply(
        lambda row: dirty_price(today, row), axis=1)

    bonds = []
    for index, row in data_today.iterrows():
        bonds += [
            [row['Coupon'], row['Close'], date.fromisoformat(row['Maturity']) , row['Dirty Price']]
        ]

    curr_bond = bonds[0]
    r_temp = -math.log(curr_bond[3] / (100)) / time_diff(today, curr_bond[2])
    curr_bond.append(r_temp)

    # curr_bond = bonds[1]
    # lhs = curr_bond[3] - curr_bond[0]*math.exp( -bonds[0][4] * time_diff(today, bonds[0][2]) )
    # r_temp = -math.log10(lhs/(100 + curr_bond[0]/2))/time_diff(today, curr_bond[2])
    # curr_bond.append(r_temp)
    #
    # curr_bond = bonds[2]
    # lhs = curr_bond[3] - curr_bond[0] * math.exp(-bonds[0][4] * time_diff(today, bonds[0][2])) - \
    #       curr_bond[0] * math.exp(-bonds[1][4] * time_diff(today, bonds[1][2]))
    # r_temp = -math.log10(lhs / (100 + curr_bond[0] / 2)) / time_diff(today,curr_bond[2])
    # curr_bond.append(r_temp)

    for i in range(1, 10):
        curr_bond = bonds[i]
        lhs = curr_bond[3]
        for j in range(0, i):
            lhs -= curr_bond[0]*math.exp( -bonds[j][4] * ( time_diff(today, curr_bond[2]) +0.5*(i-j) ) )
        r_temp = -math.log(lhs / (100 + curr_bond[0] / 2)) / time_diff(today,curr_bond[2])
        curr_bond.append(r_temp)

    maturities = [time_diff(bonds[i][2], today) for i in range(0, 10)]
    spot_rates = [bonds[i][4] for i in range(0, 10)]
    y_pts = [np.interp(x=i, xp=maturities, fp=spot_rates) for i in range(1,5)] + [spot_rates[-1]]
    x_pts = [1, 2, 3, 4] + [maturities[-1]]
    #plt.plot(x_pts, y_pts, label=today_str)

    forward_rates = [ (math.log10(spot_rates[2])-math.log10(spot_rates[1])) / (maturities[2]-maturities[1]) ]
    #print(forward_rates)
    for T in range(2,5):
        f_temp = ( (1+ y_pts[T])**T ) / ( (1+ y_pts[0])**1 )
        forward_rates += [f_temp - 1]

    plt.plot([1, 2, 3, 4], forward_rates, label=today_str)

    for i in [1, 2, 3, 4]:
        all_f[i - 1][j2 - 1] = np.interp(x=i, xp=[1, 2, 3, 4], fp=forward_rates)
    j2 += 1


for j in range(1,10):
    Y1 += [math.log(all_f[1 - 1][j + 1 - 1] / all_f[1 - 1][j - 1])]
    Y2 += [math.log(all_f[2 - 1][j + 1 - 1] / all_f[2 - 1][j - 1])]
    Y3 += [math.log(all_f[3 - 1][j + 1 - 1] / all_f[3 - 1][j - 1])]
    Y4 += [math.log(all_f[4 - 1][j + 1 - 1] / all_f[4 - 1][j - 1])]

cov_matrix_2 = np.cov([Y1, Y2, Y3, Y4])
w2, v2 = np.linalg.eig(cov_matrix_2)

plt.legend()
plt.xlabel('Years')
plt.ylabel('1-T Forward Rate (%)')
plt.title('1-4YR Forward Rate Curve for Jan02-Jan15')
#plt.show()


