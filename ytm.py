import datetime
from datetime import date
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
x = Symbol('x')

def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
        29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return date.replace(day=d,month=m, year=y)

def recent_future_coupons(today: date, maturity: date):
    coupon_dates = []
    m_delta = 0

    while True:
        coup_date = monthdelta(maturity, m_delta)
        coupon_dates += [ coup_date ]
        m_delta -= 6
        if coup_date < today:
            break

    coupon_dates = coupon_dates[::-1]
    return coupon_dates

def accured_interest(today: date, coupon: float, recent_coupon: date):
    yrs_since = ((today-recent_coupon).days)/365
    return yrs_since*coupon

def dirty_price_given_discount(future_coupons: list, coupon: float, discount: float, today: date):
    p = 0.0
    for d in future_coupons:
        yrs = ((d - today).days)/365
        disc = (1 + (discount/100) )**yrs
        p += (coupon/2)/disc

    yrs = ((d - today).days) / 365
    disc = (1 + (discount / 100)) ** yrs
    p += (100) / disc

    return p

def ytm(bonds, today):
    list_ytm = []
    list_mat = []

    for b in bonds:
        coupon, clean_price, maturity = b

        coupon_dates = recent_future_coupons(today=today, maturity=maturity)

        recent_coupon = coupon_dates[0]
        future_coupons = coupon_dates[1:]

        acc_interest = accured_interest(today=today, coupon=coupon, recent_coupon=recent_coupon)

        dirty_price = clean_price + acc_interest

        diff = []
        ytm = -5.0
        while True:
            est = dirty_price_given_discount(future_coupons, coupon, discount=ytm, today=today)
            diff += [(abs(dirty_price-est), ytm)]

            ytm += 0.01
            if ytm > 5:
                break
        ytm_final = min(diff, key= lambda t:t[0])

        list_mat += [ (maturity-today).days/365 ]
        list_ytm += [ ytm_final[1] ]

    return (list_mat, list_ytm)


