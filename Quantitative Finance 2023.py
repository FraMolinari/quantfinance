import numpy as np
import pandas as pd
from datetime import date
import numpy_financial as npf
import matplotlib.pyplot as plt
from datetime import datetime as dt
import math as m

pd.set_option('display.max_columns', None)


def simple_interest(principal, rate, years):
    amount = principal * (1+((rate/100)*years))
    round(amount,2)
    return(print(f"\n The simple interest ending balance is {round(amount, 2)}\n"))

def compound_interest(principal, rate, year, time):
    amount = principal * pow((1+rate/100), year*time)
    round(amount,2)
    return print(f"\nThe compound interest ending balance is: {round(amount, 2)}\n")

def net_pv(cfList, rate):
    f = 0
    i = 0
    pv = 0
    while f< len(cfList):
        pv += (cfList[f]/((1+rate/100)**i))
        f += 1
        i += 1
    round(pv,2)
    print(f"\nThe NPV is: {pv:.2f}")
    return pv 
    

def future_value(cfList, rate):
    f = 0
    i = (len(cfList)-1)
    pv = 0
    while f< len(cfList):
        pv += (cfList[f]*((1+rate/100)**i))
        f += 1
        i -= 1
    round(pv,2)
    return print(f"\nThe Future Value is: {pv}")

def regular_perpetuity(dividend, discount_rate):
    present_value = dividend/(discount_rate/100)
    round(present_value,2)
    return print(f"\nThe present value of the annuity is: {present_value:.2f}")

def finite_perpetuity(dividend, discount_rate, years):
    present_value = (dividend/(discount_rate/100))*(1- (1/((1+discount_rate/100)**years)))
    round(present_value,2)
    return print(f"\nThe present value of the annuity is: {present_value:.2f}")

def anticipated_perpetuity(dividend, discount_rate, years):
    present_value = ((dividend/(discount_rate/100))*(1- (1/((1+discount_rate/100)**years))))*(1 + discount_rate/100)
    round(present_value,2)
    return print(f"\nThe present value of the annuity is: {present_value:.2f}")

def delayed_perpetuity(dividend, discount_rate, years, delay):
    present_value = ((dividend/(discount_rate/100))*(1- (1/((1+discount_rate/100)**years))))*(1/((1+discount_rate/100)**delay))
    round(present_value,2)
    return print(f"\nThe present value of the annuity is: {present_value:.2f}")

def loan_calculator(principal, discount_rate, compounding, years):
    amount_paid = (principal*((discount_rate/100)/compounding))/(1-(1/((1+(discount_rate/100)/compounding)**(years*compounding))))
    round(amount_paid, 2)
    print(f"\nThe loan payment per period is: {amount_paid:.2f}")
    return amount_paid

def loan_fees_calculator(principal, payment, nominal_ir, payment_frequency, years):
    fees =  ( payment/ ((nominal_ir/100)/payment_frequency) )   * ( 1- (1 / ((1 + ((nominal_ir/100)/payment_frequency ) )** (years*payment_frequency))))-principal
    round(fees,2)
    print(f"\nThe fees and expenses are: {fees:.2f}")
    return fees

def bond_amortization(principal, interest_rate, years, payments_year):
    start_date = pd.Timestamp.today().date()

    if payments_year == 12:
       frequency = "M"
    if payments_year == 1:
        frequency = "Y"
    if payments_year == 2:
        frequency = "6M"
    if payments_year == 3:
        frequency = "4M"
    if payments_year == 4:
        frequency = "3M"
    if payments_year == 6:
        frequency = "2M"
    
    pmt = npf.pmt(interest_rate/payments_year, years*payments_year, principal)

    rng = pd.date_range(start_date, periods=years * payments_year, freq = frequency)
    rng.name = "Date"

    df = pd.DataFrame(index=rng,columns=['Previous_Balance', 'Payment', 'Interest', 'Principal', 'Balance'], dtype='float')
    df.reset_index(inplace=True)
    df.index += 1
    df.index.name = "Period"
    principal_repayment = np.round(-npf.ppmt(interest_rate/payments_year, df.index, years*payments_year, principal),2)

    df["Previous_Balance"]
    df["Payment"] = -round(npf.pmt(interest_rate/payments_year, years*payments_year, principal),2)
    df["Principal"] = np.round(-npf.ppmt(interest_rate/payments_year, df.index, years*payments_year, principal),2)
    df["Interest"] = np.round(-npf.ipmt(interest_rate/payments_year, df.index, years*payments_year, principal),2)
    df = df.round(2)


    df["Balance"] = 0
    df.loc[1, "Balance"] = principal - df.loc[1, "Principal"]

    for i in range(2, len(df)+1):
        prev_balance = df.loc[i-1, 'Balance']
        principal_1 = df.loc[i, 'Principal']
        if prev_balance == 0:
            df.loc[i, ['Previous_Balance', 'Payment', 'Interest', 'Principal', 'Balance']] = 0 
            continue
        if abs(principal_1) <= prev_balance:
            df.loc[i, 'Balance'] = prev_balance - principal_1
        else:
            if prev_balance <= abs(principal_1):
                principa_1 = -prev_balance
            else:
                df.loc[i, 'Balance'] = 0
                df.loc[i, 'Principal'] = principal_1
                df.loc[i, "Payment"] = principal_1 -  df.loc[i, "Interest"]

    df["Previous_Balance"] 
    df.loc[1, "Previous_Balance"] = principal

    for i in range(2, len(df)+1):
        prev_balance_1 = df.loc[i-1, 'Previous_Balance']
        balance_1 = df.loc[i, 'Balance']
        principal_1 = df.loc[i, 'Principal']
        if prev_balance_1 == 0:
            df.loc[i, ['Previous_Balance', 'Payment', 'Interest', 'Principal', 'Balance']] = 0 
            continue
        if abs(balance_1) <= prev_balance_1:
            df.loc[i, 'Previous_Balance'] = balance_1 + principal_1
        else:
            print("error")
    
    return print(df)

def npv (initial_investment, r, cashflows):
    pv_cashflows = 0
    for i, cashflow in enumerate(cashflows):
        pv_cashflows += cashflow / (1+r)**(i+1)
    npv_calc = pv_cashflows -initial_investment
    return npv_calc

def calculate_irr(cashflows, compounding_frequency):
    initial_guess = 0

    max_iterations = 1000
    tolerance = 0.0001

    irr_approximation = initial_guess

    for _ in range(max_iterations):
        npv = 0.0
        npv_derivative = 0.0

        for t, cashflow in enumerate(cashflows):
            npv += cashflow / (1 + irr_approximation / compounding_frequency) ** (t * compounding_frequency)
            npv_derivative -= t * cashflow / (1 + irr_approximation / compounding_frequency) ** ((t + 1) * compounding_frequency)

        irr_approximation -= npv / npv_derivative

        if abs(npv) < tolerance:
            return irr_approximation

    return irr_approximation


def accrued_interest(purchase_date, next_payment_date, previous_payment_date, coupon_rate, compounding_frequency):
    date0 = dt.strptime(purchase_date, "%Y/%m/%d")
    date1 = dt.strptime(next_payment_date, "%Y/%m/%d")
    date2 = dt.strptime(previous_payment_date, "%Y/%m/%d")
    if date0 < date1 and date0 > date2:
        delta1 = (date0-date2).days
        delta2 = (date1-date0).days
        acc_int = (delta1/(delta1+delta2))*(coupon_rate/compounding_frequency)
    else:
        print("Error: check chronological order of dates")
    return print(f"The accrued interest is: {acc_int:.2f}% of the principal")
    


def calculate_yield(par_value, coupon_rate, years_to_maturity, current_price):
    annual_coupon_payment = par_value * coupon_rate
    periods = years_to_maturity
    cash_flows = np.full(periods, annual_coupon_payment)
    cash_flows[-1] += par_value
    
    bondyield = npf.rate(nper=periods, pmt=annual_coupon_payment, pv=-current_price, fv=par_value, guess=0.05)
    print("The yield to maturity of the bond is: {:.2f}%".format(bondyield*100))

    coupon_payment = coupon_rate * par_value
    coupon_payments = np.full(years_to_maturity, coupon_payment)
    coupon_payments[-1] += par_value
    yields = np.linspace(0, 10, 101)

    prices = []
    for y in yields:
        price = npf.npv(rate=y/100, values=coupon_payments)
        prices.append(price)

    plt.plot(yields, prices)
    plt.title('Bond Price-Yield Curve')
    plt.xlabel('Yield to Maturity')
    plt.ylabel('Price')
    plt.ylim() 
    final = plt.show()

    return bondyield, final



def bond_price(ttm, n_payments, nominal_value, yield_rate, coupon_rate, yield_compounding):
    yield_rate_decimal = yield_rate/100
    coupon_rate_decimal = coupon_rate/100

    if n_payments != yield_compounding:
        periods = ttm*n_payments
        coupons = (coupon_rate_decimal / n_payments) * nominal_value
        per_period_yield = yield_rate_decimal / yield_compounding
        present_value = 0

        for i in range(periods):
            present_value += coupons / ((1 + per_period_yield) ** ((i + 1)*yield_compounding/n_payments))
        present_value += nominal_value / ((1 + per_period_yield) ** (periods*yield_compounding/n_payments))

    else: 
        periods = ttm*n_payments
        coupons = (coupon_rate_decimal / n_payments) * nominal_value
        per_period_yield = yield_rate_decimal / yield_compounding
        present_value = 0

        for i in range(periods):
            present_value += coupons / ((1 + per_period_yield) ** ((i + 1)))
        present_value += nominal_value / ((1 + per_period_yield) ** (periods))
    
    return present_value

def duration(ttm, n_payments, nominal_value, yield_rate, coupon_rate, yield_compounding):
    yield_rate_decimal = yield_rate/100
    coupon_rate_decimal = coupon_rate/100

    if n_payments != yield_compounding:
        periods = ttm*n_payments
        coupons = (coupon_rate_decimal / n_payments) * nominal_value
        per_period_yield = yield_rate_decimal / yield_compounding
        present_value = 0

        for i in range(periods):
            present_value += (coupons / ((1 + per_period_yield) ** ((i + 1)*yield_compounding/n_payments))) * (((i + 1)*yield_compounding/n_payments)/yield_compounding)
        present_value += (nominal_value / ((1 + per_period_yield) ** (periods*yield_compounding/n_payments))) * ((periods*yield_compounding/n_payments)/yield_compounding)

    else: 
        periods = ttm*n_payments
        coupons = (coupon_rate_decimal / n_payments) * nominal_value
        per_period_yield = yield_rate_decimal / yield_compounding
        present_value = 0

        for i in range(periods):
            present_value += (coupons / ((1 + per_period_yield) ** ((i + 1)))) * ((i+1)/yield_compounding)
        present_value += (nominal_value / ((1 + per_period_yield) ** (periods))) * ((periods)/yield_compounding)
    
    bond_price_1 = bond_price(ttm, n_payments ,nominal_value, yield_rate, coupon_rate, yield_compounding)
    duration = present_value / bond_price_1

    return duration

def modified_duration(duration1, y, compounding):
    ycent = y/100
    modified = duration1 * 1/((1 + (ycent / compounding)))
    return modified

def dollar_duration_easy(modified_duration, price):
    dollar = price*modified_duration
    return dollar

def price_sensitivity(time_maturity, n_payments ,nominal_value, yield_rate, coupon_rate, compounding_period, change):
    duration_s = duration(time_maturity, n_payments,nominal_value, yield_rate, coupon_rate, compounding_period)
    m_duration = modified_duration(duration_s, yield_rate, compounding_period)
    d_price = bond_price(time_maturity, n_payments,nominal_value, yield_rate, coupon_rate, compounding_period)
    d_duration = dollar_duration_easy(m_duration, d_price)
    dollar_delta = -d_duration*(change/100)
    dollar_duration_final = nominal_value - (-dollar_delta)
    dollar_duration_final2 = nominal_value - (dollar_delta)
    return print("Price:" + str(dollar_duration_final) )

def solve_portfolio_weights(d1, d2, dl):
    w1 = (d2-dl) / (d2-d1)
    w2 = (dl - d1) / (d2-d1)
    x = [w1,w2]
    return x

def liability_pv(cashflows, yield_rate, compounding_frequency, maturity):
    pv = 0
    n = len(cashflows)
    for i in range(n):
        cash = cashflows[i]
        pv += cash / (((1 + (yield_rate / compounding_frequency)) ** ((i + 1)*(compounding_frequency  / maturity))))
    return pv

def liability_duration(cashflows, yield_rate, compounding_frequency, maturity):
    du = 0
    pv = liability_pv(cashflows, yield_rate, compounding_frequency, maturity)
    n = len(cashflows)
    for i in range(n):
        cash = cashflows[i]
        du += (cash / ((1 + (yield_rate / compounding_frequency)) ** ((i + 1)*(compounding_frequency  / maturity)))) * ((i + 1)*(compounding_frequency  / maturity)) / compounding_frequency
    duu = du / pv
    duuu = duu / (1 + yield_rate / compounding_frequency)
    return duuu


def getinputs():
    n = 2
    y = int(input("Enter the yield rate: "))

    bonds = []
    for i in range(n):
        coupon_rate = float(input(f"Enter the yearly coupon rate of bond {i+1} (enter 0 if zero coupon bond): "))
        if coupon_rate == 0: 
            maturity = int(input(f"Enter the time to maturity of the bond (IN MONTHS) {i+1}: "))
            principal = float(input(f"Enter the principal of bond {i+1}: "))
            compounding = int(input(f"Enter the compounding frequency of bond {i+1}: "))
            price_v = principal/((1+((y/100)/compounding))**(maturity/12))
            duration_i = maturity/12
            dur = modified_duration(duration_i, y ,compounding)
        else:
            maturity = int(input(f"Enter the time to maturity (IN YEARS) of bond {i+1}: "))
            npayments = int(input(f"Enter the coupon payment frequency (per year) of bond {i+1}: "))
            principal = float(input(f"Enter the principal of bond {i+1}: "))
            compounding = int(input(f"Enter the compounding frequency of bond {i+1}: "))
            price_v = bond_price(maturity, npayments, principal, y, coupon_rate, compounding)
            duration_i = duration(maturity, npayments,principal, y, coupon_rate, compounding)
            dur = modified_duration(duration_i, y ,compounding)
        bonds.append((maturity, coupon_rate, principal, price_v, dur))

    cashflows = []
    maturity = int(input("Enter number of cashflows of the liability: "))
    for i in range(maturity):
        cashflow = float(input(f"Enter cashflow {i+1} of the liability: "))
        cashflows.append(cashflow)
    yy = y/100
    cf_liability = int(input("Enter the compounding frequency of the liabity: "))
    liability_price = liability_pv(cashflows, yy, cf_liability, maturity)
    dur_liability = liability_duration(cashflows, yy, cf_liability, maturity)
    bond_1 = bonds[0]
    bond_2 = bonds[1]
    d1 = bond_1[4]
    d2 = bond_2[4]
    weights = solve_portfolio_weights(d1, d2, dur_liability)

    units = []
    for i in range(len(bonds)):
        bond = bonds[i]
        weight = weights[i]
        bond_price_v = bond[3]
        unit = weight * liability_price / bond_price_v
        units.append(unit)

    u1 = round(units[0],4)
    u2 = round(units[1],4)
    w1 = round(weights[0],4)
    w2 = round(weights[1],4)

    print("U1 = " + str(u1)+ " U2 = " +str(u2) + " w1 = " +str(w1) +" w2 = " +str(w2))
    return units, weights, bonds

def convexity(time_maturity, n_payments, nominal_value, yield_rate, coupon_rate, compounding_frequency):
    yield_rate_decimal = yield_rate/100
    coupon_rate_decimal = coupon_rate/100
    price_bond_convexity = bond_price(time_maturity, n_payments,nominal_value, yield_rate, coupon_rate, compounding_frequency)
    duration_bond_convexity = duration(time_maturity, n_payments, nominal_value, yield_rate, coupon_rate, compounding_frequency)
    modified_duration_convexity = modified_duration(duration_bond_convexity, yield_rate, compounding_frequency)
    dollar_duration_convexity = dollar_duration_easy(modified_duration_convexity, price_bond_convexity) 
    periods = time_maturity*compounding_frequency
    coupons = coupon_rate_decimal*nominal_value/compounding_frequency
    per_period_yield = yield_rate_decimal/compounding_frequency
    present_value = 0
    for i in range(1, periods+1):
        present_value += (coupons / ((1+per_period_yield) ** i )) * ( (i * (i+1) / (compounding_frequency ** 2) ) )
    present_value += (nominal_value / ((1+per_period_yield) **i )) * ( (i * (i+1) / (compounding_frequency ** 2) ) )
    convexity_finale = present_value * (1 / price_bond_convexity) * (1 / ((1 + yield_rate_decimal / compounding_frequency) ** 2 ))
    convexity_dollar = convexity_finale * price_bond_convexity
    yield_increase = price_bond_convexity - dollar_duration_convexity * (0.01) + (1/2) * convexity_dollar * (0.01 ** 2)
    return convexity_finale, convexity_dollar, yield_increase
    



def bond_price_spot_rates(time_maturity, nominal_value, coupon_rate, compounding_frequency):
    periods = time_maturity*compounding_frequency
    coupon_rate_decimal = coupon_rate / 100
    coupons = coupon_rate_decimal * nominal_value / compounding_frequency
    present_value = 0
    yield_rates = []

    for i in range(periods):
        yield_rate = float(input(f"Enter yield for period {i+1}: "))
        yield_rates.append(yield_rate / 100)
        present_value += coupons / ((1 + yield_rates[i] / compounding_frequency) ** (i + 1))

    present_value += nominal_value / ((1 + yield_rates[-1] / compounding_frequency) ** periods)
    return present_value

def bond_price_spot_rates_2(time_maturity, nominal_value, coupon_rate):
    coupon_rate_decimal = coupon_rate / 100
    coupons = coupon_rate_decimal * nominal_value
    present_value = 0
    yield_rates = []

    for i in range(time_maturity):
        yield_rate = float(input(f"Enter yield for period {i+1}: "))
        yield_rates.append(yield_rate / 100)
        present_value += coupons * (2.71828 ** (-yield_rates[i] * (i+1)))
    present_value += nominal_value * (2.71828 ** (-yield_rates[-1] * time_maturity))
    
    return present_value


def forward_rate(spot1, spot2, year1, year2, compounding_period):
    spot1_p = spot1 / 100
    spot2_p = spot2 / 100

    if compounding_period == "inf":
        forward = (spot2_p * year2 - spot1_p * year1) / (year2 - year1)
    else:
        forward = compounding_period * ((((1 + (spot2_p / compounding_period)) ** (compounding_period * year2)) /
                                        ((1 + (spot1_p / compounding_period)) ** (compounding_period * year1))) ** (
                          1 / (compounding_period * (year2 - year1))) - 1)
    return forward

def bond_duration_spot_rates(time_maturity, nominal_value, coupon_rate, compounding_frequency):
    periods = time_maturity*compounding_frequency
    coupon_rate_decimal = coupon_rate / 100
    coupons = coupon_rate_decimal * nominal_value / compounding_frequency
    present_value = 0
    duration = 0
    yield_rates = []

    for i in range(periods):
        yield_rate = float(input(f"Enter yield for period {i+1}: "))
        yield_rates.append(yield_rate / 100)
        present_value += coupons / ((1 + yield_rates[i] / compounding_frequency) ** (i + 1))
    present_value += nominal_value / ((1 + yield_rates[-1] / compounding_frequency) ** periods)
    
    for i in range(periods):
        duration += (coupons / ((1 + yield_rates[i] / compounding_frequency) ** (i + 2)))*(((i+1) / compounding_frequency))
    duration += nominal_value / ((1 + yield_rates[-1] / compounding_frequency) ** (periods+1))*((periods/compounding_frequency))

    duration_finale = duration / present_value
    return duration_finale

def bobby_fisher(time_maturity, nominal_value, coupon_rate, compounding_frequency):
    coupon_rate_decimal = coupon_rate / 100
    coupons = coupon_rate_decimal * nominal_value / compounding_frequency
    present_value = 0
    fisher = 0
    periods = time_maturity*compounding_frequency
    yield_rates = []

    for i in range(periods):
        yield_rate = float(input("Yield" + str(i+1) + ": "))
        yield_rates.append(yield_rate / 100)
        present_value += coupons * (2.71828 ** (-yield_rates[i] * ((i+1)/compounding_frequency)))
    present_value += nominal_value * (2.71828 ** (-yield_rates[-1] * (periods/compounding_frequency)))

    for i in range(periods):
        fisher += (coupons * (2.71828 ** (-yield_rates[i] * (((i+1)/compounding_frequency)))))*((i+1)/compounding_frequency)
    fisher += (nominal_value * (2.71828 ** (-yield_rates[-1] * (periods/compounding_frequency))))*(periods/compounding_frequency)

    bobby_fisher_gm = fisher / present_value
    
    return bobby_fisher_gm

def asset_returns(asset_price_0, asset_price_1):
    gross_return = asset_price_1/asset_price_0
    arithmetic_return = (asset_price_1 - asset_price_0) / asset_price_0
    return(gross_return,arithmetic_return )

def short_selling(short_position, long_position):
    a_return_short = asset_returns(short_position, long_position)
    a_return_short_m = a_return_short[1]
    final_cashflow = -short_position*(1+a_return_short_m )
    profit_ss = short_position - final_cashflow
    return final_cashflow, profit_ss


def stock_r_v_c_c (price1, price2, periods):

    future_price1_list = []
    future_price2_list = []
    probabilities = []
    expectations1 =[]
    expectations2 = []


    for i in range(periods):
        future_price1 = float(input(f"Enter S{i+1}W{i+1} of the first stock: "))
        future_price1_list.append(future_price1)

    for i in range(periods):
        future_price2 = float(input(f"Enter S{i+1}W{i+1} of the second stock: "))
        future_price2_list.append(future_price2)

    for i in range(periods):
        probability = float(input(f"Enter P(W{i+1}), with P<1: "))
        probabilities.append(probability)

    for i in range(periods):
        expectation1 = future_price1_list[i] /price1 -1
        expectations1.append(expectation1)

    for i in range(periods):
        expectation2 = future_price2_list[i] /price2 -1
        expectations2.append(expectation2)

    expected_return1 = sum(expectations1[i] * probabilities[i] for i in range(periods))
    expected_return2 = sum(expectations2[i] * probabilities[i] for i in range(periods))

    variance1 = sum(((expectations1[i] ** 2) * probabilities[i]) for i in range(periods)) - (expected_return1 ** 2)
    variance2 = sum(((expectations2[i] ** 2) * probabilities[i]) for i in range(periods)) - (expected_return2 ** 2)

    covariance = sum((expectations1[i]* expectations2[i] * probabilities[i]) for i in range(periods)) - (expected_return1 * expected_return2)

    correlation = covariance / (variance1**(1/2) * variance2**(1/2))

    return expected_return1, expected_return2, variance1, variance2, covariance, correlation


def portfolio_r_v (price1, price2, periods, w1, w2):
    single_r_v = stock_r_v_c_c(price1, price2, periods)
    expected_return_pf = single_r_v[0]*w1 + single_r_v[1]*w2
    variance_pf = (w1**2) * (single_r_v[2]) + (w2**2) * (single_r_v[3]) + (2*w1*w2*single_r_v[4])
    return expected_return_pf, variance_pf


def minim_variance_2(var_1,var_2,cov):
    var_1_a = var_1*2
    var_2_b = var_2*2
    cov_ab = cov*2

    A = np.array([[var_1_a, cov_ab, -1],
              [cov_ab, var_2_b, -1],
              [1, 1, 0]])

    b = np.array([0, 0, 1])
    x = np.linalg.solve(A, b)
    a = x[0]
    b = x[1]
    c = x[2]
    return a,b,c


def minim_variance_3(var_1, var_2, var_3, cov_12, cov_13, cov_23):
    var_1_a = var_1 * 2
    var_2_b = var_2 * 2
    var_3_c = var_3 * 2
    cov_ab = cov_12 * 2
    cov_ac = cov_13 * 2
    cov_bc = cov_23 * 2

    A = np.array([[var_1_a, cov_ab, cov_ac, -1],
                  [cov_ab, var_2_b, cov_bc, -1],
                  [cov_ac, cov_bc, var_3_c, -1],
                  [1, 1, 1, 0]])

    b = np.array([0, 0, 0, 1])
    x = np.linalg.solve(A, b)
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    return a, b, c, d


def minimize_variance_eff_4(var_1, var_2,cov_12, r_1 , r_2 ,e):
    
    var_1_a = var_1 * 2
    var_2_b = var_2 * 2
    cov_ab = cov_12 * 2

    A = np.array([[var_1_a, cov_ab, -r_1, -1],
                  [cov_ab, var_2_b, -r_2, -1],
                  [r_1, r_2, 0, 0],
                  [1, 1, 0, 0]])

    b = np.array([0, 0, e, 1])
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
  
    return a, b, c, d

def minimize_variance_eff_5(var_1, var_2, var_3, cov_12, cov_13, cov_23, r_1 , r_2,r_3 ,e):
    
    var_1_a = var_1 * 2
    var_2_b = var_2 * 2
    var_3_c = var_3 * 2
    cov_ab = cov_12 * 2
    cov_ac = cov_13 * 2
    cov_bc = cov_23 * 2

    A = np.array([[var_1_a, cov_ab, cov_ac, -r_1, -1],
                  [cov_ab, var_2_b, cov_bc, -r_2, -1],
                  [cov_ac, cov_bc, var_3_c, -r_3,- 1],
                  [r_1, r_2, r_3, 0, 0],
                  [1, 1, 1, 0, 0]])

    b = np.array([0, 0, 0, e, 1])
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    f = x[4]
    return a, b, c, d, f




def menu ():
    while True:
        print("\n1: Basic theory of interest\n2: Fixed income securities\n3: Term structure of interest rates\n4: Foundamentals of mean-variance portfolio theory\n")
        choice_1 = input()
        
        if choice_1 == "1":

            print("\n1: Simple Interest\n2: Compound Interest\n3: Present Value\n4: Future Value\n5: Annuities\n6: Loans and mortgages\n7: Internal rate of return\n")
            choice_2 = input()

            if choice_2 == "1":
                try:
                    s_principal = float(input("\nInsert the Princial: "))
                    s_rate = float(input("Enter the interest rate: "))
                    s_years = float(input("Enter the number of years: "))
                    simple_interest(s_principal, s_rate, s_years)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_2 == "2":
                try:
                    c_principal = float(input("\nInsert the Princial: "))
                    c_rate = float(input("Enter the interest rate: "))
                    c_years = int(input("Enter the number of years: "))
                    c_time = int(input("Enter compound frequency (per year): "))
                    compound_interest(c_principal, c_rate, c_years, c_time)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_2 == "3":
                try:
                    input_string_npv = input("\nEnter Cashflows separated by space: ")
                    list_splitted_npv = input_string_npv.split()
                    for i in range(len(list_splitted_npv)):
                        list_splitted_npv[i] = float(list_splitted_npv[i])
                    interest_rate_npv = float((input("Enter Interest rate :")))
                    net_pv(list_splitted_npv , interest_rate_npv)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_2 == "4":
                try:
                    input_string_fv = input("\nEnter Cashflows separated by space (including initial investemtn with - sign): ")
                    list_splitted_fv = input_string_fv.split()
                    for i in range(len(list_splitted_fv)):
                        list_splitted_fv[i] = float(list_splitted_fv[i])
                    interest_rate_fv = float((input("Enter Interest rate :")))
                    net_pv(list_splitted_fv , interest_rate_fv)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_2 == "5":
                print("\n1: Regular Perpetuity\n2: N-period annuity\n3: Anticipated payments annuity\n4: Delayed annuity\n")
                choice_3 = input()
                if choice_3 == "1":
                    try:
                        dividend_rp = float(input("Enter the dividend of the perpetuity: "))
                        discount_rate_rp = float(input("Enter the discount rate of the perpetuity: "))
                        regular_perpetuity(dividend_rp, discount_rate_rp)
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")
                if choice_3 == "2":
                    try:
                        dividend_fp = float(input("Enter the dividend of the perpetuity: "))
                        discount_rate_fp = float(input("Enter the discount rate of the perpetuity: "))
                        years_fp = float(input("Enter the number of years of the perpetuity: "))
                        finite_perpetuity(dividend_fp,  discount_rate_fp, years_fp)
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")
                if choice_3 == "3":
                    try:
                        dividend_ap = float(input("Enter the dividend of the perpetuity: "))
                        discount_rate_ap = float(input("Enter the discount rate of the perpetuity: "))
                        years_ap = float(input("Enter the number of payments (n): "))
                        finite_perpetuity(dividend_ap,  discount_rate_ap, years_ap)
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")
                if choice_3 == "4":
                    try:
                        dividend_dp = float(input("Enter the dividend of the perpetuity: "))
                        discount_rate_dp = float(input("Enter the discount rate of the perpetuity: "))
                        years_dp = float(input("Enter the number of payments (n): "))
                        delay_dp = float(input("Enter the delayed periods: "))
                        delayed_perpetuity(dividend_dp,  discount_rate_dp, years_dp, delay_dp)
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_2 == "6":
                print("\n1: Loan calculation (monthly payments)\n2: Annual percentage rate\n3: Running amortization\n")
                choice_4 = input()
                if choice_4 == "1":
                    try:
                        principal_lc = float(input("Enter principal of the loan: "))
                        discount_rate_lc = float(input("Enter the annual interest rate: "))
                        counpound_lc = float(input("Enter the compounding frequency (where 1 means once a year):"))
                        years_lc = float(input("Enter the lifespan of the loan: "))
                        loan_calculator(principal_lc,  discount_rate_lc, counpound_lc, years_lc)
                    except ValueError:
                            print("Error: Invalid input. Please enter a valid number.")

                if choice_4 == "2":
                    try:
                        principal_apr = float(input("Enter principal of the loan: "))
                        nominal_rate_apr = float(input("Enter the nominal interest rate: "))
                        annual_rate_apr = float(input("Enter the annual percentage rate: "))
                        payment_frequency_apr = float(input("Enter the payment frequency per year: "))
                        loan_duration_apr = float(input("Enter the lifespan of the loan: "))
                        sentence_apr = None
                        sentence_apr = loan_calculator(principal_apr, annual_rate_apr, payment_frequency_apr, loan_duration_apr)        
                        loan_fees_calculator(principal_apr, sentence_apr, nominal_rate_apr, payment_frequency_apr, loan_duration_apr)
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

                if choice_4 == "3":
                    try:
                        principal_ra = float(input("Enter principal of the loan: "))
                        nominal_rate_ra = float(input("Enter the nominal interest rate: "))
                        nominal_rate_ra2 = nominal_rate_ra/100
                        payment_frequency_ar = float(input("Enter the payment frequency per year (1,2,3,4,6,12): "))
                        loan_duration_ar = float(input("Enter the lifespan of the loan: "))
                        bond_amortization(principal_ra,nominal_rate_ra2,loan_duration_ar,payment_frequency_ar)
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")


            if choice_2 == "7":
                try:
                    cashflows_irr = input("Enter cashflows separated by space: ")
                    list_splitted_irr = cashflows_irr.split()
                    for i in range(len(list_splitted_irr)):
                        list_splitted_irr[i] = float(list_splitted_irr[i])
                    compounding_frequency_irr = int(input("\nEnter the compounding frequency: "))
                    irr_final = calculate_irr( list_splitted_irr, compounding_frequency_irr)
                    print(f"The IRR is: {irr_final:.3f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

        elif choice_1 == "2":
            print("\n1: Accrued Interest\n2: Bond Yield\n3: Bond Price\n4: Duration\n5: Price Sensitivity\n6: Immunization\n7: Convexity\n")
            choice_5 = input()
            if choice_5 == "1":
                try:
                    date1 = input("Enter the purchase date (yyyy/mm/dd)")
                    date2 = input("Enter the previous payment date (yyyy/mm/dd)")
                    date3 = input("Enter the first future payment date (yyyy/mm/dd)")
                    cp_rate = float(input("Enter the coupon rate: "))
                    comp_freq = float(input("Enter the compounding frequency: "))
                    
                    accrued_interest(date1, date2, date3, cp_rate, comp_freq)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            if choice_5 == "2": 
                try:
                    par_value = float(input("Enter principal of the bond: "))
                    coupon_rate = float(input("Enter the nominal interest rate: "))
                    coupon_rate1 = coupon_rate/100
                    years_to_maturity = int(input("Enter the years to maturity: "))
                    current_price = float(input("Enter the current price: "))
                    loan_duration_apr = int(input("Enter the lifespan of the bond: "))
                    bond_yield = calculate_yield(par_value, coupon_rate1, years_to_maturity, current_price)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            if choice_5 == "3": 
                try:
                    time_to_maturity_p = int(input("Enter the time to maturity: "))
                    number_payments_bp = int(input("Enter the number of coupon payments each year: "))
                    nominal_value_p = float(input("Enter the nominal value: "))
                    yield_rate_p = float(input("Enter the yield rate: "))
                    coupon_rate_p = float(input("Enter the coupon rate: "))
                    compounding_frequency_p = int(input("Enter the compounding frequency: "))
                    price_calc = bond_price(time_to_maturity_p, number_payments_bp, nominal_value_p, yield_rate_p, coupon_rate_p, compounding_frequency_p)
                    print(f"The price of the bond is {price_calc:.2f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            if choice_5 == "4":
                try:
                    time_to_maturity_nd = int(input("Enter the time to maturity: "))
                    number_payments_du = int(input("Enter the number of coupon payments per year: "))
                    nominal_value_nd = float(input("Enter the nominal value: "))
                    yield_rate_nd = float(input("Enter the yield rate: "))
                    coupon_rate_nd = float(input("Enter the annual coupon rate: "))
                    compounding_frequency_nd = int(input("Enter the compounding frequency: "))
                    duration_calc = duration(time_to_maturity_nd, number_payments_du,nominal_value_nd, yield_rate_nd, coupon_rate_nd, compounding_frequency_nd)
                    duration_m = modified_duration(duration_calc, yield_rate_nd, compounding_frequency_nd)
                    price_m = bond_price(time_to_maturity_nd, number_payments_du, nominal_value_nd, yield_rate_nd, coupon_rate_nd, compounding_frequency_nd)
                    duration_d = dollar_duration_easy(duration_m, price_m)
                    print(f"\nThe duration of the bond is {duration_calc:.2f};\nThe modified duration of the bond is {duration_m:.2f};\nThe dollar duration of the bond is {duration_d:.2f}.")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")


            if choice_5 == "5":
                try:
                    time_to_maturity_e = int(input("Enter the time to maturity: "))
                    number_payments_dd = int(input("Enter the number of coupon payments per year: "))
                    nominal_value_e = float(input("Enter the nominal value: "))
                    yield_rate_e = float(input("Enter the yield rate: "))
                    coupon_rate_e = float(input("Enter the annual coupon rate: "))
                    compounding_frequency_e = int(input("Enter the compounding frequency: "))
                    change_e = float(input("Enter the percentage increase/decrease in the yield (without sign): "))
                    duration_calc = price_sensitivity(time_to_maturity_e, number_payments_dd, nominal_value_e, yield_rate_e, coupon_rate_e, compounding_frequency_e, change_e)
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")


            if choice_5 == "6":
                try:
                    getinputs()
                except ValueError:
                    print("Error: Invalid input. Please enter a valid number.")

            if choice_5 == "7":
                try:
                    time_to_maturity_cx = int(input("Enter the time to maturity: "))
                    nominal_value_cx = float(input("Enter the nominal value: "))
                    yield_rate_cx = float(input("Enter the yield rate: "))
                    coupon_rate_cx = float(input("Enter the annual coupon rate: "))
                    number_payments_co = int(input("Enter the number of coupon payments per year: "))
                    compounding_frequency_cx = int(input("Enter the compounding frequency: "))
                    results_cx = convexity(time_to_maturity_cx, number_payments_co,nominal_value_cx, yield_rate_cx, coupon_rate_cx, compounding_frequency_cx )
                    print(f"The convexity of the bond is: {results_cx[0]:.3f}, the dollar convexity is: {results_cx[1]:.2f}.\nFor a 1% increase in the yield, the price of the bond will be reduced to {results_cx[2]:.2f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

        elif choice_1 == "3":   
            print("\n1: Bond price with spot Rates\n2: Forward Rates\n3: Quasi-Modified duration\n4: Fisher-Weil duration\n") 
            choice_6 = input()

            if choice_6 == "1":  
                try:                
                    time_to_maturity_srp = int(input("Enter the time to maturity: "))
                    nominal_value_srp = float(input("Enter the nominal value: "))                   
                    coupon_rate_srp = float(input("Enter the coupon rate: "))
                    print("\n1: Discrete compounding\n2: Continuous compounding")
                    choice_8 = input()
                    if choice_8 == "1":                       
                        compounding_frequency_srp = int(input("Enter the compounding frequency: "))                  
                        bond_price_srp = bond_price_spot_rates(time_to_maturity_srp, nominal_value_srp, coupon_rate_srp, compounding_frequency_srp)
                        print(f"The price of the bond is {bond_price_srp:.2f}")
                    if choice_8 == "2":
                        bond_price_srp2 = bond_price_spot_rates_2(time_to_maturity_srp, nominal_value_srp, coupon_rate_srp)
                        print(f"The price of the bond is {bond_price_srp2:.2f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            if choice_6 == "2":
                try:
                    spot1 = float(input("Enter the first spot rate: "))
                    spot2 = float(input("Enter the second spot rate: "))
                    time1 = float(input("Enter the year of the first spot rate: "))
                    time2 = float(input("Enter the year of the second spot rate: "))
                    print("Enter '1' for discrete compounding, enter '2' for continuous compounding")
                    choice_9 = input()
                    if choice_9 == "1":
                        compounding_rate_fr = int(input("Enter the compounding frequency: "))
                    if choice_9 == "2":
                        compounding_rate_fr = "inf"
                    forward_rate_final = forward_rate(spot1, spot2, time1, time2, compounding_rate_fr)
                    print(f"The forward rate is {forward_rate_final:.3f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")


            if choice_6 == "3":
                try:
                    time_to_maturity_md = int(input("Enter the time to maturity: "))
                    nominal_value_md = float(input("Enter the nominal value: "))
                    coupon_rate_md = float(input("Enter the coupon rate: "))
                    compounding_frequency_md = int(input("Enter the compounding frequency: "))            
                    final_md = bond_duration_spot_rates(time_to_maturity_md, nominal_value_md, coupon_rate_md, compounding_frequency_md)
                    print(f"The quasi-modified duration of the bond is: {final_md:.3f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            if choice_6 == "4":
                try:
                    time_to_maturity_bf = int(input("Enter the time to maturity: "))
                    nominal_value_bf = float(input("Enter the nominal value: "))
                    coupon_rate_bf = float(input("Enter the coupon rate: "))
                    compounding_frequency_bb = int(input("Enter the coupon payment frequency per year: "))
                    final_bf = bobby_fisher(time_to_maturity_bf, nominal_value_bf, coupon_rate_bf, compounding_frequency_bb)
                    print(f"The Fisher-Weil duration of the bond is: {final_bf:.3f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

        elif choice_1 == "4":   
            print("\n1: Asset Returns\n2: Short Selling\n3: Expected Return, Variance, Covariance, Correlation\n4: Portfolio expected return and variance\n5: Minimum Variance Portfolio\n6: Mean-variance portfolio\n") 
            choice_10 = input()

            if choice_10 == "1": 
                try:
                    asset_rr = float(input("Enter asset price at time 0: "))
                    asset_rr_2 = float(input("Enter asset price at time 1: "))
                    final_asset_rr = asset_returns(asset_rr, asset_rr_2)
                    print(f"The gross return is : {final_asset_rr[0]:.3f}; the arithmetic return is: {final_asset_rr[1]:.3f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")


            elif choice_10 == "2": 
                try:
                    asset_sp = float(input("Enter the short position: "))
                    asset_sp_2 = float(input("Enter the long position: "))
                    final_asset_sp = short_selling(asset_sp, asset_sp_2)
                    print(f"The final cashflow is: {final_asset_sp[0]:.3f}; the profit(loss) is: {final_asset_sp[1]:.3f}")
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_10 == "3":
                try:
                    asset_vv = float(input("Enter the price of the first stock at Si,0: "))
                    asset_vv_2 = float(input("Enter the price of the second stock at Si,0: "))
                    periods_vv = int(input("Enter the number of state-dependent time-1 prices (n) of the two stocks: "))
                    finale_vv = stock_r_v_c_c(asset_vv, asset_vv_2, periods_vv)
                    print(f"The expected return of the first stock is: {finale_vv[0]:.3f} \
                        \nThe expected return of the second stock is: {finale_vv[1]:.3f}\
                        \nThe variance of the first stock is: {finale_vv[2]:.3f}\
                        \nThe variance of the second stock is: {finale_vv[3]:.3f}\
                        \nThe covariance is: {finale_vv[4]:.3f}\
                        \nThe correlation is: {finale_vv[5]:.3f} ") 
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")            

            elif choice_10 == "4":
                try:
                    asset_pv = float(input("Enter the price of the first stock at Si,0: "))
                    asset_pv_2 = float(input("Enter the price of the second stock at Si,0: "))
                    periods_pv = int(input("Enter the number of state-dependent time-1 prices (n) of the two stocks: "))
                    weight_1_pv = float(input("Enter the weight of the first stock: "))
                    weight_2_pv = float(input("Enter the weight of the second stock: "))
                    finale_pv = portfolio_r_v(asset_pv, asset_pv_2, periods_pv, weight_1_pv, weight_2_pv)
                    print(f"The expected return of the portfolio is: {finale_pv[0]:.3f} \
                        \nThe variance of the portfolio is: {finale_pv[1]:.3f}") 
                except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

            elif choice_10 == "5":
                print("Enter '2' for minimum variance portfolio between 2 stock or '3' for minimum variance portfolio between 3 stocks ")
                choice_11 = input()
                if choice_11 == "2":
                    try:
                        vari_1 = float(input("Enter variance of the first stock: "))
                        vari_2 = float(input("Enter variance of the second stock: "))
                        covi_1 = float(input("Enter covariance between the two stock: "))
                        covi_2_finale = minim_variance_2(vari_1,vari_2,covi_1)
                        print(f"The weight of the first stock is {covi_2_finale[0]:.3f}, the weight of the second stock is {covi_2_finale[1]:.3f}, Phi is {covi_2_finale[2]:.3f}  ")
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

                if choice_11 == "3":
                    vari_13 = float(input("Enter variance of the first stock: "))
                    vari_23 = float(input("Enter variance of the second stock: "))
                    vari_33 = float(input("Enter variance of the third stock: "))
                    covi_13 = float(input("Enter covariance between the stocks 1 and 2: "))
                    covi_23 = float(input("Enter covariance between the stocks 1 and 3: "))
                    covi_33 = float(input("Enter covariance between the stocks 2 and 3: "))
                    covi_3_finale = minim_variance_3(vari_13,vari_23, vari_33,covi_13, covi_23, covi_33)
                    print(f"The weight of the first stock is {covi_3_finale[0]:.3f}, the weight of the second stock is {covi_3_finale[1]:.3f}, the weight of the third stock is {covi_3_finale[2]:.3f}, Phi is {covi_3_finale[3]:.3f} ")

             
            elif choice_10 == "6":
                print("Enter '2' for mean-variance efficient portfolio between 2 stock or '3' for mean-variance portfolio between 3 stocks ")
                choice_12 = input()
                if choice_12 == "2":
                    try:
                        vari_113 = float(input("Enter variance of the first stock: "))
                        vari_123 = float(input("Enter variance of the second stock: "))        
                        covi_113 = float(input("Enter covariance between the stocks: "))
                        rate_1_eff = float(input("Enter the expected return of the first stock: "))
                        rate_2_eff = float(input("Enter the expected return of the second stock: "))            
                        rate_e_eff = float(input("Enter the desired return: "))
                        covi_13_finale = minimize_variance_eff_4(vari_113, vari_123, covi_113, rate_1_eff, rate_2_eff, rate_e_eff)
                        print(f"The weights of the two stocks are: {covi_13_finale[0]:.3f}, {covi_13_finale[1]:.3f}, lamda is {covi_13_finale[2]:.3f}, and Phi is {covi_13_finale[3]:.3f}")
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")

                if choice_12 == "3":
                    try:
                        vari_113 = float(input("Enter variance of the first stock: "))
                        vari_123 = float(input("Enter variance of the second stock: "))
                        vari_133 = float(input("Enter variance of the third stock: "))
                        covi_113 = float(input("Enter covariance between the stocks 1 and 2: "))
                        covi_123 = float(input("Enter covariance between the stocks 1 and 3: "))
                        covi_133 = float(input("Enter covariance between the stocks 2 and 3: "))
                        rate_1_eff = float(input("Enter the expected return of the first stock: "))
                        rate_2_eff = float(input("Enter the expected return of the second stock: "))
                        rate_3_eff = float(input("Enter the expected return of the third stock: "))
                        rate_e_eff = float(input("Enter the desired return: "))
                        covi_13_finale = minimize_variance_eff_5(vari_113, vari_123, vari_133, covi_113, covi_123, covi_133, rate_1_eff, rate_2_eff,rate_3_eff, rate_3_eff)
                        print(f"The weights of the three stocks are: {covi_13_finale[0]:.3f}, {covi_13_finale[1]:.3f}, {covi_13_finale[2]:.3f}, lamda is {covi_13_finale[3]:.3f}, and Phi is {covi_13_finale[4]:.3f}")
                    
                    except ValueError:
                        print("Error: Invalid input. Please enter a valid number.")
        else:
            print("Invalid choice. Please try again.")

print("Welcome to 'Quantitative Finance 2023'.\nNavigate the menu to discover tailored solutions for your financial challenges.")    
menu()