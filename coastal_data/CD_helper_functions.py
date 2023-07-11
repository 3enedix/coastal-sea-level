from datetime import datetime
import numpy as np
import pandas as pd

def decimal_numbers_to_datetime(decim):
    years = decim.astype('int')
    doy = np.round(((decim - years) *12 *30.5),0).astype('int')

    date = []
    for i in range(0,len(doy)):
        if doy[i] == 0:
            day = (doy[i] + 1).astype('str')
        else:
            day = (doy[i]).astype('str')
        day.rjust(3 + len(day), '0')
        year = years[i].astype('str')
        date.append(datetime.strptime(year + "-" + day, "%Y-%j").strftime("%m-%d-%Y"))

    date = pd.to_datetime(date, utc=True)
    return date

def datetime_to_decimal_numbers(date):
    return(date.year + (date.dayofyear - 1)/365.25)