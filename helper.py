### Various helper functions

import pandas as pd


# takes in a string of query or an SQL file
def SQL(query, con):
    if query[-4:] != ".sql":
        return pd.read_sql(query, con)
    else:
        with open(query, "r") as f:
            return pd.read_sql(f.read(), con)


# given a string t in the format of n (int) + unit (unit of int, y=year, m=month), convert to number of months (int)
def time_convert(t):
    unit = t[-1]
    s = ""
    # go through the string, if it's not a digit, modify accordingly
    for c in t[:-1]:
        if not c.isdigit():
            # less than or equal, ignore the sign
            if c == ">" or c == "<":
                pass
            # a range, take the longest of the range
            elif c == "-":
                s = ""
        else:
            s += c
    n = int(s)
    if unit == "y":
        return n * 12
    else:
        return n