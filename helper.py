### Various helper functions

import pandas as pd

def SQL(query, con):
    return pd.read_sql(query, con)