# Input interface for computational intelligence

import pandas as pd

def read(name,n_in,n_out):
    data=pd.read_excel(name)
    in_data=data.values[:,0:n_in]
    out_data=data.values[:,n_in:(n_in+n_out)]
    return in_data,out_data