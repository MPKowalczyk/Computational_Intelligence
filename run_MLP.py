# Run MLP for test data

from MLP import MLP
from functions import fun2 as fun
from functions import dfun2 as dfun
from interface import read

#Parameters
inputs=4
outputs=3
name='IrisData.xls'
sizes=[6,5,3]
in_data,out_data=read(name,inputs,outputs)

network=MLP(sizes,lambda x: fun(x,1),lambda x: dfun(x,1))
network.full_connection()
ni=1
for j in range(0,100):
    for i in range(0,50):
        network.connect_inputs_full(in_data[i])
        network.connect_outputs(out_data[i])
        network.online_training(ni)
        network.connect_inputs_full(in_data[i+50])
        network.connect_outputs(out_data[i+50])
        network.online_training(ni)
        network.connect_inputs_full(in_data[i+100])
        network.connect_outputs(out_data[i+100])
        network.online_training(ni)
    ni=ni*0.99