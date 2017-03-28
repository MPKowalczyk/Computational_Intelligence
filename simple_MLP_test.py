# Simple MLP test

from MLP import MLP
from functions import fun2 as fun
from functions import dfun2 as dfun

inputs=[[0.05,0,0,0],[0,0.05,0,0],[0,0,0.05,0],[0,0,0,0.05]]
outputs=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
sizes=[4,4]

network=MLP(sizes,lambda x: fun(x,1),lambda x: dfun(x,1))
network.full_connection()
ni=1
for i in range(0,1000):
    for j in range(0,4):
        network.connect_inputs_full(inputs[j])
        network.connect_outputs(outputs[j])
        network.online_training(ni)
    ni=ni*0.99
        #network.print_weights()

for j in range(0,4):
    network.connect_inputs_full(inputs[j])
    network.connect_outputs(outputs[j])
    network.count_output()
    network.print_output()