# Run MLP for test data

from MLP import MLP
from functions import fun1 as fun
from functions import dfun1 as dfun
from interface import read
import matplotlib.pyplot as plt

#Parameters
inputs=4
outputs=3
name='IrisData.xls'
sizes=[4,5,8,3]
in_data,out_data=read(name,inputs,outputs)
input_data=list()
output_data=list()
for i in range(0,50):
    output_data.append(out_data[i])
    output_data.append(out_data[50+i])
    output_data.append(out_data[100+i])
network=MLP(sizes,lambda x: fun(x,1),lambda x: dfun(x,1))
network.full_connection()
network.train()
ni=0.1
error=list()
correct=list()
#for j in range(0,100):
#    error.append(0)
#    correct.append(0)
#    for i in range(0,50):
#        network.connect_inputs_full(in_data[i])
#        network.connect_outputs(out_data[i])
#        network.online_training(ni)
#        network.connect_inputs_full(in_data[i+50])
#        network.connect_outputs(out_data[i+50])
#        network.online_training(ni)
#        network.connect_inputs_full(in_data[i+100])
#        network.connect_outputs(out_data[i+100])
#        network.online_training(ni)
#    for i in range(0,150):
#        network.connect_inputs_full(in_data[i])
#        network.connect_outputs(out_data[i])
#        network.count_output()
#        out=network.print_output()
#        error[j]=error[j]+sum((out-out_data[i])**2)
#        if(out.index(max(out))==list(out_data[i]).index(max(out_data[i]))):
#            correct[j]+=1
#    ni=ni*0.99

plt.figure()
plt.plot(error)
plt.figure()
plt.plot(correct)