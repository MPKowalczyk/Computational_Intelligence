# Run deep

from deep import deep
from functions import fun1 as fun
from functions import dfun1 as dfun
from interface import read
import matplotlib.pyplot as plt

#Parameters
inputs=4
outputs=3
name='IrisData.xls'
sizes=[[4,3,3],[7,5,3],[7,8,3],[7,5,3],[7,10,3]]
in_data,out_data=read(name,inputs,outputs)
input_data=list()
output_data=list()
for i in range(0,50):
    input_data.append(in_data[i])
    input_data.append(in_data[50+i])
    input_data.append(in_data[100+i])
    output_data.append(out_data[i])
    output_data.append(out_data[50+i])
    output_data.append(out_data[100+i])
network=deep(sizes,lambda x: fun(x,1),lambda x: dfun(x,1))
network.full_connection()
ni=0.1
error=list()
correct=list()
error,correct=network.train(input_data,output_data,100,ni,input_data,output_data)
plt.figure()
plt.plot(error)
plt.figure()
plt.plot(correct)