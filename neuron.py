# Neuron class for computational intelligence

import numpy as np

class neuron:

    def __init__(self,arg_inputs,arg_outputs,arg_fun,arg_dfun):
        self.inputs=arg_inputs
        self.outputs=arg_outputs
        self.fun=arg_fun
        self.dfun=arg_dfun
        self.weights=np.random.rand(len(self.inputs)+1)-0.5
        self.delta=0
        self.bias=1
        self.sum=0
        self.Y=0
        #self.count_output()

    def count_output(self):
        self.sum=self.bias*self.weights[0]
        for i in range(0,len(self.inputs)):
            self.sum=self.sum+self.inputs[i].Y*self.weights[i+1]
        self.Y=self.fun(self.sum)
    
#    def count_delta(self):
#        self.delta=0
#        for i in self.outputs:
#            self.delta=self.delta+i.delta#*
    def back_propagation(self):
        for i in range(0,len(self.inputs)):
            #self.inputs[i].delta=self.inputs[i].delta+self.dfun(self.inputs[i].sum)*self.delta*self.weights[i+1]
            self.inputs[i].delta=self.inputs[i].delta+self.delta*self.weights[i+1]*(1-self.inputs[i].Y)*self.inputs[i].Y
    
    def change_inputs(self,arg_inputs):
        self.inputs=arg_inputs
        self.weights=np.random.rand(len(self.inputs)+1)-0.5
    
    def change_outputs(self,arg_outputs):
        self.outputs=arg_outputs
    
    def change_fun(self,arg_fun,arg_dfun):
        self.fun=arg_fun
        self.dfun=arg_dfun
    
    def update_weight(self,ni):
        self.weights[0]=self.weights[0]+ni*self.delta*self.dfun(self.sum)
        for i in range(1,len(self.weights)):
            #self.weights[i]=self.weights[i]+ni*self.delta*self.dfun(self.sum)*self.inputs[i-1].Y
            self.weights[i]=self.weights[i]-ni*self.delta*self.inputs[i-1].Y

class initial_neuron:

    def __init__(self,arg_inputs,arg_outputs,arg_fun,arg_dfun):
        self.inputs=arg_inputs
        self.outputs=arg_outputs
        self.fun=arg_fun
        self.dfun=arg_dfun
        self.weights=np.random.rand(len(self.inputs)+1)-0.5
        self.delta=0
        self.bias=1
        self.sum=0
        self.Y=0
        #self.count_output()

    def count_output(self):
        self.sum=self.bias*self.weights[0]
        for i in range(0,len(self.inputs)):
            self.sum=self.sum+self.inputs[i]*self.weights[i+1]
        self.Y=self.fun(self.sum)
    
#    def count_delta(self):
#        self.delta=0
#        for i in self.outputs:
#            self.delta=self.delta+i.delta#*
    def back_propagation(self):
        return
    
    def change_inputs(self,arg_inputs):
        if(len(self.inputs)!=len(arg_inputs)):
            self.weights=np.random.rand(len(arg_inputs)+1)-0.5
        self.inputs=arg_inputs
    
    def change_outputs(self,arg_outputs):
        self.outputs=arg_outputs
    
    def change_fun(self,arg_fun,arg_dfun):
        self.fun=arg_fun
        self.dfun=arg_dfun

    def update_weight(self,ni):
        self.weights[0]=self.weights[0]+ni*self.delta*self.dfun(self.sum)
        for i in range(1,len(self.weights)):
            #self.weights[i]=self.weights[i]+ni*self.delta*self.dfun(self.sum)*self.inputs[i-1]
            self.weights[i]=self.weights[i]-ni*self.delta*self.inputs[i-1]

class final_neuron:

    def __init__(self,arg_inputs,arg_outputs,arg_fun,arg_dfun):
        self.inputs=arg_inputs
        self.outputs=arg_outputs
        self.fun=arg_fun
        self.dfun=arg_dfun
        self.weights=np.random.rand(len(self.inputs)+1)-0.5
        self.delta=0
        self.bias=1
        self.sum=0
        self.Y=0
        #self.count_output()

    def count_output(self):
        self.sum=self.bias*self.weights[0]
        for i in range(0,len(self.inputs)):
            self.sum=self.sum+self.inputs[i].Y*self.weights[i+1]
        self.Y=self.fun(self.sum)
    
#    def count_delta(self):
#        self.delta=0
#        for i in self.outputs:
#            self.delta=self.delta+i.delta#*
    def back_propagation(self):
        self.delta=self.Y-self.outputs
        for i in range(0,len(self.inputs)):
            #self.inputs[i].delta=self.inputs[i].delta+self.delta*self.weights[i+1]
            self.inputs[i].delta=self.inputs[i].delta+self.delta*self.weights[i+1]*(1-self.inputs[i].Y)*self.inputs[i].Y
    
    def change_inputs(self,arg_inputs):
        self.inputs=arg_inputs
        self.weights=np.random.rand(len(self.inputs)+1)-0.5
    
    def change_outputs(self,arg_outputs):
        self.outputs=arg_outputs
    
    def change_fun(self,arg_fun,arg_dfun):
        self.fun=arg_fun
        self.dfun=arg_dfun

    def update_weight(self,ni):
        self.weights[0]=self.weights[0]+ni*self.delta*self.dfun(self.sum)
        for i in range(1,len(self.weights)):
            #self.weights[i]=self.weights[i]+ni*self.delta*self.dfun(self.sum)*self.inputs[i-1].Y
            self.weights[i]=self.weights[i]-ni*self.delta*self.inputs[i-1].Y