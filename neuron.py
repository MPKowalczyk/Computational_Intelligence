# Neuron class for computational intelligence

import numpy as np

class neuron:

    def __init__(self,arg_inputs,arg_outputs,arg_fun):
        self.inputs=arg_inputs
        self.outputs=arg_outputs
        self.fun=arg_fun
        self.weights=np.random.rand(len(self.inputs)+1)
        self.count_output()
        self.delta=0
        self.bias=1

    def count_output(self):
        self.sum=0;
        for i in self.inputs:
            self.sum=self.sum+self.inputs(i).sum
    delta                                           # Delta parameter
    Y                                               # Output value