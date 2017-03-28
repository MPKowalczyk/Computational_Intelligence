# Layer class for computational intelligence

from neuron import neuron, initial_neuron, final_neuron

class layer:

    def __init__(self,n,arg_prev,arg_next,arg_fun,arg_dfun):
        self.prev=arg_prev
        self.next=arg_next
        self.neurons=[]
        for i in range(0,n):
            self.neurons.append(neuron([],[],arg_fun,arg_dfun))
            
    def change_prev(self,arg_prev):
        self.prev=arg_prev
    
    def change_next(self,arg_next):
        self.next=arg_next
    
    def count_output(self):
        for i in self.neurons:
            i.count_output()
    
    def back_propagation(self):
        for i in self.neurons:
            i.back_propagation()
    
    def clear_delta(self):
        for i in self.neurons:
            i.delta=0
    
    def update_weight(self,ni):
        for i in self.neurons:
            i.update_weight(ni)
    
#    def full_connect(self):
#        for i in self.neurons:
#            i.change_inputs(self.prev.neurons)
#            i.change_inputs(self.prev.neurons)

class initial_layer:

    def __init__(self,n,arg_prev,arg_next,arg_fun,arg_dfun):
        self.prev=arg_prev
        self.next=arg_next
        self.neurons=[]
        for i in range(0,n):
            self.neurons.append(initial_neuron([],[],arg_fun,arg_dfun))
            
    def change_prev(self,arg_prev):
        self.prev=arg_prev
    
    def change_next(self,arg_next):
        self.next=arg_next
    
    def count_output(self):
        for i in self.neurons:
            i.count_output()

    def back_propagation(self):
        for i in self.neurons:
            i.back_propagation()
    
    def clear_delta(self):
        for i in self.neurons:
            i.delta=0

    def update_weight(self,ni):
        for i in self.neurons:
            i.update_weight(ni)

#    def full_connect(self):
#        for i in self.neurons:
#            i.change_inputs(self.prev.neurons)

class final_layer:

    def __init__(self,n,arg_prev,arg_next,arg_fun,arg_dfun):
        self.prev=arg_prev
        self.next=arg_next
        self.neurons=[]
        for i in range(0,n):
            self.neurons.append(final_neuron([],[],arg_fun,arg_dfun))
            
    def change_prev(self,arg_prev):
        self.prev=arg_prev
    
    def change_next(self,arg_next):
        self.next=arg_next
    
    def count_output(self):
        for i in self.neurons:
            i.count_output()

    def back_propagation(self):
        for i in self.prev.neurons:
            i.delta=0
        for i in self.neurons:
            i.back_propagation()
    
    def clear_delta(self):
        for i in self.neurons:
            i.delta=0

    def update_weight(self,ni):
        for i in self.neurons:
            i.update_weight(ni)

#    def full_connect(self):
#        for i in self.neurons:
#            i.changel_inputs(self.prev.neurons)