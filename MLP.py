# MLP for computational intelligence
# table - numer of neurons in layers

from layer import layer, initial_layer, final_layer

class MLP:
    
    def __init__(self,table,arg_fun,arg_dfun):
        self.layers=[]
        self.layers.append(initial_layer(table[0],[],[],arg_fun,arg_dfun))
        for i in range(1,len(table)-1):
            self.layers.append(layer(table[i],self.layers[i-1],[],arg_fun,arg_dfun))
            self.layers[i-1].change_next(self.layers[i])
        self.layers.append(final_layer(table[-1],self.layers[len(table)-2],[],arg_fun,arg_dfun))
        self.layers[len(table)-2].change_next(self.layers[len(table)-1])

    def full_connection(self):
        for j in self.layers[0].neurons:
            j.change_outputs(self.layers[1].neurons)
        for i in range(1,len(self.layers)-1):
            for j in self.layers[i].neurons:
                j.change_outputs(self.layers[i+1].neurons)
                j.change_inputs(self.layers[i-1].neurons)
        for j in self.layers[len(self.layers)-1].neurons:
            j.change_inputs(self.layers[len(self.layers)-2].neurons)
    
    def connect_inputs_full(self,arg_inputs):
        for i in self.layers[0].neurons:
            i.change_inputs(arg_inputs)

    def connect_outputs(self,arg_outputs):
        k=0
        for i in self.layers[-1].neurons:
            i.outputs=arg_outputs[k]
            k=k+1
    
    def count_output(self):
        for i in self.layers:
            i.count_output()
    
    def back_propagation(self):
        for i in reversed(self.layers):
            i.back_propagation()
    
    def clear_delta(self):
        for i in self.layers:
            i.clear_delta()
    
    def update_weight(self,ni):
        for i in self.layers:
            i.update_weight(ni)
            
    def online_training(self,ni):
        self.count_output()
        self.clear_delta()
        self.back_propagation()
        self.update_weight(ni)
    
    def print_output(self):
        for i in self.layers[-1].neurons:
            print(i.Y)
    
    def print_weights(self):
        for j in self.layers:
            print('New layer')
            for k in j.neurons:
                print('New neuron')
                for z in k.weights:
                    print(z)
        print('New step')