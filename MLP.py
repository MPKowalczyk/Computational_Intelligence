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
        
    def train(self,inputs,outputs,iters,ni,full_inputs,full_outputs):
        error=list()
        correct=list()
        for i in range(0,iters):
            error.append(0)
            correct.append(0)
            for j in range(0,len(inputs)):
                self.connect_inputs_full(inputs[j])
                self.connect_outputs(outputs[j])
                self.online_training(ni)
            for k in range(0,len(full_inputs)):
                self.connect_inputs_full(full_inputs[k])
                self.connect_outputs(full_outputs[k])
                self.count_output()
                out=self.print_output()
                error[i]=error[i]+sum((out-full_outputs[k])**2)
                if(out.index(max(out))==list(full_outputs[k]).index(max(full_outputs[k]))):
                    correct[i]+=1
            ni=ni*0.99
        return error,correct
    
    def print_output(self):
        out=list()
        for i in self.layers[-1].neurons:
            out.append(i.Y)
        return out
    
    def print_weights(self):
        for j in self.layers:
            print('New layer')
            for k in j.neurons:
                print('New neuron')
                for z in k.weights:
                    print(z)
        print('New step')