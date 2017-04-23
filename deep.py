# Simple Deep class

from MLP import MLP

class deep:
    
    def __init__(self,sizes,arg_fun,arg_dfun):
        self.mlp_list=list()
        for i in sizes:
            self.mlp_list.append(MLP(i,arg_fun,arg_dfun))
    
    def add_MLP(self,sizes,arg_fun,arg_dfun):
        self.mlp_list.append(MLP(sizes,arg_fun,arg_dfun))
    
    def count_output(self):
        for i in self.mlp_list:
            i.count_output()
    
    def full_connection(self):
        for i in self.mlp_list:
            i.full_connection()
        #self.mlp_list[-1].full_connection()
    
    def train(self,inputs,outputs,number,ni,full_inputs,full_outputs):
        error=list()
        correct=list()
        err_temp,corr_temp=self.mlp_list[0].train(inputs,outputs,number,ni,full_inputs,full_outputs)
        error+=err_temp
        correct+=corr_temp
        prev_mlp=self.mlp_list[0]
        temp_inputs=list(inputs)
        temp_full_inputs=list(full_inputs)
        for i in range(1,len(self.mlp_list)):
            prev_outs=list()
            for k in range(0,len(inputs)):
                prev_mlp.connect_inputs_full(temp_inputs[k])
                prev_mlp.count_output()
                prev_outs.append(prev_mlp.print_output())
                temp_inputs[k]=prev_outs[k]+inputs[k].tolist()
            prev_full_outs=list()
            for k in range(0,len(full_inputs)):
                prev_mlp.connect_inputs_full(temp_full_inputs[k])
                prev_mlp.count_output()
                prev_full_outs.append(prev_mlp.print_output())
                temp_full_inputs[k]=prev_full_outs[k]+full_inputs[k].tolist()
            err_temp,corr_temp=self.mlp_list[i].train(temp_inputs,outputs,number,ni,temp_full_inputs,full_outputs)
            error+=err_temp
            correct+=corr_temp
            prev_mlp=self.mlp_list[i]
        return error,correct
    
    
    def connect_inputs_full(self,arg_inputs):
        self.mlp_list[0].connect_inputs_full(arg_inputs)
        self.mlp_list[0].count_output()
        prev_mlp=self.mlp_list[0]
        for i in range(1,len(self.mlp_list)):
            self.mlp_list[i].connect_inputs_full(prev_mlp.print_output()+arg_inputs)
            self.mlp_list[i].count_output()
            prev_mlp=self.mlp_list[i]
    
    def print_output(self):
        return self.mlp_list[-1].print_output()