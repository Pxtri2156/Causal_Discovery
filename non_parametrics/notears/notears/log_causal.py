
class LogCausal():
    def __init__(self,name, lamda, key_list):
        # super(Logging, self).__init__()
        self.log = {}
        self.log['name'] = name
        self.log['lamda'] = lamda
        self.log['key_list'] = key_list

        #log init by name 
        for k in key_list: 
            self.log[k] = {}

        #score, num epoch each random seed log init 
        self.log['num_epoch'] = {} 

        self.log['score'] = {}
        self.log['score']['fdr']= {} 
        self.log['score']['fpr']= {}
        self.log['score']['tpr']= {}
        self.log['score']['shd']= {}
        self.log['score']['nnz']= {}

        self.log['random_seed'] = []
        self.acc = []
        self.acc_average = {'fdr': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'shd': 0, 'nnz': 0}

    def step_update(self, result_dict):
        for k in self.log['key_list']: 
            self.log[k][self.log['random_seed'][-1]].append(result_dict[k])

    def random_seed_update(self,r): 
        self.log['random_seed'].append(r)
        for k in self.log['key_list']: 
            self.log[k][r] = []
    
    def acc_update(self, acc, num_epoch): 
        self.acc.append(acc)
        self.log['score']['fdr'][self.log['random_seed'][-1]] = acc['fdr'] 
        self.log['score']['fpr'][self.log['random_seed'][-1]] = acc['fpr']
        self.log['score']['tpr'][self.log['random_seed'][-1]] = acc['tpr']
        self.log['score']['shd'][self.log['random_seed'][-1]] = acc['shd']
        self.log['score']['nnz'][self.log['random_seed'][-1]] = acc['nnz']
        self.log['num_epoch'][self.log['random_seed'][-1]] = num_epoch

    
    def print_acc_average(self): 
        for i in self.acc: 
            self.acc_average['fdr'] += i['fdr']
            self.acc_average['tpr'] += i['tpr']
            self.acc_average['fpr'] += i['fpr']
            self.acc_average['shd'] += i['shd']
        self.acc_average['fdr'] /= len(self.acc)
        self.acc_average['tpr'] /= len(self.acc)
        self.acc_average['fpr'] /= len(self.acc)
        self.acc_average['shd'] /= len(self.acc)    
        print(f'average: | {self.acc_average}|')
        




    

