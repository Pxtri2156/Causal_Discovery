
class LogCausal():
    def __init__(self,name, lamda):
        # super(Logging, self).__init__()
        self.log = {}
        self.log['name'] = name
        self.log['lamda'] = lamda
        self.log['random_seed'] = []
        self.log['obj_func'] = {}
        self.log['loss'] = {}
        self.log['ortho'] = {}
        self.log['h_func'] = {} 
        self.log['num_epoch'] = {} 
        self.log['score'] = {}
        self.log['score']['fdr']= {} 
        self.log['score']['fpr']= {}
        self.log['score']['tpr']= {}
        self.log['score']['shd']= {}
        self.acc = []
        self.acc_average = {'fdr': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'shd': 0}

    
    def random_seed_update(self,r): 
        self.log['random_seed'].append(r)

        self.log['obj_func'][r] = []
        self.log['loss'][r] = []
        self.log['ortho'][r] = []
        self.log['h_func'][r] = []

        self.log['score']['fdr'][r]= []
        self.log['score']['fpr'][r]= []
        self.log['score']['tpr'][r]= []
        self.log['score']['shd'][r]= []
    
    def acc_update(self, acc): 
        self.acc.append(acc)
        self.log['score']['fdr'][self.log['random_seed'][-1]] = acc['fdr'] 
        self.log['score']['fpr'][self.log['random_seed'][-1]] = acc['fpr']
        self.log['score']['tpr'][self.log['random_seed'][-1]] = acc['tpr']
        self.log['score']['shd'][self.log['random_seed'][-1]] = acc['shd']
    
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
        

    def step_update(self, obj_func, loss, ortho, h):
        self.log['obj_func'][self.log['random_seed'][-1]].append(obj_func)
        self.log['loss'][self.log['random_seed'][-1]].append(loss)
        self.log['ortho'][self.log['random_seed'][-1]].append(ortho)
        self.log['h_func'][self.log['random_seed'][-1]].append(h)


    

