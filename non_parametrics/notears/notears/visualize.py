import matplotlib.pyplot as plt
import numpy as np
import os
class Visualization():
    def __init__(self, log, save_path):
        super(Visualization, self).__init__()
        self.random_seed = log['random_seed']
        self.obj = log['obj_func'] 
        self.loss = log['loss']
        self.orth = log['ortho']
        self.h = log['h_func']
        self.score = log['score'] 
        self.name = log['name']
        self.path = save_path
        self.num_epoch = log['num_epoch']
    def visualize(self):
        
        figure, axis = plt.subplots(4, 2,gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1, 1]}, figsize=(10, 8))
        for seed in self.random_seed: 
            X = range(self.num_epoch[seed])
            r = np.round(np.random.rand(),1)
            g = np.round(np.random.rand(),1)
            b = np.round(np.random.rand(),1)

            # For Objective Function
            axis[0, 0].plot(X, self.obj[seed],color=[r,g,b],label= seed)
            axis[0, 0].set_title("Objective Function")
            # For Loss Function
            axis[0, 1].plot(X, self.loss[seed],color=[r,g,b],label= seed)
            axis[0, 1].set_title("Loss Function")
            # For Loss Function
            axis[1, 0].plot(X, self.orth[seed],color=[r,g,b],label= seed)
            axis[1, 0].set_title("Orthogonal Constraint")
            # For h Function
            axis[1, 1].plot(X, self.h[seed],color=[r,g,b],label= seed)
            axis[1, 1].set_title("h Function")
            # For SCORE Function
            axis[2, 0].plot(X, self.score['tpr'][seed],color=[r,g,b],label= seed)
            axis[2, 0].set_title("TPR Function")

            axis[2, 1].plot(X, self.score['fpr'][seed],color=[r,g,b],label= seed)
            axis[2, 1].set_title("FPR Function")

            axis[3, 0].plot(X, self.score['fdr'][seed],color=[r,g,b],label= seed)
            axis[3, 0].set_title("FDR Function")

            axis[3, 1].plot(X, self.score['shd'][seed],color=[r,g,b],label= seed)
            axis[3, 1].set_title("SHD Function")

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust( wspace=0.5, hspace=0.5, right=0.8)
        name = self.name + '.png'
        save_fig = os.path.join(self.path, name)
        plt.savefig(save_fig)

def main(): 
    log = dict()
    log['random_seed'] = {200,3,4}
    log['obj_func'] = {200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['loss'] = {200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['ortho'] = {200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['h_func'] = {200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['score']= {'fdr':{200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]},
                    'fpr':{200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]},
                    'tpr': {200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]},
                    'shd':{200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]},}
    log['name'] = 'first_test'
    log['num_epoch'] = {200:9,3:9,4:9}
    save_path = 'ScalableDAG_v2/visualization'
    vis = Visualization(log, save_path)
    vis.visualize()
    return 
if __name__ == '__main__':
    main()