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
        self.lamda = log['lamda']
    def visualize(self):
        
        # plt.text(0, 0, 'Parabola $Y = x^2$', fontsize = 22)
        figure, axis = plt.subplots(len(self.random_seed) + 1, figsize=(15, 100))
        #color 
        obj = 'r'
        loss = 'b'
        orth = 'g'
        h = 'y'
        
        figure.suptitle(f'lamda_dag={self.lamda[0]}, lamda_l2={self.lamda[1]}, lamda_ortho={self.lamda[2]}', fontsize = 15)

        plt.title(f'lamda_dag={self.lamda[0]}, lamda_l2={self.lamda[1]}, lamda_ortho={self.lamda[2]}')

        #score
        fdr_scores = list(self.score['fdr'].values())
        tpr_scores = list(self.score['tpr'].values())
        fpr_scores = list(self.score['fpr'].values())
        shd_scores = list(self.score['shd'].values())
        all_scores = [fdr_scores, tpr_scores, fpr_scores, shd_scores]
        axis[0].boxplot(all_scores, labels=['FDR', 'TPR', 'FPR', 'SHD'])
        
        axis[0].set_xlabel('Metrics')
        axis[0].set_ylabel('Scores')
        axis[0].set_title(f'Metric Scores')
        axis[0].set_xticklabels(['FDR', 'TPR', 'FPR', 'SHD'])

        #plot in each random seed 
        for idx, seed in enumerate(self.random_seed): 
            idx+=1
            X = range(self.num_epoch[seed])

            # For Objective Function
            axis[idx].plot(X, self.obj[seed],color=obj,label='objective function')

            # For Loss Function
            axis[idx].plot(X, self.loss[seed],color=loss,label='squared loss')

            # For Orthogonal Function
            axis[idx].plot(X, self.orth[seed],color=orth,label= 'orthogonal contraints')

            # For h Function
            axis[idx].plot(X, self.h[seed],color=h,label='h function')

            axis[idx].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axis[idx].set_title('Random seed: ' +  str(seed))
        
        
        plt.subplots_adjust( wspace=0.5, hspace=0.5, right=0.8)
        name = self.name + '.png'
        save_fig = os.path.join(self.path, name)
        plt.savefig(save_fig)

def main(): 
    log = dict()
    log['random_seed'] = {200,3,4}
    log['obj_func'] = {200:[1,2,3,4,5,6,7,8,9],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['loss'] = {200:[9,8,7,6,5,4,3,2,1],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['ortho'] = {200:[5,5,5,5,6,6,4,4,5],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['h_func'] = {200:[1,2,3,1,2,3,3,2,1],3:[9,8,7,6,5,4,3,2,1],4:[5,5,5,5,6,6,4,4,5]}
    log['score']= {'fdr':{200:1,3:9,4:5},
                    'fpr':{200:9,3:7,4:4},
                    'tpr': {200:9,3:4,4:6},
                    'shd':{200:5,3:3,4:4},}
    log['name'] = 'first_test'
    log['num_epoch'] = {200:9,3:9,4:9}
    log['lamda'] = [0.1, 0.1, 0.1]
    save_path = 'ScalableDAG_v2/visualization'
    vis = Visualization(log, save_path)
    vis.visualize()
    return 
if __name__ == '__main__':
    main()