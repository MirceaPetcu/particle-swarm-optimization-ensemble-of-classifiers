import numpy as np
from sklearn import metrics
from tqdm import tqdm
import argparse
from sklearn.tree import DecisionTreeClassifier
import logging
import pandas as pd
import random
import warnings
import copy
from rnn import RNN
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--n_particles', type=int, default=50)
parser.add_argument('--inertia', type=float, default=0.9)
parser.add_argument('--cognitive_constant', type=float, default=2.05)
parser.add_argument('--social_constant', type=float, default=2.05)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--max_inertia', type=float, default=0.9)
parser.add_argument('--min_inertia', type=float, default=0.2)
parser.add_argument('--mode', type=str, default='min')
parser.add_argument('--dataset-train', type=str, default='apple_train.csv')
parser.add_argument('--dataset-val', type=str, default='apple_val.csv')
parser.add_argument('--dataset-test', type=str, default='apple_test.csv')
parser.add_argument('--target-variable', type=str, default='close')
parser.add_argument('--no-classes', type=int, default=2)


def softmax(x):
    cp = copy.deepcopy(x)
    return np.exp(cp) / np.sum(np.exp(cp),axis=0)

def encode_one_hot(x,no_classes):
    encoded = np.zeros((x.shape[0],no_classes),dtype=np.float64)
    for i in range(x.shape[0]):
        encoded[i][x[i]] = 1
    return encoded


class Particle:
    def __init__(self,initial_fitness,no_classes) -> None:
        self.dimension = 315
        self.rnn = RNN()
        self.dimension = self.rnn.W_hx.shape[0]*self.rnn.W_hx.shape[1] + self.rnn.W_hh.shape[0]*self.rnn.W_hh.shape[1] + self.rnn.W_hy.shape[0]*self.rnn.W_hy.shape[1]
        self.position = np.concatenate((self.rnn.W_hx.reshape(1,-1).ravel(),self.rnn.W_hh.reshape(1,-1).ravel(),self.rnn.W_hy.reshape(1,-1).ravel()))
        self.velocity = np.random.uniform(low=-2, high=2,size=(self.dimension,))
        self.best_individual_position = self.position
        self.fitness_particle_position = initial_fitness
        self.fitness_best_individual_position = initial_fitness
        self.no_classes = no_classes
    
    def objective_function(self,x,y):
        self.copy_particle_positions_to_ann_weights()
        y_pred = []
        window_size = 30
        y = y[window_size-1:]
        for i in range(window_size-1,x.shape[0],1):
            pred = self.rnn.forward(x[i-window_size+1:i+1])
            y_pred.append(pred)
        y_pred = np.array(y_pred)
        accuracy = metrics.mean_absolute_error(y,y_pred)
        return accuracy
    
    def copy_particle_positions_to_ann_weights(self):
        params_for_W_hx = self.position[:self.rnn.W_hx.shape[0]*self.rnn.W_hx.shape[1]]
        self.rnn.W_hx = params_for_W_hx.reshape(self.rnn.W_hx.shape[0],self.rnn.W_hx.shape[1])
        params_for_W_hh = self.position[self.rnn.W_hx.shape[0]*self.rnn.W_hx.shape[1]:\
                                        self.rnn.W_hx.shape[0]*self.rnn.W_hx.shape[1]\
                                        + self.rnn.W_hh.shape[0]*self.rnn.W_hh.shape[1]]
        self.rnn.W_hh = params_for_W_hh.reshape(self.rnn.W_hh.shape[0],self.rnn.W_hh.shape[1])
        params_for_W_hy = self.position[self.rnn.W_hx.shape[0]*self.rnn.W_hx.shape[1]\
                                        + self.rnn.W_hh.shape[0]*self.rnn.W_hh.shape[1]:]
        self.rnn.W_hy = params_for_W_hy.reshape(self.rnn.W_hy.shape[0],self.rnn.W_hy.shape[1])        
        
    def evaluate(self,x,y):
        self.fitness_particle_position = self.objective_function(x,y)
        if self.fitness_particle_position < self.fitness_best_individual_position or self.fitness_best_individual_position == float('inf'):
            self.best_individual_position = np.copy(self.position)
            self.fitness_best_individual_position = self.fitness_particle_position
    
    def update_position(self):
        for d in range(self.dimension):
            self.position[d] = self.position[d] + self.velocity[d]
        
    
    def update_velocity(self,w,c1,c2,best_group_position):
        for d in range(self.dimension):
            r1 = np.random.rand()
            r2 = np.random.rand()
            cognitive_term = c1*r1*(self.best_individual_position[d] - self.position[d])
            social_term = c2*r2*(best_group_position[d] - self.position[d])
            self.velocity[d] = w*self.velocity[d] + cognitive_term + social_term
        
class PSO():
    def __init__(self,n_particles,inertia,cognitive_constant,social_constant,rnn,initial_fitness,max_iter,max_inertia,min_inertia,x,y,no_classes) -> None:
       self.best_group_position = np.concatenate((rnn.W_hx.reshape(1,-1).ravel(),rnn.W_hh.reshape(1,-1).ravel(),rnn.W_hy.reshape(1,-1).ravel()))
       self.best_group_fitness = initial_fitness
       self.n_particles = n_particles
       self.particles = [Particle(initial_fitness,no_classes) for _ in range(self.n_particles)]
       self.inertia = inertia
       self.cognitive_constant = cognitive_constant
       self.social_constant = social_constant
       self.max_iter = max_iter
       self.max_inertia = max_inertia
       self.min_inertia = min_inertia
       self.best_particle = None
       self.x,self.y = x,y
       print(f'Initial best group fitness: {self.best_group_fitness}')
                
    def update_inertia(self,t):
        self.inertia = self.max_inertia - (self.max_inertia - self.min_inertia) * t / self.max_iter
    
    def fit(self):
        best_positions = []
        for t in tqdm(range(self.max_iter)):
            print(f'Iteration {t+1} with best position {best_positions[-1][0] if len(best_positions) > 0 else self.best_group_fitness}')
            for particle in self.particles:
                particle.evaluate(self.x,self.y)
                # update fitness and position of the swarm
                if particle.fitness_particle_position < self.best_group_fitness or self.best_group_fitness == float('inf'):
                    self.best_group_fitness = particle.fitness_particle_position
                    self.best_group_position = particle.position
                    self.best_particle = particle
            
            for particle in self.particles:
                particle.update_velocity(self.inertia,self.cognitive_constant,self.social_constant,self.best_group_position)
                particle.update_position()
            self.update_inertia(t)
            
            best_positions.append((self.best_group_fitness,self.best_particle.position))
            
            no_improvement = 0
            early_stop = False
            for i in range(len(best_positions)-1,1,-1):
                if no_improvement == int(10e5):
                    early_stop = True
                if best_positions[i][0] == best_positions[i-1][0]:
                    no_improvement += 1
                else:
                    break
                
            if early_stop:
                return best_positions
            
    
        return best_positions


def get_data(train_dataset_name,val_dataset_name,test_dataset_name):
    df_train = pd.read_csv(train_dataset_name)
    df_val = pd.read_csv(val_dataset_name)
    df_test = pd.read_csv(test_dataset_name)
    return df_train,df_val,df_test


def split_x_y(df,target_variable):
    x = df.drop(columns=[target_variable])
    y = df[target_variable]
    return x,y
    
    


def get_solution(solution_positions,x_test,y_test):
    rnn = RNN()

    params_for_W_hx = solution_positions[:rnn.W_hx.shape[0]*rnn.W_hx.shape[1]]
    rnn.W_hx = params_for_W_hx.reshape(rnn.W_hx.shape[0],rnn.W_hx.shape[1])
    params_for_W_hh = solution_positions[rnn.W_hx.shape[0]*rnn.W_hx.shape[1]:\
                                        rnn.W_hx.shape[0]*rnn.W_hx.shape[1]\
                                        + rnn.W_hh.shape[0]*rnn.W_hh.shape[1]]
    rnn.W_hh = params_for_W_hh.reshape(rnn.W_hh.shape[0],rnn.W_hh.shape[1])
    params_for_W_hy = solution_positions[rnn.W_hx.shape[0]*rnn.W_hx.shape[1]\
                                        + rnn.W_hh.shape[0]*rnn.W_hh.shape[1]:]
    rnn.W_hy = params_for_W_hy.reshape(rnn.W_hy.shape[0],rnn.W_hy.shape[1])        

    
    return rnn

def rnn_predict(x,rnn): 
    y_pred = []
    window_size = 30
    y_test = y_test[window_size-1:]
    for i in range(window_size-1,x.shape[0],1):
        pred = rnn.forward(x[i-window_size+1:i+1])
        y_pred.append(pred)
    y_pred = np.array(y_pred)
    mae = metrics.mean_absolute_error(y_test,y_pred)
    return mae

def get_args():
    args = parser.parse_args()
    return args.n_particles,args.inertia,args.cognitive_constant,args.social_constant,args.max_iter,args.max_inertia,args.min_inertia,args.mode,args.dataset_train,args.dataset_val,args.dataset_test,args.target_variable,args.no_classes


def main():
    n_particles, inertia, cognitive_constant, social_constant, max_iter, max_inertia, min_inertia,mode,train_dataset_name,val_dataset_name,test_dataset_name,target_variable,no_classes = get_args()
    
    rnn = RNN()
    
    df_train,df_val,df_test = get_data(train_dataset_name,val_dataset_name,test_dataset_name)
    x_train,y_train = split_x_y(df_train,target_variable)
    x_val,y_val = split_x_y(df_val,target_variable)
    x_test,y_test = split_x_y(df_test,target_variable)
    
    
    if mode == 'max':
        initial_fitness = -float('inf')
    else:
        initial_fitness = float('inf')
        
    pso = PSO(n_particles,inertia,cognitive_constant,social_constant,rnn,initial_fitness,max_iter,max_inertia,min_inertia,x_train,y_train,no_classes)
    
    x_test = pd.concat([x_val,x_test])
    y_test = pd.concat([y_val,y_test])
    
    best_positions = pso.fit()
    trained_model = get_solution(best_positions[-1][1],x_test,y_test)
    
    with open('rnn_performance.txt','a') as f:
        f.write(f'PSO parameters:\n')
        f.write(f'Number of particles: {n_particles}, cognitive constant: {cognitive_constant}, social constant: {social_constant}, max iterations: {max_iter}, max inertia: {max_inertia}, min inertia: {min_inertia}\n') 
        mae_test = rnn_predict(x_test,trained_model)
        f.write(f'Mean absolute error on test set: {mae_test}\n')
        mae_train = rnn_predict(x_train,trained_model)
        f.write(f'Mean absolute error on train set: {mae_train}\n')
        f.write('-----------------------------------------------------------\n')

if __name__ == '__main__':
    main()