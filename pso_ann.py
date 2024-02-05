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
from ann import ANN
warnings.filterwarnings('ignore')


logging.basicConfig(filename='pso.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser()
parser.add_argument('--n_particles', type=int, default=200)
parser.add_argument('--inertia', type=float, default=0.9)
parser.add_argument('--cognitive_constant', type=float, default=2.05)
parser.add_argument('--social_constant', type=float, default=2.05)
parser.add_argument('--max_iter', type=int, default=400)
parser.add_argument('--max_inertia', type=float, default=0.9)
parser.add_argument('--min_inertia', type=float, default=0.1)
parser.add_argument('--mode', type=str, default='max')
parser.add_argument('--dataset-train', type=str, default='diabetes_train.csv')
parser.add_argument('--dataset-val', type=str, default='diabetes_val.csv')
parser.add_argument('--dataset-test', type=str, default='diabetes_test.csv')
parser.add_argument('--target-variable', type=str, default='Outcome')
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
        self.ann = ANN()
        self.dimension = self.ann.input_size*self.ann.hidden_size + self.ann.hidden_size*self.ann.hidden_size2 + self.ann.hidden_size2*self.ann.output_size
        self.position = np.concatenate((self.ann.W1.reshape(1,-1).ravel(),self.ann.W2.reshape(1,-1).ravel(),self.ann.W3.reshape(1,-1).ravel()))
        self.velocity = np.random.uniform(low=-2, high=2,size=(self.dimension,))
        self.best_individual_position = self.position
        self.fitness_particle_position = initial_fitness
        self.fitness_best_individual_position = initial_fitness
        self.no_classes = no_classes
    
    def objective_function(self,x,y):
        self.copy_particle_positions_to_ann_weights()
        y_pred = []
        y_pred = np.apply_along_axis(self.ann.forward, axis=1, arr=x)
        accuracy = metrics.accuracy_score(y,y_pred)
        return accuracy
    
    def copy_particle_positions_to_ann_weights(self):
        params_for_W1 = self.position[:self.ann.input_size*self.ann.hidden_size] 
        self.ann.W1 = params_for_W1.reshape(self.ann.W1.shape[0],self.ann.W1.shape[1])
        params_for_W2 = self.position[self.ann.input_size*self.ann.hidden_size:
            self.ann.hidden_size*self.ann.hidden_size2 + self.ann.input_size*self.ann.hidden_size] 
        self.ann.W2 = params_for_W2.reshape(self.ann.W2.shape[0],self.ann.W2.shape[1])
        params_for_W3 = self.position[self.ann.hidden_size*self.ann.hidden_size2+self.ann.input_size*self.ann.hidden_size:] #self.ann.hidden_size2*self.ann.output_size
        self.ann.W3 = params_for_W3.reshape(self.ann.W3.shape[0],self.ann.W3.shape[1])
        
        
    def evaluate(self,x,y):
        self.fitness_particle_position = self.objective_function(x,y)
        if self.fitness_particle_position > self.fitness_best_individual_position or self.fitness_best_individual_position == -float('inf'):
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
    def __init__(self,n_particles,inertia,cognitive_constant,social_constant,ann,initial_fitness,max_iter,max_inertia,min_inertia,x,y,no_classes) -> None:
       self.best_group_position = np.concatenate((ann.W1.reshape(1,-1).ravel(),ann.W2.reshape(1,-1).ravel(),ann.W3.reshape(1,-1).ravel()))
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
                if particle.fitness_particle_position > self.best_group_fitness or self.best_group_fitness == -float('inf'):
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
    
    
def test_ann(x_test,y_test,ann):
    y_pred = []
    y_pred = np.apply_along_axis(ann.forward, axis=1, arr=x_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    return accuracy


def get_solution(solution_positions,x_test,y_test,x_train,y_train):
    ann = ANN()
    
    params_for_W1 = solution_positions[:ann.input_size*ann.hidden_size]
    ann.W1 = params_for_W1.reshape(ann.W1.shape[0],ann.W1.shape[1])
    params_for_W2 = solution_positions[ann.input_size*ann.hidden_size:ann.hidden_size*ann.hidden_size2+
                                       ann.input_size*ann.hidden_size]
    ann.W2 = params_for_W2.reshape(ann.W2.shape[0],ann.W2.shape[1])
    params_for_W3 = solution_positions[ann.hidden_size*ann.hidden_size2+ann.input_size*ann.hidden_size:]
    ann.W3 = params_for_W3.reshape(ann.W3.shape[0],ann.W3.shape[1])
    
    return ann


def ann_predict(x,ann):
    return np.apply_along_axis(ann.forward, axis=1, arr=x)


def get_args():
    args = parser.parse_args()
    return args.n_particles,args.inertia,args.cognitive_constant,args.social_constant,args.max_iter,args.max_inertia,args.min_inertia,args.mode,args.dataset_train,args.dataset_val,args.dataset_test,args.target_variable,args.no_classes



def main():
    n_particles, inertia, cognitive_constant, social_constant, max_iter, max_inertia, min_inertia,mode,train_dataset_name,val_dataset_name,test_dataset_name,target_variable,no_classes = get_args()
    
    ann = ANN()
    
    df_train,df_val,df_test = get_data(train_dataset_name,val_dataset_name,test_dataset_name)
    x_train,y_train = split_x_y(df_train,target_variable)
    x_val,y_val = split_x_y(df_val,target_variable)
    x_test,y_test = split_x_y(df_test,target_variable)
    
    
    if mode == 'max':
        initial_fitness = -float('inf')
    else:
        initial_fitness = float('inf')
        
    pso = PSO(n_particles,inertia,cognitive_constant,social_constant,ann,initial_fitness,max_iter,max_inertia,min_inertia,x_train,y_train,no_classes)
    
    x_test = pd.concat([x_val,x_test])
    y_test = pd.concat([y_val,y_test])
    best_positions = pso.fit()
    trained_model = get_solution(best_positions[-1][1],x_test,y_test,x_train,y_train)

    with open('ann_performance.txt','a') as f:
        f.write('PSO params:\n')
        f.write(f'n_particles: {n_particles}, cognitive_constant: {cognitive_constant}, social_constant: {social_constant}, max_iter: {max_iter}, max_inertia: {max_inertia}, min_inertia: {min_inertia}, mode: {mode}\n')
        y_pred_test = ann_predict(x_test,trained_model)
        y_pred_train = ann_predict(x_train,trained_model)
        f.write(f'Accuracy on test set: {metrics.accuracy_score(y_test,y_pred_test)}\n')
        f.write(f'Accuracy on train set: {metrics.accuracy_score(y_train,y_pred_train)}\n')
        f.write(f'Cofusion matrix on test set: {metrics.confusion_matrix(y_test,y_pred_test)}\n')
        f.write('---------------------------------------------------\n')
    
    

if __name__ == '__main__':
    main()