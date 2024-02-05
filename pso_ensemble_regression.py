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
warnings.filterwarnings('ignore')


logging.basicConfig(filename='pso.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser()
parser.add_argument('--n_particles', type=int, default=30)
parser.add_argument('--inertia', type=float, default=0.9)
parser.add_argument('--cognitive_constant', type=float, default=2.05)
parser.add_argument('--social_constant', type=float, default=1.4945)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--max_inertia', type=float, default=0.9)
parser.add_argument('--min_inertia', type=float, default=0.4)
parser.add_argument('--mode', type=str, default='min')
parser.add_argument('--dataset-train', type=str, default='airline_train.csv')
parser.add_argument('--dataset-val', type=str, default='airline_val.csv')
parser.add_argument('--dataset-test', type=str, default='airline_test.csv')
parser.add_argument('--target-variable', type=str, default='price')
parser.add_argument('--no-classes', type=int, default=2)


def softmax(x):
    cp = copy.deepcopy(x)
    from sklearn.preprocessing import normalize
    cp = normalize([cp],norm='l1')[0]
    return np.exp(cp) / (np.sum(np.exp(cp),axis=0) + 1e-6)

def encode_one_hot(x,no_classes):
    encoded = np.zeros((x.shape[0],no_classes),dtype=np.float64)
    for i in range(x.shape[0]):
        encoded[i][x[i]] = 1
    return encoded


class Particle:
    def __init__(self,clfs,initial_fitness,no_classes) -> None:
        self.dimension = len(clfs)
        self.classifiers = clfs
        self.position = np.random.uniform(low=0, high=1,size=(self.dimension,))
        self.velocity = np.random.uniform(low=-1, high=1,size=(self.dimension,))
        self.best_individual_position = self.position
        self.fitness_particle_position = initial_fitness
        self.fitness_best_individual_position = initial_fitness
        self.no_classes = no_classes
    
    def objective_function(self,x,y):
        normalized_position = softmax(self.position)
        ensemble_y_pred = np.zeros_like(y,dtype=np.float64)
        
        for i,classifier in enumerate(self.classifiers):
            y_pred = classifier.predict(x)
            ensemble_y_pred += np.round(normalized_position[i],6) * np.round(y_pred,6)
            
        try:
            mse = metrics.mean_squared_error(y,np.round(ensemble_y_pred,6))
        except ValueError as e:
            print(e)
            
        return mse
        
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
    def __init__(self,n_particles,inertia,cognitive_constant,social_constant,classifiers,initial_fitness,max_iter,max_inertia,min_inertia,x,y,no_classes) -> None:
       self.best_group_position = np.random.uniform(low=0, high=1,size=(len(classifiers),))
       self.best_group_fitness = initial_fitness
       self.n_particles = n_particles
       self.particles = [Particle(classifiers,initial_fitness,no_classes) for _ in range(self.n_particles)]
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
                if no_improvement == 20:
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
    
    
HYPERPARAMETERS = {
   'criterion': ['gini', 'entropy','log_loss'],
   'splitter': ['best', 'random'],
   'max_depth': [None, 2, 4, 6, 8, 10,15,20,25,50],
   'min_samples_split': [2, 5, 10,12],
   'min_samples_leaf': [1, 2, 4,5],
   'max_features': ['auto', 'sqrt', 'log2', None],
   'random_state': [42],
   'max_leaf_nodes': [None, 2, 4, 6, 8, 10],
   'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
   'class_weight': ['balanced', None],
   'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
   
}


def train_base_classifiers(x_train,y_train,x_val,y_val,x_test,y_test,n_classifiers=3):
    classifiers = []
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import AdaBoostRegressor
    xgb_reg = xgb.XGBRegressor()
    lgb_reg = lgb.LGBMRegressor(verbose=-1)
    cb_reg = cb.CatBoostRegressor()
    rf_ref = RandomForestRegressor()
    ridge = Ridge()
    ada = AdaBoostRegressor()
    
    print('xgb')
    xgb_reg.fit(x_train,y_train)
    print(metrics.mean_squared_error(y_val,xgb_reg.predict(x_val)))
    print(metrics.mean_squared_error(y_test,xgb_reg.predict(x_test)))
    
    print('lgb')
    lgb_reg.fit(x_train,y_train)
    print(metrics.mean_squared_error(y_val,lgb_reg.predict(x_val)))
    print(metrics.mean_squared_error(y_test,lgb_reg.predict(x_test)))
    print('cb')
    cb_reg.fit(x_train,y_train,logging_level='Silent')
    print(metrics.mean_squared_error(y_val,cb_reg.predict(x_val)))
    print(metrics.mean_squared_error(y_test,cb_reg.predict(x_test)))
    print('rf')
    rf_ref.fit(x_train,y_train)
    print(metrics.mean_squared_error(y_val,rf_ref.predict(x_val)))
    print(metrics.mean_squared_error(y_test,rf_ref.predict(x_test)))
    print('ridge')
    ridge.fit(x_train,y_train)
    print(metrics.mean_squared_error(y_val,ridge.predict(x_val)))
    print(metrics.mean_squared_error(y_test,ridge.predict(x_test)))
    print('ada')
    ada.fit(x_train,y_train)
    print(metrics.mean_squared_error(y_val,ada.predict(x_val)))
    print(metrics.mean_squared_error(y_test,ada.predict(x_test)))
    
    classifiers.append(xgb_reg)
    classifiers.append(lgb_reg)
    classifiers.append(cb_reg)
    classifiers.append(rf_ref)
    classifiers.append(ridge)
    classifiers.append(ada)
        
          
    return classifiers


def predict_unweighted_ensemble(classifiers,x,y,no_classes):
    ensemble_y_pred = np.zeros_like(y,dtype=np.float64)
    
    weight = 1 / len(classifiers)
    for i,classifier in enumerate(classifiers):
        y_pred = classifier.predict(x)
        ensemble_y_pred += weight * y_pred
        
    return metrics.mean_squared_error(y,ensemble_y_pred)

def test_swarm(particle,x,y,classifiers):
    ensemble_y_pred = np.zeros_like(y,dtype=np.float64)
    
    for i,classifier in enumerate(classifiers):
        y_pred = classifier.predict(x)
        ensemble_y_pred += particle[i] * y_pred
        
    return metrics.mean_squared_error(y,ensemble_y_pred)

def test_models(classifiers,x,y):
    with open('regression_performance.txt','a') as f:
        for classifier in classifiers:
            f.write(f'MSE for {classifier.__class__.__name__} on test data: {metrics.mean_squared_error(y,classifier.predict(x))}\n')
            
def get_args():
    args = parser.parse_args()
    return args.n_particles,args.inertia,args.cognitive_constant,args.social_constant,args.max_iter,args.max_inertia,args.min_inertia,args.mode,args.dataset_train,args.dataset_val,args.dataset_test,args.target_variable,args.no_classes


def main():
    n_particles, inertia, cognitive_constant, social_constant, max_iter, max_inertia, min_inertia,mode,train_dataset_name,val_dataset_name,test_dataset_name,target_variable,no_classes = get_args()
    
    df_train,df_val,df_test = get_data(train_dataset_name,val_dataset_name,test_dataset_name)
    x_train,y_train = split_x_y(df_train,target_variable)
    x_val,y_val = split_x_y(df_val,target_variable)
    x_test,y_test = split_x_y(df_test,target_variable)
    
    classifiers = train_base_classifiers(x_train,y_train,x_val,y_val,x_test,y_test)
    
    if mode == 'max':
        initial_fitness = -float('inf')
    else:
        initial_fitness = float('inf')
    pso = PSO(n_particles,inertia,cognitive_constant,social_constant,classifiers,initial_fitness,max_iter,max_inertia,min_inertia,x_val,y_val,no_classes)
    
    best_positions = pso.fit()
    with open('regression_performance.txt','a') as f:
        f.write(f'PSO Paramaters:\n')
        f.write(f'Number of particles: {n_particles}, Cognitive Constant: {cognitive_constant}, Social Constant: {social_constant}, Max Iterations: {max_iter}, Max Inertia: {max_inertia}, Min Inertia: {min_inertia}\n')
        f.write(f'Maximum objective function for validation data: {best_positions[-1][0]}\n')
        best_unweighted = predict_unweighted_ensemble(classifiers,x_test,y_test,no_classes)
        test_models(classifiers,x_test,y_test)
        f.write(f'MSE for unweighted ensemble on test data: {best_unweighted}\n')
        f.write(f'MSE for swarm on test data: {test_swarm(softmax(best_positions[-1][1]),x_test,y_test,classifiers)}\n')
        f.write(f'Models weights: {np.round(softmax(best_positions[-1][1]),3)}\n')
        f.write('\n')
    

if __name__ == '__main__':
    main()