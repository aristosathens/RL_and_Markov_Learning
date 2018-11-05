# import os
# import math
# import itertools
import csv

import pandas as pd
import numpy as np
# import scipy as sp
# import networkx as nx
# import matplotlib.pyplot as plt

from enum import Enum
# from collections import defaultdict
# from networkx.drawing.nx_agraph import to_agraph


'''
    Classes
'''

class MarkovLearner():
    '''
        Parent class for Markov Learner classes.
    '''

    def __init__(self, file_name, learning_rate = 1, discount_factor=0.95, num_s=None, num_a=None):
        '''
            Init MarkovLearner object
        '''

        self.load_data(file_name)

        if num_s == None:
            self.num_states = np.max(self.data[:, 0]) + 1
        else:
            self.num_states = num_s

        if num_a == None:
            self.num_actions = np.max(self.data[:, 1]) + 1
        else:
            self.num_actions = num_a

        self.alpha = learning_rate
        self.gamma = discount_factor

        self.child_init()






        """
        # Datasets
        # small: Normal, model-based/model-free RL. Just implement what is in the book
            
            # note: If we knew R & T, then do Value Iteration (VI), that is NOT reinforcement learning
                    # this is planning or solving an MDP

        # medium: Exploit Structure, Function Approximation 
        # large: Exploit structure (might not just want to do Function Approximation here)g

        """

    def child_init(self):
        raise Exception("MarkovLearner does not implement child_init(). Child class must implement it.")

    def learn(self):
        raise Exception("MarkovLearner does not implement learn(). Child class must implement it.")

   

    def load_data(self, file_name):
        ''' 
            Read csv into matrix
        '''

        self.input_file = file_name
        try:
            # Get column names
            with open(file_name, newline='') as f:
                reader = csv.reader(f)
                self.data_names = next(reader)
            # Get numerical data
            self.data = np.asarray(pd.read_csv(file_name))[1:,:] #- np.array([1, 1, 0, 1])
        except:
            raise Exception("Could not load csv with name: " + fileName + ".")


    def write_policy_to_file(self, file_name = None):
        '''
            Writes policy to a txt file
        '''
        if file_name is not None:
            np.savetxt(file_name, self.Pi, fmt='%d')
        else:
            try:
                output_file = self.input_file[:self.input_file.find('.')] + ".policy"
                np.savetxt(output_file, self.Pi, fmt='%d')
            except:
                raise Exception("Can't form output file name (input file name missing period)")


    def init_N(self):
        '''
            Initialize N matrix
        '''
        self.N = np.zeros((self.num_states, self.num_actions, self.num_states))

        for row in self.data:
            self.N[row[0], row[1], row[3]] += 1


    def init_R(self):
        '''
            Initialize R matrix
        '''
        self.R = np.zeros((self.num_states, self.num_actions))
        temp = [[[] for i in range(self.num_actions)] for j in range(self.num_states)]

        for row in self.data:
            temp[row[0]][row[1]].append(row[2])

        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.R[s, a] = np.mean(temp[s][a])


    def init_T(self):
        '''
            Initialize T matrix
        '''
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                s_a_sum = np.sum(self.N[s, a, :])
                for sp in range(self.num_states):
                    self.T[s, a, sp] = self.N[s, a, sp] / s_a_sum





    def generate_Pi(self):
        '''
            Generate policy from Q
        '''
        self.Pi = np.zeros(self.num_states)
        for i, row in enumerate(self.Q):
            self.Pi[i] = np.argmax(row) + 1


    def dyna_learn(self):
        '''
            Use dyna algorithm to learn Q from R and T
        '''

        for s in range(num_states):
            for a in range(num_actions):
                self.Q[s, a] = R[s, a] + self.gamma * np.sum(self.T[s, a, :] * np.max(self.Q, axis=1))


    def priority_sweep(self, epsilon = 1e-2):
        '''
            Use priority sweeping to learn U
        '''

        self.U = np.ones(self.num_states)
        queue = {s : 0 for s in range(self.num_states)}

        for s in range(self.num_states):
            queue[s] = 1e25

            while len(queue) > 0:
                s = max(queue.items(), key=lambda pair: pair[1])[0]
                queue.pop(s)

                u = self.U[s]

                # the_max = 0
                # for a in range(self.num_actions):
                #     val = self.R[s, a] + self.gamma * (self.T[s, a, :].dot(self.U))
                #     if val > the_max:
                #         self.U[s] = a
                #         the_max = val
                #     print(self.U[s] == np.argmax(self.R[s, :] + self.gamma * (self.T[s, :, :].dot(self.U))))

                self.U[s] = np.argmax(self.R[s, :] + self.gamma * (self.T[s, :, :].dot(self.U)))

                for sp in range(self.num_states):
                    for a in range(self.num_actions):

                        if self.T[sp, a, s] == 0 or s == sp:
                            continue

                        p = abs(self.T[sp, a, s] * (self.U[s] - u))
                        if p > epsilon:
                            # if sp not in queue or p > queue[sp]:
                            if sp in queue and p > queue[sp]:
                                queue[sp] = p
                        elif sp in queue:
                            queue.pop(sp)



    







class QLearner(MarkovLearner):
    '''
        Q Learner
    '''
    def __init__(self,
                    file_name,
                    num_s=None,
                    num_a=None,
                    learning_rate = 1,
                    discount_factor=0.95,
                    interpolation_window=5,
                    weighted_inerpolation=False,
                    ):
        '''
            Init QLearner object
        '''

        self.load_data(file_name)

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.interpolation_range = interpolation_window
        self.weight_inerpolation = weighted_inerpolation

        self.state_indices = None
        self.action_indices = None

        if num_s != None:
            self.num_states = num_s
            self.data[:, 0] -= 1
            self.data[:, 3] -= 1
        else:
            self.assign_state_indices()

        if num_a != None:
            self.num_actions = num_a
            self.data[:, 1] -= 1 #np.array([0, 1, 0, 0])
        else:
            self.assign_action_indices()

        self.init_Q()


    def init_Q(self):
        '''
            Initialize Q matrix.
            Q is size (num_states x num_actions).
        '''
        self.Q = np.zeros((self.num_states, self.num_actions))


    def assign_state_indices(self):
        '''
            Assign an index for each state, action. Do this for datasets where values don't map well to indices.
        '''
        state_index = 0
        self.state_indices = {}

        for row in self.data:

            if row[0] not in self.state_indices:
                self.state_indices[row[0]] = state_index
                state_index += 1
            if row[3] not in self.state_indices:
                self.state_indices[row[3]] = state_index
                state_index += 1

        self.num_states = len(self.state_indices)
        self.reverse_state_indices = {v : k for (k, v) in self.state_indices.items()}


    def assign_action_indices(self):
        '''
            Assign an index for each state, action. Do this for datasets where values don't map well to indices
        '''
        action_index = 0
        self.action_indices = {}

        for row in self.data:
            if row[1] not in self.action_indices:
                self.action_indices[row[1]] = action_index
                action_index += 1

        self.num_actions = len(self.action_indices)
        self.reverse_action_indices = {v : k for (k, v) in self.action_indices.items()}

        print(self.action_indices)


    def state_index(self, s):
        if self.state_indices is None:
            return s
        else:
            return self.state_indices[s]

    def action_index(self, a):
        if self.action_indices is None:
            return a
        else:
            return self.action_indices[a]



    def learn(self):
        self.Q_learn()
        self.generate_Pi()


    def generate_Pi(self):
        '''
            Generate policy Pi from action value matrix Q.
            Pi will have shape num_states x 1
        '''
        self.Pi = np.zeros(self.num_states)

        for i, row in enumerate(self.Q):

            if self.action_indices is not None:
                action_index = np.argmax(row)
                self.Pi[i] = self.reverse_action_indices[action_index]

            else:
                self.Pi[i] = np.argmax(row) + 1

        print(self.Pi)


    def Q_learn(self):
        '''
            Learn Q from data
        '''

        for row in self.data:
            s = self.state_index(row[0])
            sp = self.state_index(row[3])
            a = self.action_index(row[1])
            r = row[2]

            self.Q[s, a] +=  self.alpha * (r + self.gamma * (np.max(self.Q[sp, :]) - self.Q[s, a]))

        # Do some interpolation of rewards for rows with all-zeros
        if self.weight_inerpolation:
            self.weighted_linear_Q_interpolation()
        else:
            self.unweighted_linear_Q_interpolation()


    def unweighted_linear_Q_interpolation(self):
        '''
            Linearly interpolate empty Q rows by taking mean of neighboring Q rows
        '''
        for i, row in enumerate(self.Q):
            if not row.any():
                i_min = np.max([0, i - self.interpolation_range])
                i_max = np.min([self.num_states - 1, i + self.interpolation_range])
                self.Q[i, :] = np.mean(self.Q[i_min:i_max, :], axis=0)

    def weighted_linear_Q_interpolation(self):
        '''
            Linearly interpolate empty Q rows by taking weighted mean of neighboring Q rows
        '''
        for i, row in enumerate(self.Q):
            if not row.any():
                i_min = np.max([0, i - self.interpolation_range])
                i_max = np.min([self.num_states - 1, i + self.interpolation_range])

                weights = np.linspace(0, 1, i - i_min, endpoint = False)
                weights = np.append(weights, np.flip(np.linspace(0, 1, i_max - i, endpoint = False), axis = 0))

                weights /= np.sum(weights)  # normalize
                weights = np.expand_dims(weights, axis= 1)

                self.Q[i, :] = np.sum(weights * self.Q[i_min:i_max, :], axis = 0)















def main():
    '''
        Construct MarkovLearner Objects and learn best policy for each data set
    '''

    # Small.csv
    print("Finding policy for small.csv...")
    learner = QLearner("small.csv",
                        num_s = 100,
                        num_a = 4)
    learner.learn()
    learner.write_policy_to_file()

    # Medium.csv
    print("Finding policy for medium.csv...")
    learner = QLearner("medium.csv",
                        num_s = 50000,
                        num_a = 7,
                        discount_factor = 1.0,
                        interpolation_window = 3)
    learner.learn()
    learner.write_policy_to_file()

    # Large.csv
    print("Finding policy for large.csv...")
    learner = QLearner("large.csv",
                        num_s = 312020,
                        num_a = 9,
                        interpolation_window = 41)
    learner.learn()
    learner.write_policy_to_file()


if __name__ == "__main__":
    main()