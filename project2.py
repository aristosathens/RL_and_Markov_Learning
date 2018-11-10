import csv
import datetime

import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize

from enum import Enum


'''
    Classes
'''

class MarkovLearner():
    '''
        Parent class for Markov Learner classes.
    '''

    def __init__(self,
                    file_name,
                    num_s=None,
                    num_a=None,
                    learning_rate = 1,
                    discount_factor=0.95,
                    interpolation_window=5,
                    weighted_interpolation=False,
                    ):
        '''
            Init QLearner object
        '''

        self.load_data(file_name)

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.interpolation_range = interpolation_window
        self.weight_interpolation = weighted_interpolation

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
            self.data[:, 1] -= 1
        else:
            self.assign_action_indices()

        self.init()


    def init(self):
        raise Exception("Child class must implement its own init() function (MarkovLearner implements __init__()).")

    def learn(self):
        raise Exception("MarkovLearner does not implement learn(). Child class must implement it.")


    def init_Q(self):
        '''
            Initialize Q matrix.
            Q is size (num_states x num_actions).
        '''
        self.Q = np.zeros((self.num_states, self.num_actions))


    def init_U(self):
        '''
            Initialize U vector.
            U iis size (num_states).
        '''
        self.U = np.zeros(self.num_states)


    def init_N(self):
        '''
            Initialize N matrix
        '''
        self.N = np.zeros((self.num_states, self.num_actions, self.num_states))


    def count_N(self):
        '''
            Find counts for N matrix from data.
        '''
        for row in self.data:
            s, a, sp = self.get_indices_from_row(row)
            self.N[s, a, sp] += 1
   

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


    def get_state_index(self, s):
        if self.state_indices is None:
            return s
        elif np.size(s) == 1:
            return self.state_indices[s]
        else:
            return [self.state_indices[state] for state in s]

    def get_action_index(self, a):
        if self.action_indices is None:
            return a
        elif np.size(a) == 1:
            return self.action_indices[a]
        else:
            return [self.action_indices[action] for action in a]

    def get_state_from_index(self, i):
        if self.reverse_state_indices is None:
            raise("Reverse action indices no implemented. Can't call get_state_index(). You *can* still use get_state_index().")
        elif np.size(i) == 1:
            return self.reverse_state_indices[i]
        else:
            return [self.reverse_state_indices[index] for index in i]

    def get_action_from_index(self, i):
        if self.reverse_action_indices is None:
            raise("Reverse action indices no implemented. Can't call get_action_index(). You *can* still use get_action_index().")
        elif np.size(i) == 1:
            return self.reverse_action_indices[i]
        else:
            return [self.reverse_action_indices[index] for index in i]

    def get_indices_from_row(self, row):
        '''
            Given a row from a dataset, returns the indices for s, a, sp by looking at row[0], row[1], row[3]
        '''
        s = self.get_state_index(row[0])
        a = self.get_action_index(row[1])
        sp = self.get_state_index(row[3])
        return (s, a, sp)


    def generate_Pi(self):
        '''
            Generate policy Pi from action value matrix Q.
            Pi will have shape num_states x 1
        '''

        self.Pi = np.zeros(self.num_states)

        for i, row in enumerate(self.Q):

            if self.action_indices is not None:
                action_index = np.argmax(row)
                self.Pi[i] = self.get_action_from_index(action_index) #self.reverse_action_indices[action_index]

            else:
                self.Pi[i] = np.argmax(row) + 1

        print(self.Pi)


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




# -------------------------------------------- Sub Classes -------------------------------------------- #

class QLearner(MarkovLearner):
    '''
        Q Learner. Uses model-free Q learning to determine Q, then determine policy Pi from Q.
    '''

    def init(self):
        '''
            Init QLearner object. Called in MarkovLearner.__init__()
        '''
        self.init_Q()

    def learn(self):
        self.Q_learn()
        self.generate_Pi()

    def Q_learn(self):
        '''
            Learn Q from data
        '''
        for row in self.data:
            s, a, sp = self.get_indices_from_row(row)
            r = row[2]

            self.Q[s, a] +=  self.alpha * (r + self.gamma * (np.max(self.Q[sp, :]) - self.Q[s, a]))

        # Do some interpolation of rewards for rows with all-zeros
        if self.interpolation_range == None or self.interpolation_range == -1:
            return
        if self.weight_interpolation:
            self.weighted_linear_Q_interpolation()
        else:
            self.unweighted_linear_Q_interpolation()



class GaussSeidelLearner(MarkovLearner):
    '''
        Gauss-Seidel Learner. Uses Gauss-Seidel learning to determine Q, then determine policy Pi from Q.
    '''

    def init(self):
        '''
            Init GaussSeidelLearner object. Called in MarkovLearner.__init__()
        '''
        self.init_U()
        self.init_N()
        self.count_N()
        self.init_T()
        self.init_R()

    def learn(self):
        self.gauss_learn()

    def init_R(self):
        '''
            Initialize R matrix
        '''
        self.R = np.zeros((self.num_states, self.num_actions))
        temp = [[[] for i in range(self.num_actions)] for j in range(self.num_states)]

        for row in self.data:
            s, a, sp = self.get_indices_from_row(row)
            temp[s][a].append(row[2])

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
                    if s_a_sum == 0:
                        self.T[s, a, sp] = 0
                    else:
                        self.T[s, a, sp] = self.N[s, a, sp] / s_a_sum


    def gauss_learn(self):
        '''
            Learn Q from data
        '''
        self.Pi = np.zeros(self.num_states)

        for row in self.data:
            s, a, sp = self.get_indices_from_row(row)
            r = row[2]

            self.U[s] = np.max(self.R[s, :] + self.gamma * (self.T[s, :, :].dot(self.U)))
            self.Pi[s] = np.argmax(self.R[s, :] + self.gamma * (self.T[s, :, :].dot(self.U))) + 1

        print(self.Pi)


class SarsaLearner(MarkovLearner):
    '''
        Sarsa Learner. Uses model-free Sarsa learning to determine Q, then determine policy Pi from Q.
    '''

    def init(self):
        '''
            Init SarsaLearner object. Called in MarkovLearner.__init__()
        '''
        self.init_Q()

    def learn(self):
        '''
            Learn parameters and generate policy.
        '''
        self.sarsa_learn()
        self.generate_Pi()

    def sarsa_learn(self):
        '''
            Learn Q from data
        '''
        s_p = None; a_p = None; r_p = None; sp_p = None
        for row in self.data:
            s, a, sp = self.get_indices_from_row(row)
            r = row[2]

            if s_p is not None:
                self.Q[s_p, a_p] +=  self.alpha * (r_p + self.gamma * (self.Q[sp, a] - self.Q[s_p, a_p]))
            s_p = s
            a_p = a
            r_p = r
            sp_p = sp

        # Do some interpolation of rewards for rows with all-zeros
        if self.weight_inerpolation:
            self.weighted_linear_Q_interpolation()
        else:
            self.unweighted_linear_Q_interpolation()



class SarsaLambdaLearner(MarkovLearner):
    '''
        Sarsa Learner. Uses model-free Sarsa learning to determine Q, then determine policy Pi from Q.
    '''

    def init(self):
        self.init_N()
        self.init_Q()

    def init_N(self):
        '''
            Initialize N matrix. N has shape (num_states x num_actions).
        '''
        self.N = np.zeros((self.num_states, self.num_actions))

    def learn(self, Lambda=0.9):
        '''
            Learn parameters and generate policy.
        '''
        self.Lambda = Lambda
        self.sarsa_lambda_learn()
        self.generate_Pi()

    def sarsa_lambda_learn(self):
        '''
            Use eligibility tracing and sarsa to learn Q.
        '''
        len_data = self.data.shape[0]
        for i, row in enumerate(self.data):
            if i == len_data - 1:
                break
            s, a, sp = self.get_indices_from_row(row)
            r = row[2]

            ap = self.get_action_index(self.data[i + 1, 1])
            self.N[s, a] += 1
            d = r + self.gamma * self.Q[sp, ap] - self.Q[s, a]

            self.Q += self.alpha * d * self.N
            self.N *= self.gamma * self.Lambda


class ModelLearner(MarkovLearner):
    '''
        Model Learner
    '''

    def init(self):
        self.fit_next_state_predictor()
        self.init_N()
        self.init_R()
        self.init_Q()

    def learn(self):
        self.dyna_learn()
        self.generate_Pi()

    def init_R(self):
        '''
            Initialize R matrix
        '''
        self.R = np.zeros((self.num_states, self.num_actions))
        temp = [[[] for i in range(self.num_actions)] for j in range(self.num_states)]

        for row in self.data:
            s, a, sp = self.get_indices_from_row(row)
            temp[s][a].append(row[2])

        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.R[s, a] = np.mean(temp[s][a])

    def next_state(self, inputs, p0=None, p1=None, p2=None, p3=None, p4=None, p5=None, p6=None, p7=None, p8=None):
        '''
            Get's the expected next state from the current state.
            If no arguments passed in (p1 is none), call next_state recursively using self.next_state_coefficients.
            Self.next_state_coefficients are learned once with a call to self.fit_next_state_predictor().
        '''
        if p0 is None:
            return int(round(self.next_state(inputs, *self.next_state_coefficients)))
        else:
            s = inputs[0]
            a = inputs[1]
            return p0 + p1*(s) + p2*(a) + p3*(s*s) + p4*(a*a) + p5*(s*a) + p6*(s*s*a) + p7*(s*a*a) + p8*(s*s*a*a)

    def fit_next_state_predictor(self):
        '''
            Uses scipy.optimize_curve_fit() to fit parameters of self.next_state() function
        '''
        all_s = self.get_state_index(self.data[:, 0])
        all_a = self.get_action_index(self.data[:, 1])
        outputs = self.data[:, 3]
        self.next_state_coefficients, _ = sp.optimize.curve_fit(self.next_state, (all_s, all_a), outputs, p0 = np.zeros(9))
        print(self.next_state_coefficients)

    def get_T(self, s, a, sp):
        '''
            Method for getting T value on demand, instead of storing entire T matrix
        '''
        s_a_sum = np.sum(self.N[s, a, :])
        if sp == "all":
            retVec = np.zeros(self.num_states)
            for spp in range(self.num_states):
                retVec[spp] = self.N[s, a, spp] / s_a_sum
            return retVec
        else:
            return self.N[s, a, sp] / s_a_sum

    def dyna_learn(self):
        '''
            Use dyna algorithm to learn Q from R and T
        '''
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.Q[s, a] = self.R[s, a] + self.gamma * np.sum(self.get_T(s, a, "all") * np.max(self.Q, axis=1))






# -------------------------------------------- Main -------------------------------------------- #

def main():
    '''
        Construct MarkovLearner Objects and learn best policy for each data set
    '''




    # Small.csv
    print("Finding policy for small.csv...")
    print(datetime.datetime.now())
    learner = GaussSeidelLearner("small.csv",
                        num_s = 100,
                        num_a = 4,
                        interpolation_window = -1)
    learner.learn()
    learner.write_policy_to_file()

    # Medium.csv
    print("Finding policy for medium.csv...")
    print(datetime.datetime.now())
    learner = QLearner("medium.csv",
                        num_s = 50000,
                        num_a = 7,
                        discount_factor = 1.0,
                        interpolation_window = 3)
    learner.learn()
    learner.write_policy_to_file()

    # Large.csv
    print("Finding policy for large.csv...")
    print(datetime.datetime.now())
    learner = QLearner("large.csv",
                        num_s = 312020,
                        num_a = 9,
                        interpolation_window = 41)
    learner.learn()
    learner.write_policy_to_file()
    print(datetime.datetime.now())



if __name__ == "__main__":
    main()






# -------------------------------------------- Unused -------------------------------------------- #

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