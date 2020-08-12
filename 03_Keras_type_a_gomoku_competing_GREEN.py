import sys
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque
import matplotlib.pyplot as plt
import copy
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

n_ticks = 13
size_chkr_brd = n_ticks*n_ticks

# get size of state and action from environment
state_size = size_chkr_brd
action_size = size_chkr_brd

file_name =  sys.argv[0][:-3]
model_path = "save_model/"
graph_path = "save_graph/"

if not os.path.isdir(model_path):
    os.mkdir(model_path)
if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN_p1:
    def __init__(self, state_size, action_size):
        
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # train time define
        self.training_time = 60*60
        
        # These are hyper parameters for the DQN_p1
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.0001
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 256, 256
        
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 32
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 200

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state, ep_step):
        # print("Episode step :",ep_step)
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate or ep_step == 0:
            # print("  BLK player-Random action selected!!")
            p1_action = np.random.randint(0,n_ticks*n_ticks)
            while abs(state[0][p1_action]) > 0:
                p1_action = np.random.randint(0,n_ticks*n_ticks)
            
        else:
            q_value = self.model.predict(state)
            
            nonzero_chkr_brd = state > 0
            q_value = ~nonzero_chkr_brd * q_value            p1_action = np.argmax(q_value[0])
        return p1_action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        tgt_q_value_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(tgt_q_value_next[i]))
                
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate -= self.epsilon_decay
            
# DQN agent_p2 for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQN_p2:
    def __init__(self, state_size, action_size):
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # these is hyper parameters for the Double DQN
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.0002
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 300, 300
        
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 32
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 200

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state, ep_step):
        # print("Episode step :",ep_step)
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate or ep_step == 0:
            # print("  WHT player-Random action selected!!")
            p1_action = np.random.randint(0,n_ticks*n_ticks)
            while abs(state[0][p1_action]) > 0:
                p1_action = np.random.randint(0,n_ticks*n_ticks)
            
        else:
            q_value = self.model.predict(state)
            
            nonzero_chkr_brd = state > 0
            q_value = ~nonzero_chkr_brd * q_value
            p1_action = np.argmax(q_value[0])
        return p1_action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        q_value_next     = self.model.predict(next_states)
        tgt_q_value_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(q_value_next[i])
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (tgt_q_value_next[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate -= self.epsilon_decay
        
        
def main():
    
    agent_p1 = DQN_p1(state_size, action_size)
    agent_p2 = DoubleDQN_p2(state_size, action_size)
    
    agent_p1.model.load_weights(model_path + "/model_p1.h5")
    agent_p2.model.load_weights(model_path + "/model_p2.h5")
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agent_p1.episode = 0
    ep_step = 0
    
    BLK_prev_reward = 0
    WHT_prev_reward = 0
    
    # while agent_p1.episode < 3:
    while time.time() - start_time < 110*60:
        
        done_p1 = False
        done_p2 = False
        ep_step = 0
        
        # reset environment
        checker_board = np.zeros(size_chkr_brd, dtype=int)
        plt_ck_brd = np.zeros((n_ticks,n_ticks))

        while not done_p1 and ep_step < size_chkr_brd:
        # while ep_step < size_chkr_brd:
            # print(plt_ck_brd)
            
            if len(agent_p1.memory) < agent_p1.size_replay_memory:
                agent_p1.progress = "Exploration"
            else :
                agent_p1.progress = "Training"

            p1_state = copy.deepcopy(checker_board)
            p1_state = np.reshape(p1_state, [1, state_size])
            p1_action = agent_p1.get_action(p1_state, ep_step)
            
            ep_step += 1
            
            # Calculate next_state, reward, done
            # next_state, reward, done, _ = env.step(action)
            checker_board[p1_action] = 2
            p1_next_state = copy.deepcopy(checker_board)
            p1_next_state = np.reshape(p1_next_state, [1, state_size])
            
            share_v,remainder_v = divmod(p1_action,n_ticks)
            plt_ck_brd[share_v][remainder_v] = 2
    
            # check the number of 5 stones
            BLK_stn_5 = 0
            BLK_patt_5 = np.array([2,2,2,2,2])
            BLK_stn_4 = 0
            BLK_patt_4 = np.array([2,2,2,2])
            BLK_stn_3 = 0
            BLK_patt_3 = np.array([2,2,2])

            for row_idx in range(n_ticks):
                arr_x = np.array(plt_ck_brd[row_idx])
                done_num_5 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_5)] for j in range(len(arr_x) - len(BLK_patt_5) + 1)]
                          if np.array_equal(k, BLK_patt_5))
                BLK_stn_5 += done_num_5

                done_num_4 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_4)] for j in range(len(arr_x) - len(BLK_patt_4) + 1)]
                          if np.array_equal(k, BLK_patt_4))
                BLK_stn_4 += done_num_4

                done_num_3 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_3)] for j in range(len(arr_x) - len(BLK_patt_3) + 1)]
                          if np.array_equal(k, BLK_patt_3))
                BLK_stn_3 += done_num_3

            tp_plt_ck_brd = np.transpose(plt_ck_brd)
            for col_idx in range(n_ticks):
                arr_x = np.array(tp_plt_ck_brd[col_idx])
                done_num_5 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_5)] for j in range(len(arr_x) - len(BLK_patt_5) + 1)]
                          if np.array_equal(k, BLK_patt_5))
                BLK_stn_5 += done_num_5 

                done_num_4 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_4)] for j in range(len(arr_x) - len(BLK_patt_4) + 1)]
                          if np.array_equal(k, BLK_patt_4))
                BLK_stn_4 += done_num_4

                done_num_3 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_3)] for j in range(len(arr_x) - len(BLK_patt_3) + 1)]
                          if np.array_equal(k, BLK_patt_3))
                BLK_stn_3 += done_num_3

            dia_idx = -8
            while dia_idx < 9:
                arr_x = np.diag(plt_ck_brd, k=dia_idx)
                done_num_5 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_5)] for j in range(len(arr_x) - len(BLK_patt_5) + 1)]
                          if np.array_equal(k, BLK_patt_5))
                BLK_stn_5 += done_num_5 

                done_num_4 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_4)] for j in range(len(arr_x) - len(BLK_patt_4) + 1)]
                          if np.array_equal(k, BLK_patt_4))
                BLK_stn_4 += done_num_4

                done_num_3 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_3)] for j in range(len(arr_x) - len(BLK_patt_3) + 1)]
                          if np.array_equal(k, BLK_patt_3))
                BLK_stn_3 += done_num_3

                dia_idx += 1

            flip_dia_idx = -8
            while flip_dia_idx < 9:
                arr_x = np.fliplr(plt_ck_brd).diagonal(flip_dia_idx)
                done_num_5 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_5)] for j in range(len(arr_x) - len(BLK_patt_5) + 1)]
                          if np.array_equal(k, BLK_patt_5))
                BLK_stn_5 += done_num_5 

                done_num_4 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_4)] for j in range(len(arr_x) - len(BLK_patt_4) + 1)]
                          if np.array_equal(k, BLK_patt_4))
                BLK_stn_4 += done_num_4

                done_num_3 = sum(1 for k in 
                          [arr_x[j:j+len(BLK_patt_3)] for j in range(len(arr_x) - len(BLK_patt_3) + 1)]
                          if np.array_equal(k, BLK_patt_3))
                BLK_stn_3 += done_num_3

                flip_dia_idx += 1

            if BLK_stn_5 > 0 or ep_step == size_chkr_brd:
                done_p1 = True

            BLK_ttl_reward = BLK_stn_3*1 + BLK_stn_4*10 + BLK_stn_5*1000
            p1_reward = 1 + BLK_ttl_reward - BLK_prev_reward
            BLK_prev_reward = BLK_ttl_reward
            
            agent_p1.append_sample(p1_state, p1_action, p1_reward, p1_next_state, done_p1)
            # print(plt_ck_brd)                
            # sys.exit()            
            
            if agent_p1.progress == "Training":
                agent_p1.train_model()
                
                # if done or ep_step % agent_p1.target_update_cycle == 0:
            if done_p1:
                # return# copy q_net --> target_net
                agent_p1.Copy_Weights()
                agent_p2.Copy_Weights()
                agent_p1.episode += 1
                print("   BLK stone win!!")
                print("episode :",agent_p1.episode,"BLK mem :",len(agent_p1.memory),"White mem :",len(agent_p2.memory))
                # break
                
            if not done_p1:
                if len(agent_p2.memory) < agent_p2.size_replay_memory:
                    agent_p2.progress = "Exploration"
                else :
                    agent_p2.progress = "Training"

                ep_step += 1
                    
                p2_state = copy.deepcopy(checker_board)
                p2_state = np.reshape(p2_state, [1, state_size])
                p2_action = agent_p2.get_action(p2_state, ep_step)

                # Calculate next_state, reward, done
                # next_state, reward, done, _ = env.step(action)
                checker_board[p2_action] = 1
                p2_next_state = copy.deepcopy(checker_board)
                p2_next_state = np.reshape(p2_next_state, [1, state_size])

                share_2,remainder_2 = divmod(p2_action,n_ticks)
                plt_ck_brd[share_2][remainder_2] = 1
                
                WHT_stn_5 = 0
                WHT_patt_5 = np.array([1,1,1,1,1])
                WHT_stn_4 = 0
                WHT_patt_4 = np.array([1,1,1,1])
                WHT_stn_3 = 0
                WHT_patt_3 = np.array([1,1,1])

                for row_idx in range(n_ticks):
                    arr_x = np.array(plt_ck_brd[row_idx])
                    done_num_5 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_5)] for j in range(len(arr_x) - len(WHT_patt_5) + 1)]
                              if np.array_equal(k, WHT_patt_5))
                    WHT_stn_5 += done_num_5

                    done_num_4 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_4)] for j in range(len(arr_x) - len(WHT_patt_4) + 1)]
                              if np.array_equal(k, WHT_patt_4))
                    WHT_stn_4 += done_num_4

                    done_num_3 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_3)] for j in range(len(arr_x) - len(WHT_patt_3) + 1)]
                              if np.array_equal(k, WHT_patt_3))
                    WHT_stn_3 += done_num_3

                tp_plt_ck_brd = np.transpose(plt_ck_brd)
                for col_idx in range(n_ticks):
                    arr_x = np.array(tp_plt_ck_brd[col_idx])
                    done_num_5 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_5)] for j in range(len(arr_x) - len(WHT_patt_5) + 1)]
                              if np.array_equal(k, WHT_patt_5))
                    WHT_stn_5 += done_num_5 

                    done_num_4 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_4)] for j in range(len(arr_x) - len(WHT_patt_4) + 1)]
                              if np.array_equal(k, WHT_patt_4))
                    WHT_stn_4 += done_num_4

                    done_num_3 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_3)] for j in range(len(arr_x) - len(WHT_patt_3) + 1)]
                              if np.array_equal(k, WHT_patt_3))
                    WHT_stn_3 += done_num_3

                dia_idx = -8
                while dia_idx < 9:
                    arr_x = np.diag(plt_ck_brd, k=dia_idx)
                    done_num_5 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_5)] for j in range(len(arr_x) - len(WHT_patt_5) + 1)]
                              if np.array_equal(k, WHT_patt_5))
                    WHT_stn_5 += done_num_5 

                    done_num_4 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_4)] for j in range(len(arr_x) - len(WHT_patt_4) + 1)]
                              if np.array_equal(k, WHT_patt_4))
                    WHT_stn_4 += done_num_4

                    done_num_3 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_3)] for j in range(len(arr_x) - len(WHT_patt_3) + 1)]
                              if np.array_equal(k, WHT_patt_3))
                    WHT_stn_3 += done_num_3

                    dia_idx += 1

                flip_dia_idx = -8
                while flip_dia_idx < 9:
                    arr_x = np.fliplr(plt_ck_brd).diagonal(flip_dia_idx)
                    done_num_5 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_5)] for j in range(len(arr_x) - len(WHT_patt_5) + 1)]
                              if np.array_equal(k, WHT_patt_5))
                    WHT_stn_5 += done_num_5 

                    done_num_4 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_4)] for j in range(len(arr_x) - len(WHT_patt_4) + 1)]
                              if np.array_equal(k, WHT_patt_4))
                    WHT_stn_4 += done_num_4

                    done_num_3 = sum(1 for k in 
                              [arr_x[j:j+len(WHT_patt_3)] for j in range(len(arr_x) - len(WHT_patt_3) + 1)]
                              if np.array_equal(k, WHT_patt_3))
                    WHT_stn_3 += done_num_3

                    flip_dia_idx += 1

                if WHT_stn_5 > 0 :
                    done_p2 = True
                    done_p1 = True

                WHT_ttl_reward = WHT_stn_3*1 + WHT_stn_4*10 + WHT_stn_5*1000
                
                p2_reward = 1 + WHT_ttl_reward - WHT_prev_reward
                WHT_prev_reward = WHT_ttl_reward

                agent_p2.append_sample(p2_state, p2_action, p2_reward, p2_next_state, done_p2)

                if agent_p2.progress == "Training":
                    agent_p2.train_model()
                if done_p2:
                    # return# copy q_net --> target_net
                    agent_p1.Copy_Weights()
                    agent_p2.Copy_Weights()
                    agent_p1.episode += 1
                    print("   WHT stone win!!")
                    print("episode :",agent_p1.episode,"BLK mem :",len(agent_p1.memory),"White mem :",len(agent_p2.memory))
                    # break
                # print(plt_ck_brd.astype(int))            
                
    agent_p1.model.save_weights(model_path + "/model_p1.h5")
    agent_p2.model.save_weights(model_path + "/model_p2.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
    
if __name__ == "__main__":
    main()