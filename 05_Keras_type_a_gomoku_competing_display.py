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
import pygame
from pygame.locals import QUIT

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

class Gomoku:
    def __init__(self):
        self.n_ticks = n_ticks
        self.size_chkr_brd = self.n_ticks * self.n_ticks
        self.WHT_prev_reward = 0
        self.BLK_prev_reward = 0
        
    def reset_env(self):
        self.checker_board = np.zeros(self.size_chkr_brd, dtype=int)

        return self.checker_board
        
    def p1_frame_step(self, p1_action, ep_step):

        self.checker_board[p1_action] = 2
        p1_next_state = copy.deepcopy(self.checker_board)
        
        plt_ck_brd = np.reshape(self.checker_board, (-1, n_ticks))
        
        p1_done = False
        
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
            p1_done = True

        BLK_ttl_reward = BLK_stn_3*1 + BLK_stn_4*10 + BLK_stn_5*1000
        p1_reward = 1 + BLK_ttl_reward - self.BLK_prev_reward
        self.BLK_prev_reward = BLK_ttl_reward

        return p1_next_state, p1_reward, p1_done
    
    def p2_frame_step(self, p2_action, ep_step):
        self.checker_board[p2_action] = 1
        p2_next_state = copy.deepcopy(self.checker_board)
        
        plt_ck_brd = np.reshape(p2_next_state, (-1, n_ticks))

        p2_done = False
        p1_done = False
        
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
            p2_done = True
            p1_done = True

        WHT_ttl_reward = WHT_stn_3*1 + WHT_stn_4*10 + WHT_stn_5*1000

        p2_reward = 1 + WHT_ttl_reward - self.WHT_prev_reward
        self.WHT_prev_reward = WHT_ttl_reward

        return p2_next_state, p2_reward, p2_done, p1_done

# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN_p1:
    def __init__(self, state_size, action_size):
        
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # These are hyper parameters for the DQN_p1
        self.learning_rate = 0.001
        
        self.hidden1, self.hidden2 = 256, 256
        
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 32
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # create main model and target model
        self.model = self.build_model()
        
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

    # get action from model using epsilon-greedy policy
    def get_action(self, state, ep_step):
        # print("Episode step :",ep_step)
        #Exploration vs Exploitation
        if ep_step == 0:
            # print("  BLK player-Random action selected!!")
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
        
        self.hidden1, self.hidden2 = 300, 300
        
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 32
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # create main model and target model
        self.model = self.build_model()
        
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

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        
        q_value = self.model.predict(state)
        nonzero_chkr_brd = state > 0
        q_value = ~nonzero_chkr_brd * q_value
        p2_action = np.argmax(q_value[0])
        return p2_action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
            
def main():
    
    agent_p1 = DQN_p1(state_size, action_size)
    agent_p2 = DoubleDQN_p2(state_size, action_size)
    game = Gomoku()
    
    agent_p1.model.load_weights(model_path + "/model_p1.h5")
    agent_p2.model.load_weights(model_path + "/model_p2.h5")
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agent_p1.episode = 0
    ep_step = 0
    
    pygame.init()
    SURFACE = pygame.display.set_mode((40*(n_ticks+1), 40*(n_ticks+1)))
    FPSCLOCK = pygame.time.Clock()
    
    while agent_p1.episode < 5:
       
        # reset environment
        checker_board = game.reset_env()
        
        p1_done = False
        p2_done = False
        ep_step = 0
        
        SURFACE.fill((238,197,145))
        # RGB(238,197,145)
        for idx in range(n_ticks):
            pygame.draw.line(SURFACE, (0, 0, 0), (40+idx*40,40), (40+idx*40, 40+(n_ticks-1)*40),1)
            
        for idx in range(n_ticks+1):
            pygame.draw.line(SURFACE, (0, 0, 0), (40,40+idx*40), (40+(n_ticks-1)*40, 40+idx*40),1)
        
        while not p1_done and ep_step < size_chkr_brd:
            
            p1_state = copy.deepcopy(checker_board)
            
            p1_state = np.reshape(p1_state, [1, state_size])
            
            p1_action = agent_p1.get_action(p1_state, ep_step)
            
            ep_step += 1
            
            # Calculate next_state, reward, done
            
            p1_next_state, p1_reward, p1_done = game.p1_frame_step(p1_action, ep_step)
            plt_ck_brd = np.reshape(p1_next_state, (-1, n_ticks))
            p1_next_state = np.reshape(p1_next_state, [1, state_size])
            
            share_v,remainder_v = divmod(p1_action,n_ticks)
            plt_ck_brd[share_v][remainder_v] = 2
            
            pygame.draw.circle(SURFACE, (0, 0, 0), ((share_v+1)*40,(remainder_v+1)*40), 15)
            
            pygame.display.update()
            time.sleep(0.2)
            # time.sleep(10)
            FPSCLOCK.tick(30)
            
            agent_p1.append_sample(p1_state, p1_action, p1_reward, p1_next_state, p1_done)
            
            if p1_done:
                agent_p1.episode += 1
                print("   BLK stone win!!")
                time.sleep(5)
                print(plt_ck_brd.astype(int))
                print("episode :",agent_p1.episode,"BLK mem :",len(agent_p1.memory),"White mem :",len(agent_p2.memory))
                # break
                
            if not p1_done:
                ep_step += 1
                
                p2_state = copy.deepcopy(p1_next_state)
                
                plt_ck_brd = np.reshape(p2_state, (-1, n_ticks))
                # print(plt_ck_brd)
                
                p2_state = np.reshape(p2_state, [1, state_size])
                p2_action = agent_p2.get_action(p2_state)
                p2_next_state, p2_reward, p2_done, p1_done = game.p2_frame_step(p2_action, ep_step)
                
                plt_ck_brd = np.reshape(p2_next_state, (-1, n_ticks))
                p2_next_state = np.reshape(p2_next_state, [1, state_size])

                share_2,remainder_2 = divmod(p2_action,n_ticks)
                plt_ck_brd[share_2][remainder_2] = 1
                
                pygame.draw.circle(SURFACE, (255, 255, 255), ((share_2+1)*40,(remainder_2+1)*40), 15)
                
                pygame.display.update()
                time.sleep(0.2)
                # time.sleep(10)
                FPSCLOCK.tick(30)
                
                agent_p2.append_sample(p2_state, p2_action, p2_reward, p2_next_state, p2_done)

                if p2_done:
                    agent_p1.episode += 1
                    print("   WHT stone win!!")
                    time.sleep(5)
                    print(plt_ck_brd.astype(int))
                    print("episode :",agent_p1.episode,"BLK mem :",len(agent_p1.memory),"White mem :",len(agent_p2.memory))
                    # break
                # print(plt_ck_brd.astype(int))            
                
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
    
if __name__ == "__main__":
    main()
