from __future__ import division
import Snake
import time
from HebiWrapper import Gains
from tools import *
import pickle
import Optitrack
from DataLogging import DataLogger

import numpy as np
pi = np.pi

import tensorflow as tf
import copy

import SnakeEnvironment
import random
import matplotlib.pyplot as plt
import sys, os
import multiprocessing
import threading
import shutil
import GroupLock

import cv2

## To open tensorboard:
## tensorboard --logdir w1:'train_W_0',w2:'train_W_1',w3:'train_W_2',w4:'train_W_3'

TIME_FACTOR = 1

## A3C parameters
OUTPUT_GRAPH            = True
LOG_DIR                 = './log'
GLOBAL_NET_SCOPE        = 'Global_Net'
UPDATE_GLOBAL_ITER      = 89
GAMMA                   = 0.995
ENTROPY_BETA            = 0.001
LR_A                    = 0.001    # learning rate for actor
LR_C                    = 0.001    # learning rate for critic
GLOBAL_REWARD           = []
GLOBAL_EP               = 5000
N_WORKERS               = 6 #should =multiprocessing.cpu_count()
model_path              = './model_offlineA3C'
s_size                  = 7
a_size                  = 9
load_model              = True
continue_EP             = True
#continue_EP             = False

# Other Params
MIN_AMPLITUDE           = 0
MIN_SPFREQ              = 1/3.

# To make the code closer to source
class Struct:
    pass
#
## Make structs
snakeCompl = Struct()
snakeNom = Struct()
snakeConst = Struct()
windows = Struct()
cmd = Struct()

# initialize snake
snake = Snake.Snake('SA043', cmdLifetime = 0)

# initialize gains (default values should be ok)

gains = Gains(snake.numModules)
snake.setGains(gains)

snake.setFeedbackFrequency(200)

# joystick stuff would go here if we had it


# Set up parameters for SEA Snake
snakeConst.numModules     = 16
snakeConst.numWaves       = 1.5
snakeConst.spFreq         = 2 * pi * snakeConst.numWaves
snakeConst.tmFreq         = 1.8
snakeConst.ampEvn         = pi / 4
snakeConst.ampOdd         = 0


# Initialize all parameters for the windows
windows.numWindows        = 6
windows.origin            = 1
windows.steerIndex        = 3

windows.array             = initialize_Windows(snakeConst, windows)
windows.amp               = initialize_AmpWindows(windows)
windows.spFreq            = initialize_SpFreqWindows(windows)
windows.offset            = initialize_OffsetWindows(windows)


# Initialize all parameters for the compliant variables

numWindows                = windows.numWindows
snakeCompl.amp            = snakeConst.ampEvn * np.ones((1, numWindows))
snakeCompl.spFreq         = snakeConst.spFreq * np.ones((1, numWindows))
# snakeCompl.torsion        = np.zeros((1, numWindows))

## tmFreq is used as a 1x1 matrix
snakeCompl.tmFreq         = np.array([[ 0 ]])
snakeCompl.slope          = 200
snakeCompl.phi            = 0

## np.concatenate simulates matlab array behavior
snakeCompl.tau_D          = np.concatenate((3.5 * np.ones((numWindows, 1)),
                                            0 * np.ones((numWindows, 1))))
snakeNom.steer            = 0
snakeNom.sigma_Nom        = np.concatenate((np.zeros((numWindows,1)),
                                            np.transpose(snakeCompl.amp),
                                            np.transpose(snakeCompl.spFreq),
                                            np.transpose(snakeCompl.tmFreq)))

snakeNom.sigma_D          = np.concatenate((np.zeros((numWindows,1)),
                                            2 * np.transpose(snakeCompl.amp),
                                            np.transpose(snakeCompl.spFreq),
                                            np.transpose(snakeCompl.tmFreq)))

snakeNom.MdPrime           = np.diag(np.concatenate((1.5 * np.ones((1, windows.numWindows)),
                                              (2 * np.ones((1, windows.numWindows)))),axis=1)[0])

snakeNom.BdPrime           = np.diag(np.concatenate((3 * np.ones((1, windows.numWindows)),
                                              (1 * np.ones((1, windows.numWindows)))),axis=1)[0])
snakeNom.KdPrime           = np.diag(np.concatenate((4 * np.ones((1, windows.numWindows)),
                                              (1 * np.ones((1, windows.numWindows)))),axis=1)[0])

snakeNom.dsigmaD_dt       = np.zeros((2 * numWindows, 1))

t                         = 0
dt                        = TIME_FACTOR * pi / 160 # CHANGED

# Initial Shape of the snake (for use while calibrating the complementary filter)
snakeCompl.offset      = np.zeros((numWindows, 1))

commanded_angles       = get_NewAngles2D(snakeCompl, snakeConst, windows)[0]
## cmd.position           = changeUnifiedToSEA(snakeData,1.7*np.transpose(commanded_angles))
## no need to flip
cmd.position           = (1.9 * np.transpose(commanded_angles))

snake.setAngles(cmd.position, send = False)
##snake.set(cmd); -> snake.sendCommand()
snake.sendCommand()

## Class Actor and Critic network
class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # Only need parameters of global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, s_size], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # Uselocal net to calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, s_size], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                #Get policy(a_prob) and value(v) from actor, critic net
                self.a_prob, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, a_size, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    #Found someone use entropy to encourage exploration
                    #larger entropy means more stochastic actions
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # Synchronization
            with tf.name_scope('sync'):
                #assign global parameters to local parameters
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [local_param.assign(global_param) for local_param, global_param in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [local_param.assign(global_param) for local_param, global_param in zip(self.c_params, globalAC.c_params)]
                #update params of global net by pushing the calculated gradients of local net to global net
                with tf.name_scope('push'):
                    self.update_a_op = trainer_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = trainer_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            #dense layer is normal NN,see: https://www.tensorflow.org/api_docs/python/tf/layers/dense
            layer4 = tf.layers.dense(self.s, a_size, tf.nn.relu6, kernel_initializer=w_init, name='layer4')
            layer3 = tf.layers.dense(layer4, a_size, tf.nn.relu6, kernel_initializer=w_init, name='layer3')
            layer2 = tf.layers.dense(layer3, a_size, tf.nn.relu6, kernel_initializer=w_init, name='layer2')
            a_prob = tf.layers.dense(layer2, a_size, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            v4 = tf.layers.dense(self.s, 1, tf.nn.relu6, kernel_initializer=w_init, name='v4')#relu6 is better than relu
            v3 = tf.layers.dense(v4, 1, tf.nn.relu6, kernel_initializer=w_init, name='v3')#relu6 is better than relu
            v2 = tf.layers.dense(v3, 1, tf.nn.relu6, kernel_initializer=w_init, name='v2')#relu6 is better than relu
            v = tf.layers.dense(v2, 1, tf.nn.relu6, kernel_initializer=w_init, name='v')#relu6 is better than relu
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        c_loss,a_loss,_,_ = SESS.run([self.c_loss, self.a_loss,self.update_a_op, self.update_c_op], feed_dict)# local grads applies to global net
        return a_loss,c_loss

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: np.reshape(np.array(s),[1,s_size])})
        print(prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

##Worker class
class Worker(object):
    def __init__(self, name, globalAC, group_number,groupLock):
        self.group_number=group_number
        self.groupLock=groupLock
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.model_path = model_path
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.name))
        self.shouldDie = False

    def work(self):
        global GLOBAL_REWARD, GLOBAL_EP, dsigmaD,thread_feedback
        buffer_s, buffer_a, buffer_r = [], [], []
        s = s1 = self.Initialize_states(self.name)
        ep_step = ep_r = 0
        while True:
            self.groupLock.acquire(self.group_number,self.name)

            if self.shouldDie:
                # release lock and stop thread
                print('Thread %s exiting now...' % (self.name))
                self.groupLock.release(self.group_number,self.name)
                return

            s     = self.Read_feedback(self.name)['state']
            a     = self.AC.choose_action(s)
            r     = self.Read_feedback(self.name)['reward']
            if s[1] + TIME_FACTOR * actions_list[a][0] < MIN_AMPLITUDE:
                a += 3
            if s[2] + TIME_FACTOR * actions_list[a][1] < MIN_SPFREQ:
                a += 1
            self.Write_dsigma(self.name,a)

            # Update time step counters
            ep_step    += 1
            ep_r       += r

            if ep_step == UPDATE_GLOBAL_ITER and self.name == 'W_0':
                GLOBAL_REWARD.append(ep_r)

                if GLOBAL_EP % 5 == 0:
                    mean_reward = np.nanmean(GLOBAL_REWARD[-5:])
                    print(mean_reward)
                    summary=tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary_writer.add_summary(summary, GLOBAL_EP)
                    self.summary_writer.flush()

                ep_step    = 0
                ep_r       = 0
                GLOBAL_EP += 1
            self.groupLock.release(self.group_number,self.name)


    def Read_feedback(self,name):
        global thread_feedback
        return {'W_0':thread_feedback[0],'W_1':thread_feedback[1],\
                'W_2':thread_feedback[2],'W_3':thread_feedback[3],\
                'W_4':thread_feedback[4],'W_5':thread_feedback[5]}[name]

    def Write_dsigma(self,name,a):
        global dsigmaD
        if name =="W_0":
            dsigmaD[0][0],dsigmaD[0][6]      = actions_list[a][0],actions_list[a][1]
        if name =="W_1":
            dsigmaD[0][1],dsigmaD[0][7]      = actions_list[a][0],actions_list[a][1]
        if name =="W_2":
            dsigmaD[0][2],dsigmaD[0][8]      = actions_list[a][0],actions_list[a][1]
        if name =="W_3":
            dsigmaD[0][3],dsigmaD[0][9]      = actions_list[a][0],actions_list[a][1]
        if name =="W_4":
            dsigmaD[0][4],dsigmaD[0][10]     = actions_list[a][0],actions_list[a][1]
        if name =="W_5":
            dsigmaD[0][5],dsigmaD[0][11]     = actions_list[a][0],actions_list[a][1]

    def Initialize_states(self, name):
        #windows.numWindows
        len               = 6
        #windows.steerIndex
        index             = 1
        #snakeNom.sigma_Nom
        sigma_Nom         = np.concatenate((np.zeros((numWindows,1)),
                                                np.transpose(snakeCompl.amp),
                                                np.transpose(snakeCompl.spFreq),
                                                np.transpose(snakeCompl.tmFreq)))
        #snakeNom.sigma_D
        sigma_D           = np.concatenate((np.zeros((numWindows,1)),
                                                2 * np.transpose(snakeCompl.amp),
                                                np.transpose(snakeCompl.spFreq),
                                                np.transpose(snakeCompl.tmFreq)))
        serp_period       = 2 * np.pi / snakeConst.tmFreq
        initialization =[]
        for win in range(len):
            state               = np.array( [0., sigma_D[len+win], sigma_D[2*len+win], 0., 0., snakeConst.ampEvn, snakeConst.spFreq] ).transpose() #time=0, amp, spFreq, amp_tau=0, spFreq_tau=0
            initialization.append(state)
        return{'W_0':initialization[0],'W_1':initialization[1],\
               'W_2':initialization[2],'W_3':initialization[3],\
               'W_4':initialization[4],'W_5':initialization[5],}[name]




if __name__ == "__main__":
    groups = [['main'],[]]
    for i in range(N_WORKERS):#N_WORKERS
        i_name = 'W_%i' % i   # worker name
        groups[1].append(i_name)
    groupLock = GroupLock.GroupLock(groups)

    with open("actionlist.snake","rb") as file:
        dict                = pickle.load(file)
        actions_list        = dict['action']
    if continue_EP:
       with open('runData.snake', 'rb') as file:
            data                = pickle.load(file)
            print(data.keys())
            GLOBAL_EP           = data['GLOBAL_EP']
    log = DataLogger(GLOBAL_EP, recordVideo = True)
    log.logData('snakeConst, windows, snakeCompl, snakeNom, dt', snakeConst, windows, snakeCompl, snakeNom, dt)

    groupLock.acquire(0,'main')#lock here
    SESS = tf.Session()

    #Tensorflow and start threads
    with tf.device("/cpu:0"):
        trainer_A = tf.train.AdamOptimizer(LR_A, name='RMSPropA')
        trainer_C = tf.train.AdamOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC, 1, groupLock))
        saver = tf.train.Saver(max_to_keep=5)

    #COORD = tf.train.Coordinator()
    if load_model ==True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(SESS,ckpt.model_checkpoint_path)
    else:
        SESS.run(tf.global_variables_initializer())


    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        th = threading.Thread(target = worker.work)
        th.start()
        worker_threads.append(th)

    #initialize stuff
    ## Python API

    ##

    spFreqs = []
    amps    = []

    input('press enter to initialize\n')
    rewardData = Optitrack.Reward(bufferSize = 5, episodeLength = 1, rewardFactor = 100)
    time.sleep(0.2)
    reward = t = 0

    log.logData('tau_Ext, tau_Applied, dsigmaD, thread_feedback, snakeNom, snakeCompl, windows, commanded_angles, current_time, reward, oldpos, currentpos')
    try:
        while True: ## press Ctrl-C to stop
            tic = time.perf_counter()

        ## joystick stuff goes here

            forward = 1

            index = 1
            while (windows.offset[index][1] < 0):
                index = index + 1

        ## getting feedback is different with python API
            '''
            fbk =snake.getNextFeedback();
            while isempty(fbk)
            display('No Feedback... :-(');
            fbk = snake.getNextFeedback();
            end
            '''
        ##
        ## Omitting "Guillaume's crap"

            snakeNom.scale          = forward
            t                      += 1

            reward, (currentpos, oldpos) = rewardData.getReward()
            #if t % UPDATE_GLOBAL_ITER == 0:
                #print('Resetting Origin')
                #rewardData.setOrigin()
            tau_Ext                 = snake.getTorques() ## torque should be automatically converted to unified
            tau_Applied             = get_AppliedForce2D(tau_Ext, snakeConst, snakeCompl, windows)
            global thread_feedback
            global dsigmaD
            dsigmaD = np.zeros([1,12])
            thread_feedback = []
            serp_period       = 2 * np.pi / snakeConst.tmFreq

            for win in range(windows.numWindows):
                state               = np.array( [((t*dt) % serp_period) / serp_period, snakeNom.sigma_D[windows.numWindows+win],\
                                                 snakeNom.sigma_D[2*windows.numWindows+win], tau_Applied[win], tau_Applied[windows.numWindows + win], snakeConst.ampEvn, snakeConst.spFreq]).transpose()
                thread_feedback.append({'state': state,'reward':reward})

            groupLock.release(0,'main')
            groupLock.acquire(0,'main')
            snakeNom.sigma_D[6:18] += TIME_FACTOR * np.reshape(dsigmaD[0],[12,1])
            snakeCompl, snakeNom    = get_NewNomParam_A3C(snakeCompl=snakeCompl, snakeNom=snakeNom, snakeConst=snakeConst, windows=windows, tau_Applied=tau_Applied, dt=dt, t=t*dt, worker=worker, actions_list=actions_list, sess=SESS)
            windows                 = update_Windows(snakeCompl, windows)
            snakeCompl, snakeNom, windows\
                        = update_Structures2D(snakeCompl,snakeConst,snakeNom,windows)

            commanded_angles, mainAxis, currSerpPos, shapeReg \
                        = get_NewAngles2D(snakeCompl, snakeConst, windows)
            cmd.position        = commanded_angles

            snake.setAngles(cmd.position.transpose(),send= False)
            snake.sendCommand()

            #amps.append(snakeCompl.amp)
            #spFreqs.append(snakeCompl.spFreq)

            current_time = time.perf_counter() - tic
            log.logData(tau_Ext, tau_Applied, dsigmaD, thread_feedback, snakeNom, snakeCompl, windows, commanded_angles, current_time, reward, oldpos, currentpos)
            if current_time > dt:
                print(t*dt, snakeCompl.phi, snakeCompl.amp, snakeCompl.spFreq, reward, end='\n')
#            time.sleep(max(0,dt-current_time))
    except KeyboardInterrupt:
        # saver.save(SESS, model_path+'/model-'+str(GLOBAL_EP)+'.cptk')
        # print ("Saved Model")

        with open('runData.snake', 'wb') as file:
            data = {}
            #data['amps'] = amps
            #data['spFreqs'] = spFreqs
            data['GLOBAL_EP'] = GLOBAL_EP
            pickle.dump(data, file)
        #shutil.copyfile('runData.snake', 'ExperimentsOff/runData-%i.snake'%(GLOBAL_EP))
        log.saveData()
        print('\nI am CompliantSnake.py and I saved things.')


        # Kill threads
        for w in workers:
            print('Asking thread %s to stop' % (w.name))
            w.shouldDie = True
        time.sleep(0.2)
        groupLock.release(0,'main')

        #cap.release()
        #out.release()
        #cv2.destroyAllWindows()

        print('You can Ctrl-C now...')
        sys.exit()
        try:
            SESS.close()
        except KeyboardInterrupt:
            exit(0)
