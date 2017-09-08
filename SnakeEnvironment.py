import gym
import numpy as np
from gym.utils import seeding
from gym import spaces

inf = float('inf')

#EPS_AMP = 0.005
#EPS_SPF = 0.02

EPS_AMP = 0.005
EPS_SPF = 0.012
TANH_FACT = 100
REW_FACT = 1

AMP_NOM = np.pi / 4
SPF_NOM = 2 * np.pi * 1.5

PROBLEMATIC_RUN = 268
#PROBLEMATIC_RUN = 269

class DeterministicBox(spaces.Box):
    def __init__(self, low, high, shape=None, samplefn=None):
        """ Deterministic box - supply samplefn and env to produce deterministic samples.
        samplefn is called when sample is called, and is called as samplefn()"""
        self.samplefn = samplefn
        spaces.Box.__init__(self,low,high,shape)

    def sample(self):
        return self.samplefn()


class SnakeEnv(gym.Env):
    def __init__(self, experiments, workerID, episode_length = 89):
        """ experiments should be an array, where each element is itself an
        array representing a single experiment. Within each experiment (array),
        elements should be dictionaries with 4 fields (modular_time, snake_shape,
        torques, position). modular_time is a float, snake_shape is a 3x2 array,
        torques is a 3x2 array, and position is a 2x1 array (x,y) representing
        the position of the snake"""
        self.experiments = experiments
        self.episode_length = episode_length

        #self.action_space = DeterministicBox(-MAX_DELTA, MAX_DELTA, (3,2), self.current_action)

        #self.observation_space = spaces.Tuple((spaces.Box(0,1, ()),
        #                                       spaces.Box(0, inf, (3,2)),
        #                                       spaces.Box(-45,45,(3,2))))
        self.action_space = DeterministicBox(0, 8, (), self.current_action)

        self.observation_space = spaces.Tuple((spaces.Box(0,1, ()),
                                               spaces.Box(0, inf, (1,2)),
                                               spaces.Box(-45,45,(1,2))))
        self.reward_range = (-1,1)

        self.current_episode = None
        self.state = None
        self.action_index = 0
        self.seed()

        self.current_discrete_state = [0, 0]
        self.action_list = []
        for a1 in [-EPS_AMP,0,EPS_AMP]:
            for a2 in [-EPS_SPF,0,EPS_SPF]:
                self.action_list.append([a1,a2])
        self.experiment_index=0
        self.workerID = workerID

    def current_action(self):
        # print(len(self.current_episode))
        #print(self.action_index)
        actions = list(self.current_episode[self.action_index]['action'])
        discrete_actions = actions[:]
        shape = self.current_episode[self.action_index]['snake_shape'][self.window_index]
        actions = -(np.array(self.current_discrete_state) - np.array(shape))

#        discrete_actions[0] = np.sign(actions[0]) * EPS_AMP
#        discrete_actions[1] = np.sign(actions[1]) * EPS_SPF

#        if abs(actions[0]) < EPS_AMP/2.0:
#            discrete_actions[0] = 0
#        if abs(actions[1]) < EPS_SPF/2.0:
#            discrete_actions[1] = 0

        """
        amp_actions = [-EPS_AMP, 0, EPS_AMP]
        actions[0] = np.random.normal(loc = actions[0] / 10, scale = EPS_AMP / 2)
        spf_actions = [-EPS_SPF, 0, EPS_SPF]
        actions[1] = np.random.normal(loc = actions[1] / 10, scale = EPS_SPF / 2)
        """

        discrete_actions[0] = np.sign(actions[0]) * EPS_AMP
        discrete_actions[1] = np.sign(actions[1]) * EPS_SPF

        if abs(actions[0]) < EPS_AMP/2.0:
            discrete_actions[0] = 0
        if abs(actions[1]) < EPS_SPF/2.0:
            discrete_actions[1] = 0
        self.current_discrete_state = discrete_actions + np.array(self.current_discrete_state)

        #dists_amp /= np.sum(abs(dists_amp))
        #prob = np.random.rand()
        #if prob < dists_amp[0]:
        #    discrete_actions[0] = amp_actions[0]
        #elif prob < dists_amp[0]+dists_amp[1]:
        #    discrete_actions[0] = amp_actions[1]
        #else:
        #    discrete_actions[0] = amp_actions[2]

        #spf_actions = [-EPS_SPF, 0, EPS_SPF]
        #dists_spf = np.array(spf_actions) - actions[1]
        #dists_spf /= np.sum(abs(dists_spf))
        #prob = np.random.rand()
        #if prob < dists_spf[0]:
        #    discrete_actions[1] = spf_actions[0]
        #elif prob < dists_spf[0]+dists_spf[1]:
        #    discrete_actions[1] = spf_actions[1]
        #else:
        #    discrete_actions[1] = spf_actions[2]

        # return discretized action, as well as continuous (recorded) action for plotting purposes
        return self.action_list.index(discrete_actions), list(self.current_episode[self.action_index]['action'])
        # for continuous-A3C
#        return self.action_list.index(discrete_actions), list(self.current_episode[self.action_index]['action']), list(shape)

    def _extract_state(self, read_state):
        self.state = (np.array(read_state['modular_time']),
                      np.array(read_state['snake_shape'][self.window_index]), # Should be a 1x2 array of amps and spacial frequencies
                      np.array(read_state['torques'][self.window_index]), # Should be a 1x2 array of amp torque and spacial torque
                      np.array([AMP_NOM, SPF_NOM])) # Nominal Amplitude/Spatial Frequency

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # uncomment the next line if we actually care about the next action
        # assert self.action_space.contains(action), '%r (%s) invalid' % (action, type(action))
        state = self.current_episode[self.action_index]
        self._extract_state(state)
        done = state['done']

        reward = state['progression'] - self.current_episode[self.action_index-1]['progression']
        reward = REW_FACT * np.tanh( TANH_FACT * reward )
        '''if reward > 0:
            reward = 1
        else:
            reward = -1'''
        self.action_index += 1
        # print(self.action_index)

        return flattenState(self.state), reward, done, {}

    def _reset(self):
        experiment_index = self.np_random.randint(0, len(self.experiments))
        #experiment_index = self.experiment_index
        while experiment_index == PROBLEMATIC_RUN:
            experiment_index = self.np_random.randint(0, len(self.experiments))
        experiment = self.experiments[experiment_index]
        assert len(experiment) >= self.episode_length or self.episode_length == -1
        if self.episode_length != -1:
            start = self.np_random.randint(0, len(experiment) - self.episode_length)
            episode = experiment[start:start+self.episode_length]

        else:
            episode = experiment

#        self.window_index = self.np_random.randint(0, 3)
        self.window_index = self.workerID

        origin = np.array(episode[0]['position'])
        for i, state in enumerate(episode):
            state['progression'] = sum((np.array(state['position']) - origin) ** 2)
            if self.episode_length == -1:
                state['done'] = i >= len(episode)
            else:
                state['done'] = i >= self.episode_length
            if i > 0:
                state['action'] = np.array(state['snake_shape'][self.window_index]) - np.array(episode[i-1]['snake_shape'][self.window_index])
            else:
                state['action'] = np.array([0,0])

        self.current_episode = episode
        self.action_index = 0

        self._extract_state(state)
        self.current_discrete_state = episode[0]['snake_shape'][self.window_index]
        return flattenState(self.state),len(experiment),experiment_index
    
    def resetExperienceReplay(self,workerID, gamma, experiencelength): #gamma: reward discount
        experiment_index = self.np_random.randint(0, len(self.experiments))
        #experiment_index = self.experiment_index
        while experiment_index == PROBLEMATIC_RUN:
            experiment_index = self.np_random.randint(0, len(self.experiments))
        experiment = self.experiments[experiment_index]
        assert len(experiment) >= self.episode_length or self.episode_length == -1
        
        start = self.np_random.randint(0, len(experiment) - experiencelength)
        episode = experiment[start:start+experiencelength+1]
        
        self.window_index = workerID 

        origin = np.array(episode[0]['position'])
        for i, state in enumerate(episode):
            state['progression'] = sum((np.array(state['position']) - origin) ** 2)
            state['done'] = i >= self.episode_length
            if i > 0:
                state['action'] = np.array(state['snake_shape'][self.window_index]) - np.array(episode[i-1]['snake_shape'][self.window_index])
            else:
                state['action'] = np.array([0,0])

        self._extract_state(state)
        self.current_discrete_state = episode[0]['snake_shape'][self.window_index]
        
        self.current_episode = episode
        self.action_index = 0
        
        a        = self.action_space.sample()
        s1,r,_,_ = self._step(a)

        v_s1 = 0
        for i in range(experiencelength):
            _,r,_,_ = self._step(None)
            v_s1 += gamma**i * r
        
        return flattenState(self.state), a, r, v_s1

    def getbatch(self,batch_size,trace_length):
        traceBuffer = []
        batchBuffer = []
       # print(batch_size)
        for i in range(batch_size):
            experiment_index = self.np_random.randint(0, len(self.experiments))
            experiment = self.experiments[experiment_index]
            assert len(experiment) >= trace_length
            start = self.np_random.randint(0, len(experiment) - trace_length)
            episode = experiment[start:start+trace_length]
            self.current_episode = episode
            self.window_index = 2
            origin = np.array(episode[0]['position'])
            # print(i)
            for k, state in enumerate(episode):
                state['progression'] = sum((np.array(state['position']) - origin) ** 2)
                state['done'] = k >= trace_length
                if k > 0:
                	state['action'] = np.array(state['snake_shape'][self.window_index]) - np.array(episode[k-1]['snake_shape'][self.window_index])
                else:
                    state['action'] = np.array([0,0])
            self.action_index = 0
            self._extract_state(state)
            self.current_discrete_state = episode[0]['snake_shape'][self.window_index]
            s = flattenState(self.state)
                # print(flattenState(self.state))
                # print(len(episode))
            for j in range(trace_length):
                a,_ = self.action_space.sample()
                # print(j,a)
                s1,r,d,_ = self.step(a)
                traceBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
                # print(traceBuffer)
                self.state = s1
            batchBuffer = np.append(batchBuffer,traceBuffer)
            # print('batchBuffer')
            # print (batchBuffer)
            batchBuffer = np.array(batchBuffer)
            traceBuffer=[]
        return np.reshape(batchBuffer,[batch_size*trace_length,5])


    def resetRandomRun(self,experiment_index=None):
        if experiment_index == None:
            experiment_index = self.np_random.randint(0, len(self.experiments))
            while experiment_index == f:
                experiment_index = self.np_random.randint(0, len(self.experiments))
        print(experiment_index)
#        self.window_index = self.np_random.randint(0, 3)
        self.window_index = 2

        experiment = self.experiments[experiment_index]
        assert len(experiment) >= self.episode_length
        episode = experiment

        origin = np.array(episode[0]['position'])
        for i, state in enumerate(episode):
            state['progression'] = sum((np.array(state['position']) - origin) ** 2)
            state['done'] = i >= self.episode_length
            if i > 0:
                state['action'] = np.array(state['snake_shape'][self.window_index]) - np.array(episode[i-1]['snake_shape'][self.window_index])
            else:
                state['action'] = np.array([0,0])

        self.current_episode = episode
        self.action_index = 0
        self.current_discrete_state = episode[0]['snake_shape'][self.window_index]

        self._extract_state(state)
        return flattenState(self.state), len(episode)


def flattenState(state):
    # to convert state to a 1d vector:
    vector = list(np.array([[x for x in y] for y in state[1:]]).flatten())
    vector.insert(0, float(state[0]))
    vector = [vector]
    return vector

if __name__ == '__main__':
    fake_snake_exp = [[{'modular_time':np.random.uniform(0,1,()), 'snake_shape':np.random.uniform(0, 20,(3,2)), 'torques':np.random.uniform(-45,45,(3,2)), 'position':np.random.uniform(-20,20,(2,))} for x in range(100)] for x in range (20)]
    env = SnakeEnv(fake_snake_exp)
    # s = env.reset()
    # print(s)
    # s1 = env.step(None)
    # print(s1)
    import pickle
    with open ("p_experiments.snake","rb") as f:
	    exps = pickle.load(f)
    env = SnakeEnv(exps)
    # bt = env.getbatch(batch_size=4,trace_length=8)
    a,_ = env.action_space.sample()
    print(a)
