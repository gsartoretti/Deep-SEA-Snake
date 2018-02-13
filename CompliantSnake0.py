import Snake
import time
from HebiWrapper import Gains
from tools0 import *
from DataLogging import DataLogger
import Optitrack

import numpy as np
pi = np.pi

GLOBAL_EP = 5000
continue_ep = True
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
snakeCompl.tmFreq         = np.array([[0]])
snakeCompl.slope          = 200
snakeCompl.phi            = 0

## np.concatenate simulates matlab array behavior
snakeCompl.tau_D          = np.concatenate((3 * np.ones((numWindows, 1)),
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
dt                        = pi / 160

# Initial Shape of the snake (for use while calibrating the complementary filter)
snakeCompl.offset      = np.zeros((numWindows, 1))

commanded_angles       = get_NewAngles2D(snakeCompl, snakeConst, windows)[0]
## no need to flip
cmd.position	       = (1.7 * np.transpose(commanded_angles))

if continue_ep:
    import pickle
    with open('runData.snake', 'rb') as file:
        data = pickle.load(file)
        print(data.keys())
        GLOBAL_EP = data['GLOBAL_EP']
    GLOBAL_EP += 1

log = DataLogger(GLOBAL_EP, recordVideo = True)
log.logData('snakeConst, windows, snakeCompl, snakeNom, dt', snakeConst, windows, snakeCompl,     snakeNom, dt)

log.logData('tau_Ext, tau_Applied, snakeNom, snakeCompl, windows,       commanded_angles, current_time, reward, oldpos, currentpos')

## Python API
snake.setAngles(cmd.position, send = False)
##

TzOffset = commanded_angles[1] / 2

snake.sendCommand()
##

input('press enter to initialize\n')

rewardData = Optitrack.Reward(bufferSize = 5, episodeLength = 1, rewardFactor = 100)
time.sleep(0.2)
reward = 0
try:
    while True: ## press Ctrl-C to stop
        tic = time.clock()

        forward = 1

        index = 1

        while (windows.offset[index][1] < 0):
            index = index + 1
        windows.steerIndex = index

        snakeNom.scale          = forward
        t                       = t + dt

        tau_Ext                 = snake.getTorques() ## torque should be automatically converted to unified
        tau_Applied             = get_AppliedForce2D(tau_Ext, snakeConst, snakeCompl, windows)
        snakeCompl, snakeNom    = get_NewNomParam2D(snakeCompl, snakeNom, snakeConst, windows, tau_Applied, dt)
        windows                 = update_Windows(snakeCompl, windows)
        snakeCompl, snakeNom, windows\
                                = update_Structures2D(snakeCompl,snakeConst,snakeNom,windows)

        commanded_angles, mainAxis, currSerpPos, shapeReg \
                                = get_NewAngles2D(snakeCompl, snakeConst, windows)
        cmd.position		= commanded_angles

        snake.setAngles(cmd.position.transpose(),send= False)
        reward, (currentpos, oldpos) = rewardData.getReward()

        snake.sendCommand()
        current_time = time.clock()-tic
        log.logData(tau_Ext, tau_Applied, snakeNom, snakeCompl, windows, commanded_angles, current_time, reward, oldpos, currentpos)
        print(t, end='\r')
        time.sleep(max(0,dt-current_time))
except KeyboardInterrupt:
    with open('runData.snake', 'wb') as file:
        data = {}
        data['GLOBAL_EP'] = GLOBAL_EP
        pickle.dump(data, file)
    log.saveData()
print('\ndone.')
