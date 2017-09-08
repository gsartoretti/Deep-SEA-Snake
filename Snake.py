import time
import HebiWrapper as hw
import Curves
import numpy as np
from math import pi, sin

try:
    from math import inf
except ImportError:
    inf = float('inf')

HebiLookup = hw.HebiLookup()
class Snake:
    def __init__(self, rootModuleName, rootModuleFamily = '*', moduleLength = 0.0639, cmdLifetime=100):
        self.snakeGroup = HebiLookup.getConnectedGroupFromName(rootModuleName, rootModuleFamily, cmdLifetime = cmdLifetime)
        if self.snakeGroup == None:
            raise IOError ('Snake not found')
        else:
            self.numModules = self.snakeGroup.numModules()
            self.moduleLength = 0.0639
    def _reverse_angles(self, angles):
        for i, angle in enumerate(angles):
            if i // 2 % 2:
                angles[i] = -angles[i]
        return angles
    def _unifiedToSea(self, angles):
        if len(np.array(angles).shape) < 2:
            angles = np.array([angles])
        return self._reverse_angles(list(np.fliplr(angles)[0]))
    def _SeaToUnified(self,angles):
        angles = self._reverse_angles(angles)
        if len(np.array(angles).shape) < 2:
            angles = np.array([angles])
        return list(np.fliplr(angles)[0])
    def getAngles(self, timeout = 15):
        return self._SeaToUnified(self.snakeGroup.getAngles(timeout = timeout))
    def getTorques(self, timeout = 15):
        return self._SeaToUnified(self.snakeGroup.getTorques(timeout = timeout))
    def setAngles(self, angles, send = True, toSEA = True):
        if toSEA:
            angles = self._unifiedToSea(angles)
        if len(angles) != self.numModules:
            return False
        self.snakeGroup.setAngles(angles, send = send)
    def sendCommand(self):
        '''Sends queued command'''
        self.snakeGroup.sendCommand()
    def waitForFeedback(self):
#blocks code until snake responds
        self.snakeGroup.getModuleFeedback(timeout = 300)
    def getFeedback(self, timeout = 15):
        return SnakeFeedback(self, timeout = 15)
    def setFeedbackFrequency(self, hz):
        self.snakeGroup.setFeedbackFrequency(hz)
    def setGains(self, gains):
        """ Give this a function a HebiWrapper.Gains object to set the gains on the snake"""
        commandPointerList = self.snakeGroup.getCommandList()
        gains.setGains(commandPointerList)
        self.snakeGroup.sendCommand(release = True)

class SnakeFeedback:
    def __init__(self, snake, timeout = 15):
        self.torques = snake.getTorques(timeout = timeout)
        self.angles  = snake.getAngles(timeout = timeout)

def goToPosition(snake, angles, maxSpeed = 0.2, dt = pi / 160, error = 0.5):
    def error(angle1, angle2): # dot product
        return sum([x * y for (x,y) in zip(angle1, angle2)])
    while error(angles, snake.getAngles()) > 0.5:
        tic = time.clock()
        newOffsets = [x - y for (x,y) in zip(snake.getAngles(), angles)]
        newOffsets = [max(min(x, maxSpeed), -maxSpeed) for x in newOffsets]
        newAngles = [x + y for (x,y) in zip(snake.getAngles(), newOffsets)]
        snake.setAngles(newAngles)
        toc = time.clock()
        time.sleep(max(0,dt-(toc-tic)))

def curriedAngle(curve):
    def innerAngle(t):
        def innerInnerAngle(tuple):
            distanceToHead, moduleType = tuple
            return Curves.calculateAngle(curve, t, distanceToHead, moduleType)
        return innerInnerAngle
    return innerAngle

def calculateAngles(curve, t, moduleLength, numModules):
    angleFn = curriedAngle(curve)(t)
    angles = [(i * moduleLength, i % 2) for i in range(numModules)]
    angles = list(map(angleFn, angles))
    return angles

def runCurve(snake, dt = pi / 160, loopCount = inf, angleFn = None, curve=Curves.sidewind):
    if not angleFn:
        angleFn = lambda t, feedback, moduleLength, numModules : calculateAngles(curve, t, moduleLength, numModules)
    angles = angleFn(0, None, snake.moduleLength, snake.numModules)
    #goToPosition(snake, angles)
    loops = 0
    averageTime = 0
    while loops < loopCount:
        tic = time.clock()
        #snake.waitForFeedback()
        loops += 1
        feedback = snake.getTorques()
        angles = angleFn(loops * dt, feedback, snake.moduleLength, snake.numModules)
        snake.setAngles(angles)
        toc = time.clock()
        averageTime += toc-tic
        if (dt-toc+tic) < 0:
            print('Lagging{}'.format(' ' * 20))
        print(loops * dt, end='\r')
        time.sleep(max(0,dt-(toc-tic)))
    return averageTime/loops
if __name__ == '__main__':
    #time.sleep(3)
    s = Snake('SA074')
    gains = hw.Gains(s.numModules)
    gains.positionKp = [7 for g in gains.positionKp]
    cmdlist = s.snakeGroup.getCommandList()
    gains.setGains(cmdlist)
    s.snakeGroup.sendCommand(release=True)
    r = Curves.reverseSidewind
    sw = Curves.sidewind
    runCurve(s, loopCount =  10 * 160 / pi, angleFn=None, curve=sw)

