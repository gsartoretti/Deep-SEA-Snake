import NatNet as NN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

def mag(array):
#    return np.sqrt(np.sum(array ** 2))
    return np.sum(array ** 2) # to ensure consistency with SnakeEnvironment.py

class Buffer:
    def __init__(self, size):
        self.size = size
        self.bufferdata = deque([])
    def _pop(self):
        if len(self.bufferdata) > 0:
            return self.bufferdata.popleft()
        return None
    def empty(self):
        return len (self.bufferdata) == 0
    def peek(self):
        return self.bufferdata[0]
    def popBlock(self):
        out = self._pop()
        while out == None:
            out = self._pop()
        return out
    def popNoBlock(self):
        out = self._pop()
        if self.empty():
            self.push(out)
        return out
    def push(self, data):
        self.bufferdata.append(data)
        if len(self.bufferdata) > self.size:
            self.bufferdata.popleft()


class Reward:
    def __init__(self, bufferSize = 5, episodeLength = 89, rewardFactor = 10):
        self.client = NN.NatNetClient()
        self.obuffer = Buffer(bufferSize)
        self.pbuffer = Buffer(episodeLength)
        self.client.newFrameListener = self.bufferedRecieveNewFrame(self.obuffer)
        self.rewardFactor = rewardFactor
        self.client.run()
        self.setOrigin()
    def setOrigin(self):
        count, self.origin = self.obuffer.popBlock()
        self.pbuffer.push(self.origin)
    def bufferedRecieveNewFrame(self, outputBuffer):
        def recieveNewFrame(frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount, labeledMarkerCount, latency, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged, unlabeledMarkers):
            outputBuffer.push((unlabeledMarkersCount,np.mean(unlabeledMarkers, axis = 0)))
        return recieveNewFrame
    def getReward(self):
        oldpos = self.pbuffer.peek()
        count, currentpos = self.obuffer.popNoBlock()
        self.pbuffer.push(currentpos)

        reward = np.tanh(self.rewardFactor * (mag(currentpos - self.origin) - mag(oldpos - self.origin)))
        #reward = mag(currentpos - self.origin) - mag(oldpos - self.origin)

        return reward, (currentpos, oldpos)

if __name__ == '__main__':
    '''streamingClient = NN.NatNetClient()

    dataBuffer = Buffer(30)

    streamingClient.newFrameListener = Reward.bufferedRecieveNewFrame(None, dataBuffer)

    streamingClient.run()

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d')
    ax.set_xlim(-0.6, 0)
    ax.set_ylim(0, 0.6)
    ax.set_zlim(-1, 1)

    pts = ax.scatter([0],[0],[0])
    plt.ion()
    print(dataBuffer.peek())
    while True:
        plt.pause(0.01)
        c, points = dataBuffer.popBlock()
        if c > 0:
            print(list(points))
            x, z, y = (points)
            pts.remove()
            pts = ax.scatter(x, y, z, c='b', s = 500)
            plt.draw()
            print('\n{} markers in frame'.format(c))'''

    import time
    reward = Reward(bufferSize = 5, episodeLength = 1, rewardFactor = 100)
    time.sleep(0.2)
    while True:
        print(reward.getReward())
        time.sleep(0.02)
