import pickle
import threading
import cv2
import os
import numpy as np

DATA_FOLDER  = '/media/deep-sea-snake/Experiments/data'
VIDEO_FOLDER = 'ExperimentsNew/videos'
DATA_FOLDER  = '/home/guillaume/Deep-SEA-Snake/Experiments/data'
VIDEO_FOLDER = '/home/guillaume/Deep-SEA-Snake/Experiments/videos'

FILE_NAME_FORMAT  = 'Experiment_{}.snake'
VIDEO_NAME_FORMAT = 'Experiment_{}.mkv'


#if not os.path.exists(DATA_FOLDER):
#    os.makedirs(DATA_FOLDER)
#if not os.path.exists(VIDEO_FOLDER):
#    os.makedirs(VIDEO_FOLDER)
class DataLogger:
    def __init__(self, runId = 0, recordVideo = True, videoAddress = 'rtsp://admin:biorobotics@10.10.10.118/', fps = 30.0, resolution = (1920, 1080)):
        self.runData = []
        self.runId = runId
        self.fps = fps
        self.resolution = resolution
        self.recordingVideo = recordVideo
        if self.recordingVideo:
            self.videoAddress = videoAddress
            self._cap = cv2.VideoCapture()
            self._cap.open(self.videoAddress)
            self._fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            threading.Thread(target = self.recordVideo).start()
        else:
            self.videoAdddress = None
    def recordVideo(self):
        print('Starting video capture')
        out = cv2.VideoWriter(os.path.join(VIDEO_FOLDER, VIDEO_NAME_FORMAT.format(self.runId)), self._fourcc, self.fps, self.resolution)
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                out.write(frame)
            else:
                break
        print('Finished up video capture...')

    def logData(self, *args):
        self.runData.append(args)
    def saveData(self):
        print('Saving run data...')
        with open(os.path.join(DATA_FOLDER, FILE_NAME_FORMAT.format(self.runId)), 'wb') as file:
            pickle.dump(self.runData, file)
        print('Data saved.')
        print('Stopping video capture...')
        self._cap.release()
class Struct:
    pass

def compFilter(iterator, initPoints = 5, oldRatio = 0.9):
    initPoints = min(len(iterator), initPoints)
    filteredData = [np.mean(iterator[:initPoints])]
    for dataPoint in iterator:
        filteredData.append(filteredData[-1] * oldRatio + (1 - oldRatio) * dataPoint)
    return filteredData

def medFilter(points, width = 10):
    data = [points[i:] for i in range(width)]
    filteredData = [np.mean(point) for point in zip(*data)]
    return filteredData
def eulerIntegrator(fn, dx = 0.01, C = 0):
    def integrate(f):
        start = C
        points = [start]
        for i in f:
            start += dx * i
            points.append(start)
        return points
    return integrate(fn)
mag = lambda v : np.sqrt(np.sum(v ** 2))
class RunData:
    '''Used to load data from CompliantSnake.py'''
    def __init__(self, fileName):
        fileName = os.path.join(DATA_FOLDER, FILE_NAME_FORMAT.format(fileName))
        with open(fileName, 'rb') as file:
            self.raw = pickle.load(file)
        self.parseData()

    def parseData(self):
        initData, header    = self.raw[:2]
        experimentData      = self.raw[2:]
        self.dataNames      = header[0].split(', ')
        self.initNames      = initData[0].split(', ')

        self.initData       = {name:data for name,data in zip(self.initNames, initData[1:])}
        self.runData        = [{name:data for name, data in zip(self.dataNames, step)} for step in experimentData]

    def extractData(self, key):
        assert key in self.dataNames
        return [step[key] for step in self.runData]

if __name__ == '__main__':
    runToUse = 5048
    dt = np.pi / 160

    run = RunData(runToUse)
    reward = run.extractData('reward')
    rewardCumulative = eulerIntegrator(reward, dx = dt)
    width = 10

    import matplotlib.pyplot as plt
    plt.plot(rewardCumulative[width // 2:])
    plt.plot(medFilter(rewardCumulative, width = width))
    plt.show()
    rc = rewardCumulative
