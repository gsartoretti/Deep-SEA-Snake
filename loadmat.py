import scipy.io
import os
loadmat = scipy.io.loadmat

def loadFile(filename):
    mat = loadmat(filename)
    """
    assert len(mat['mXpos']) == len(mat['mYpos'])
    assert len(mat['mYpos']) == len(mat['times'])
    assert len(mat['times']) == len(mat['amps'])
    assert len(mat['amps']) == len(mat['spFreqs'])
    """
    mat['mYpos'] = list(mat['mYpos'].flatten())
    mat['mXpos'] = list(mat['mXpos'].flatten())
    mat['times'] = list(mat['times'].flatten())
    #mat['spFreqs'] = list(mat['spFreqs'].flatten())
    #mat['amptorques'] = list(mat['amptorques'].flatten())
    #mat['freqtorques'] = list(mat['freqtorques'].flatten())
    experiment = []
    dts = min(len(mat['mXpos']),
                       len(mat['mYpos']),
                       len(mat['times']),
                       len(mat['amps']),
                       len(mat['spFreqs']),
                       len(mat['amptorques']),
                       len(mat['freqtorques']))
    for i in range(dts):
        experiment.append({'modular_time':mat['times'][i],
                           'snake_shape':[[mat['amps'][i][j],mat['spFreqs'][i][j]] for j in range(len(mat['amps'][i]))],
                           'torques':[[mat['amptorques'][i][j],mat['freqtorques'][i][j]] for j in range(len(mat['amptorques'][i]))],
                           'position':[mat['mXpos'][i],mat['mYpos'][i]]})
        print('{}/{}'.format(i+1,dts) + ' ' * 20, end='\r')
    print('')
    return experiment

def loadAllExperiments(directory):
    exps = []
    files = os.listdir(directory)
    for i,file in enumerate(files):
        exps.append(loadFile(os.path.join(directory,file)))
        print('Loaded file {}/{}'.format(i+1,len(files)))
    return exps
