""" Contains extra functions for doing things like matlab
    As with CompliantSnake.py, only what is required to be changed from the
    original matlab source has been changed. All original comments are
    preserved as comments, and new comments are noted with a double pound
    sign (##) to denote their difference."""
import numpy as np
pi = np.pi
def initialize_Windows(snakeConst, windows):
    '''This function is called once when intializing the windows at the begin-
    ning of both of the two main functions. Windows are essentially an array
    with cells containing parameters of the segment of the snake within that
    particular window. This allows sections of the snake to move independent-
    ly from the rest.

    Explanation of variables:
	 windowsOrigin - a value declared in the main function
			 represents the position where the windows will
			 originate. Important for updating and moving the
			 originating position to different places, which
			 is needed for going backwards.
	 numWindows    - a value declared in the main function
			 represents the number of total number of windows
	 numWaves      - a value declared in the main function
			 represents
	 initialized_windows
		       - the returned value.
			 represents the initial state of the windows_array


    REQUIRES: 0 < numWindows
	   0 < windowsOrigin <= 3          This precondition will be
					   hopefully improved
    ENSURES: initialized_windows(windowsOrigin, numWindows, numWaves) returns
	  a matrix of size 1-by-(numWindows + 1) with the value of at
	  index windowsOrigin 0 and the value of initialized_windows(i) =
	  initialized_windows(i - 1) + 1/ (2 * numWaves)'''


    # Acquire necessary variables from structure
    posOrigin  = - pi / (2 * snakeConst.spFreq)
    origin     = windows.origin
    numWindows = windows.numWindows
    numWaves   = snakeConst.numWaves

    # Initialize the matrix with the proper size
    initialized_windows = np.zeros((1, numWindows + 1));


    ## Note: due to 1 indexing in Matlab, i is replaced with i+1 in equations
    ## the first [0] is because numpy matrices require both positional
    ## arguments
    for i in range(numWindows + 1):
        initialized_windows[0][i] = 1 / (2 * numWaves) * (i + 1 - origin) \
                                                               + 4 * posOrigin

    ## returning output variable
    return initialized_windows

def initialize_AmpWindows(windows):
    '''This function is called once when intializing the spFreq_windows array at
      the beginning of the main function.

      Explanation of variables:
              windows       - the structure that contains all the information
                              surrounding windows.
              amp_windows   - the returned value.
                              represents the initial state of the
                              amp_windows array


      REQUIRES : windows structure is defined for numWindows and windows_array
      ENSURES : returns a matrix of size numWindows-by-2 array such that the
                first element'''
    numWindows          =  windows.numWindows
    array               =  windows.array[0] ## To make one-dimensional

    amp_windows         =  np.zeros((numWindows, 2))
    ## indexes have been decremented by 1
    amp_windows[0]      = [array[0] - (array[1] - array[0])/2, \
			   array[0] + (array[1] - array[0])/2]
    for i in range(1, numWindows):
        amp_windows[i] = [array[i] - (array[i]   - array[i-1])/2, \
			  array[i] + (array[i+1] - array[i])  /2]

    ## returning output variable
    return amp_windows

def initialize_SpFreqWindows(windows):
    ''' This function is called once when intializing the spFreq_windows array at
        the beginning of the main function.

        Explanation of variables:
                windows       - the structure that contains all the information
                                surrounding windows.
                spFreq_windows
                              - the returned value.
                                represents the initial state of the
                                spFreq_windows array


        REQUIRES : windows structure is defined for numWindows and windows_array
        ENSURES : returns a matrix of size numWindows-by-2 array such that the
                  first element in the ith row is the coordinate where the window
                  begins and the second element is the coordinate where the
                  window ends. '''
    numWindows     = windows.numWindows
    windows_array  = windows.array[0] ## To make one-dimensional

    spFreq_windows = np.zeros((numWindows, 2))

    for i in range(numWindows):
        spFreq_windows[i] = [windows_array[i], windows_array[i + 1]]

    ## returning output variable
    return spFreq_windows

def initialize_OffsetWindows(windows):
    ## No description
    return windows.spFreq

def get_NewAngles2D(snakeCompl, snakeConst, windows):
    FUDGEFACTOR = 0.8471    # Something about the snake doing a cos (sin(pi/2 + ...))
                            # but the initial mainAxis is not zero and blabla
                            # something weird...
    numModules        = snakeConst.numModules
    angles_array      = np.zeros((numModules, 1))
    angles_array0     = np.zeros((numModules, 1))
    phi               = snakeCompl.phi

    for i in range(numModules):
        norm        = (i+1) / numModules
        ampNew      = -modulate_AmpSnake(norm, snakeCompl, windows)
        spFreqNew   = modulate_SpFreqSnake(norm, snakeCompl, windows)
        offsetNew   = modulate_OffsetSnake(norm, snakeCompl, windows)
#       torsionNew  = modulate_torsionSnake(norm, snakeCompl, windows);
        angle       = offsetNew + ampNew * np.sin(spFreqNew - phi + pi/2)
#       angle0      = ampNew * sin(spFreqNew - phi + pi/2);

        if (i + 1) % 2 > 0:
#             angle = angle * cos(torsionNew);
#             angle0 = angle0 * cos(torsionNew);

            if (i+1) == 1:
                # defining the head's heading -> used to get main Axis
                mainAxis = ampNew * np.cos(spFreqNew - phi + pi/2 - FUDGEFACTOR)
                currSerpPos = spFreqNew - phi + pi/2
        else:
            angle = 0
#             angle0 = angle0 * sin(torsionNew);
        angles_array[i] = min(pi/2, max(-pi/2, angle))
#         angles_array0(i) = min(pi/2, max(-pi/2, angle0));
    angles_array[-1] = -pi / 2
    angles_array0[-1] = -pi / 2
    shapeRegularity = sum( (angles_array - angles_array0) ** 2 )

    return (angles_array, mainAxis, currSerpPos, shapeRegularity)

def modulate_AmpSnake(norm, snakeCompl, windows):
    amp_snake           = snakeCompl.amp
    amp_windows         = windows.amp
    len                 = windows.numWindows

    slope               = snakeCompl.slope

    dummy_mod           = np.zeros((len, 1))

    for i in range(len):

        sigmoidLeft      = np.exp(slope * (norm - amp_windows[i][1]))/\
                             (1 + np.exp(slope * (norm - amp_windows[i][1])))
        sigmoidRight     = np.exp(slope * (norm - amp_windows[i][0]))/\
                             (1 + np.exp(slope * (norm - amp_windows[i][0])))
        dummy_mod[i]  = (amp_snake[0][i] * (-sigmoidLeft + sigmoidRight))

    modulatedAmp         = sum(dummy_mod)

    return modulatedAmp


def modulate_SpFreqSnake(norm, snakeCompl, windows):
    spFreq_snake        = snakeCompl.spFreq
    spFreq_windows      = windows.spFreq
    len                 = np.size(spFreq_windows, 0)

    slope               = snakeCompl.slope

    dummy_mod           = np.zeros((len, 1))
    for i in range(len):

        sigmoidLeft      = np.exp(slope * (norm - spFreq_windows[i][1]))

        sigmoidRight     = np.exp(slope * (norm - spFreq_windows[i][0]))
        dummy_mod[i]   = spFreq_snake[0][i] / slope * \
                           (-np.log(sigmoidLeft + 1) + np.log(sigmoidRight + 1));

    modulatedSpFreq      = sum(dummy_mod);
    return modulatedSpFreq

def modulate_OffsetSnake(norm, snakeCompl, windows):
# Get information from the structure
    slope           = 200
    offset_windows  = windows.offset
    offset          = snakeCompl.offset

#     disp('snake offset');
#     disp(offset);
    # Initialize the local variables
    len          = np.size(offset, 1)
#     disp('len');
#     disp(len);
    dummy_mod    = np.zeros((len, 1))

    # Scale the offsets to the sigmoid function
    for i in range(len):
#
        sigmoidLeft      = np.exp(slope * (norm - offset_windows[i][1]))/\
                             (1 + np.exp(slope * (norm - offset_windows[i][1])))
        sigmoidRight     = np.exp(slope * (norm - offset_windows[i][0]))/\
                             (1 + np.exp(slope * (norm - offset_windows[i][0])))

        dummy_mod[i]  = offset[0][i] * (-sigmoidLeft + sigmoidRight);

    modulatedOffset = sum(dummy_mod);

    return modulatedOffset


def get_AppliedForce2D(tau_Ext, snakeConst, snakeCompl, windows):
    J           = generate_Jacobian2D(snakeConst, snakeCompl, windows)
    tau_D       = snakeCompl.tau_D

    tau_Applied = tau_D - np.matmul(J.transpose(), np.array([tau_Ext]).transpose())
    ## returning matlab variable
    return tau_Applied

def generate_Jacobian2D(snakeConst, snakeCompl, windows):
    numModules   = snakeConst.numModules
    len          = windows.numWindows
    phi          = snakeCompl.phi
    J_out        = np.zeros((numModules, 2 * len))

    for i in range(numModules):
        norm     = (i + 1) / numModules
        spFreqW  = modulate_SpFreqWindows(norm, snakeCompl, windows)
        ampW     = modulate_AmpWindows(norm, snakeCompl, windows)
        spFreqS  = modulate_SpFreqSnake(norm, snakeCompl, windows)
        ampS     = modulate_AmpSnake(norm, snakeCompl, windows)
#       torS     = modulate_torsionSnake(norm, snakeCompl, windows);

        ## adding 1 to use of i in for loop due to 1 indexed matlab arrays
        if (i + 1) % 2:
            J_out[i] =  np.concatenate((ampW * np.sin(spFreqS + pi/2 - phi), \
                                    spFreqW * ampS * np.cos(spFreqS + pi/2 - phi))).transpose()
#                          ampS * -sin(torS) * sin(spFreqS + pi/2 - phi) * ones(len,1)];
        else:
            J_out[i] = np.concatenate((np.zeros((len,1)), np.zeros((len,1)))).transpose()
#                          ampS * cos(torS) * sin(spFreqS + pi/2 - phi) * ones(len,1)];
    ## returning matlab variable
    return J_out
def modulate_SpFreqWindows(norm, snakeCompl, windows):
    spFreq_windows  = windows.spFreq
    len             = windows.numWindows

    slope           = snakeCompl.slope
    modulatedSpFreq = np.zeros((len,1))

    for i in range(1,len):
        sigmoidLeft     = np.exp(slope * (norm - spFreq_windows[i][1]))/ \
                            (1 + np.exp(slope * (norm - spFreq_windows[i][1])))
        sigmoidRight    = np.exp(slope * (norm - spFreq_windows[i][0])) / \
                            (1 + np.exp(slope * (norm - spFreq_windows[i][0])))

        modulatedSpFreq[i] = -sigmoidLeft + sigmoidRight
    ## returning matlab variable
    return modulatedSpFreq
def modulate_AmpWindows(norm, snakeCompl, windows):
    amp_windows         = windows.amp
    len                 = windows.numWindows

    slope               = snakeCompl.slope

    modulatedAmp        = np.zeros((len, 1))

    for i in range(len):

        sigmoidLeft     = np.exp(slope * (norm - amp_windows[i][1]))/ \
		            (1 + np.exp(slope * (norm - amp_windows[i][1])))
        sigmoidRight    = np.exp(slope * (norm - amp_windows[i][0]))/ \
	    	        (1 + np.exp(slope * (norm - amp_windows[i][0])))

        modulatedAmp[i]  = -sigmoidLeft + sigmoidRight

    ## set nopaste
    return modulatedAmp
def get_NewNomParam2D(snakeCompl, snakeNom, snakeConst, windows, tau_Applied, dt):
    compliantQ = True
    steerQ = False

    len               = windows.numWindows

    MdPrime           = snakeNom.MdPrime
    BdPrime           = snakeNom.BdPrime
    KdPrime           = snakeNom.KdPrime

    index             = windows.steerIndex
    sigma_Nom         = snakeNom.sigma_Nom
    sigma_D           = snakeNom.sigma_D
    dsigmaD_dt        = snakeNom.dsigmaD_dt
    ## no steer for you
    '''
    if steerQ
        steer1 = -pi/2; steer2 = 0; steer3 = pi/2;

        if(index > 0)
            alphaC = 1.; Kh = 3.;
            sigma_D(index)   = sigma_D(index) + dt * (- alphaC * (sigma_D(index) - steer1).*(sigma_D(index) - steer2).*(sigma_D(index) - steer3) + Kh * (-snakeNom.h/2 - sigma_D(index)));
            sigma_Nom(index) = sigma_Nom(index) + dt * (- alphaC * (sigma_Nom(index) - steer1).*(sigma_Nom(index) - steer2).*(sigma_Nom(index) - steer3) + Kh * (-snakeNom.h/2 - sigma_Nom(index)));
            snakeNom.steeringOffset = sigma_Nom(index);
    '''
    ##
    if compliantQ:
        d2sigma_dt2         = np.linalg.lstsq(MdPrime, (tau_Applied - np.matmul(BdPrime, dsigmaD_dt) - np.matmul(KdPrime, (sigma_D[len:-1] - sigma_Nom[len:-1]))))[0]
        dsigmaD_dt          = dsigmaD_dt + d2sigma_dt2 * dt
        sigma_D[len:-1]    = sigma_D[len:-1] + dsigmaD_dt  * dt
        snakeNom.dsigmaD_dt = dsigmaD_dt

    offset            = sigma_D[          0:     len]
    amp               = sigma_D[    len    : 2 * len]
    spFreq            = sigma_D[2 * len    : 3 * len]
#     torsion           = sigma_D(3 * len + 1: 4 * len);
    tmFreq            = snakeCompl.tmFreq + snakeConst.tmFreq * dt
    if any(spFreq <= 0.01):
        for i,f in enumerate(spFreq):
            if f <= 0.01:
                spFreq[i] = 0.01

    snakeCompl.offset   = offset.transpose()
    snakeCompl.amp      = amp.transpose() ## transpose is to put amp back into standard form
    snakeCompl.spFreq   = spFreq.transpose()
#     snakeCompl.torsion  = torsion;
    snakeCompl.tmFreq   = tmFreq

    snakeNom.sigma_Nom  = sigma_Nom
    snakeNom.sigma_D    = sigma_D
    return (snakeCompl, snakeNom)
def update_Windows(snakeCompl, windows):
    numWindows      = windows.numWindows
    array_windows   = windows.array
    spFreq_snake    = snakeCompl.spFreq

    for i in range(numWindows):
        array_windows[0][i + 1] = array_windows[0][i] + pi / spFreq_snake[0][i]

    windows.array = array_windows

    ## returning matlab variable
    return windows
def update_Structures2D(snakeCompl, snakeConst, snakeNom, windows):
    windows_array   = windows.array.transpose()
    len             = windows.numWindows

    offset_snake    = snakeCompl.offset
    amp_snake       = snakeCompl.amp.transpose()
    spFreq_snake    = snakeCompl.spFreq.transpose()
#%     torsion_snake      = snakeCompl.torsion
    tauD            = snakeCompl.tau_D

    sigma_Nom       = snakeNom.sigma_Nom
    sigma_D         = snakeNom.sigma_D
    dsigmaD_dt      = snakeNom.dsigmaD_dt

    if (windows.spFreq[0][0] > -0.333333333):
        windows_array  = np.concatenate(([windows_array[0]-abs(windows_array[0]) - windows_array[1]], windows_array[1:]))
        amp_snake = np.concatenate(([amp_snake[0]], amp_snake[1:]))
        spFreq_snake = np.concatenate(([spFreq_snake[0]], spFreq_snake[1:]))

#         torsion_snake   =  [ torsion_snake(1);      torsion_snake(1 : end-1)];
        tauD           = np.concatenate(([tauD[0]],tauD[:len-1],[tauD[len]],tauD[len:2*len-1]))

##np.concatenate(([tauD[0]],tauD[1:len-2],[tauD[len]],tauD[len+1:2*len-2]))
#                                  tauD(2*len+1);      tauD(2*len+1 : 3*len-1)];

        sigma_Nom      =  np.concatenate(([sigma_Nom[0]], sigma_Nom[0:len-1], [sigma_Nom[len]],sigma_Nom[len:2*len-1],[sigma_Nom[2*len]], sigma_Nom[2*len:3*len-1], [sigma_Nom[-1]]))

#                             sigma_Nom(3 * len+1);   sigma_Nom(3 * len+1 : 4*len-1); ...
        sigma_D        = np.concatenate(([sigma_D[0]], sigma_D[0:len-1], [sigma_D[len]], sigma_D[len:2*len-1],[sigma_D[2*len]], sigma_D[2*len:3*len-1], [sigma_D[-1]]))
##np.concatenate((sigma_D[0], sigma_D[1:len-2], sigma_D[len], sigma_D[len:2*len-2],sigma_D[2*len], sigma_D[2*len+1:3*len-2]))

#                               sigma_D(3 * len+1);     sigma_D(3 * len+1 : 4*len - 1); ...
        dsigmaD_dt     = np.concatenate(([dsigmaD_dt[0]], dsigmaD_dt[0:len-1],[dsigmaD_dt[len]],dsigmaD_dt[len:2*len-1]))
##np.concatenate((disgmaD-dt[0], dsigmaD_dt[1:len-2],dsigmaD_dt[len], dsigmaD_dt[len:2*len-2]))
        snakeCompl.phi = snakeCompl.tmFreq;

    windows.array       = windows_array.transpose()

    windows.offset      = update_OffsetWindows(windows)
    windows.spFreq      = update_SpFreqWindows(snakeCompl, snakeConst, windows)
    windows.amp         = update_AmpWindows(snakeCompl, snakeConst, windows)

    snakeCompl.offset   = offset_snake
    snakeCompl.amp      = amp_snake.transpose()
    snakeCompl.spFreq   = spFreq_snake.transpose()
    #     snakeCompl.torsion  = torsion_snake;
    snakeCompl.tau_D    = tauD

    snakeNom.sigma_Nom  = sigma_Nom
    snakeNom.sigma_D    = sigma_D
    snakeNom.dsigmaD_dt = dsigmaD_dt

    ## returning
    return (snakeCompl, snakeNom, windows)

def update_OffsetWindows(windows):
    return windows.spFreq

def update_SpFreqWindows(snakeCompl, snakeConst, windows):
    numWindows     = windows.numWindows
    windows_array  = windows.array[0]
    phi            = snakeCompl.tmFreq
    spFreq         = snakeConst.spFreq

    spFreqNew      = np.zeros((numWindows, 2))

    for i in range(numWindows):
        spFreqNew[i] = np.array([windows_array[i], windows_array[i + 1]]) + phi / spFreq

    return spFreqNew

def update_AmpWindows(snakeCompl, snakeConst, windows):
    numWindows     = windows.numWindows
    array  = windows.array[0]
    phi            = snakeCompl.tmFreq
    spFreq         = snakeConst.spFreq

    ampNew = np.zeros((numWindows,2))
    ampNew[0] = np.array([array[0] - pi/(2 * spFreq), array[0] + ((array[1]-array[0])/2)]) + phi/spFreq

    for i in range(numWindows-1):
        ampNew[i+1] = np.array([array[i] + ((array[i+1]-array[i])/2),array[i+1] + ((array[i+2]-array[i+1])/2)]) + phi/spFreq


    return ampNew
