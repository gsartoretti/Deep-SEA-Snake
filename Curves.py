from math import pi, sin
from copy import copy
LATERAL = 0
DORSAL  = 1
class SerpenoidCurve:
    lateralAngularOffset    = 0
    lateralAmplitude        = 0
    lateralTemporalFreq     = 0
    lateralSpacialFreq      = 0
    dorsalAngularOffset     = 0
    dorsalAmplitude         = 0
    dorsalTemporalFreq      = 0
    dorsalSpacialFreq       = 0
    phaseShift              = 0


sidewind = SerpenoidCurve()
sidewind.lateralAngularOffset       = 0
sidewind.lateralAmplitude           = 0.5
sidewind.lateralTemporalFrequency   = 2
sidewind.lateralSpacialFrequency    = 2 * pi
sidewind.dorsalAngularOffset        = 0
sidewind.dorsalAmplitude            = 0.5
sidewind.dorsalTemporalFrequency    = 2
sidewind.dorsalSpacialFrequency     = 2 * pi
sidewind.phaseShift                 = 3 * pi / 4

reverseSidewind = copy(sidewind)
reverseSidewind.phaseShift = 5 * pi / 4


def calculateAngle(curve, t, distanceToHead, moduleType):
    if moduleType == LATERAL:
        angle = curve.lateralAngularOffset + curve.lateralAmplitude * \
                sin(curve.lateralSpacialFrequency * distanceToHead -  \
                    curve.lateralTemporalFrequency * t)
    else:
        angle =  curve.dorsalAngularOffset + curve.dorsalAmplitude * \
                sin(curve.dorsalSpacialFrequency * distanceToHead -  \
                    curve.dorsalTemporalFrequency * t + curve.phaseShift)
    return angle

