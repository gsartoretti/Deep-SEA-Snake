import ctypes
import time
from platform import system
from math import pi
from ctypes import c_int
from HebiConstants import *


platform = system()
if platform == 'Windows':
    hebi = ctypes.cdll.LoadLibrary('./hebi.dll')
else:
    if platform != 'Linux':
        print('System not recognized, attempting to use linux library')
    hebi = ctypes.cdll.LoadLibrary("./libhebi.so.{}".format(API_VERSION)) # change this to libhebi.so.[version number]


MAC_ADDRESS = ctypes.c_byte * 8 # change this 8 if the number of bytes in a mac address ever changes
class HebiMacAddress(ctypes.Structure):
    _fields_ = [('bytes_', MAC_ADDRESS)]

hebi.hebiLookupEntryListGetMacAddress.restype = HebiMacAddress
hebi.hebiInfoGetString.restype = ctypes.c_char_p
hebi.hebiFeedbackGetFloat.restype = ctypes.c_float
hebi.hebiInfoGetFloat.restype = ctypes.c_float



class HebiPointer:
    addr = 0
    def __init__(self, addr):
        self.addr = addr
    def __del__(self):
        self.close()
    def close(self):
        pass
class HebiModule(HebiPointer):
    r = 0
    name = ''
    family = ''
    command = None
    def __init__(self, name, family, addr = 0):
        self.name = name
        self.family = family
        HebiPointer.__init__(self, addr)
    def close(self):
        #hebi.hebiReleaseModule(self.addr)
        if self.command:
            self.command.close()
    def sendCommand(self, command):
        if hebi.hebiModuleSendCommand(self.addr, command.addr):
            return False
        return True
    def setAngle(self, angle):
        if not self.command:
            self.command = HebiCommand()
        self.command.setAngle(angle)
        self.sendCommand(self.command)

class HebiCommand(HebiPointer):
    def __init__(self):
        HebiPointer.__init__(self, hebi.hebiCommandCreate())
    def close(self):
        hebi.hebiCommandRelease(self.addr)
    def clearAngle(self):
        hebi.hebiCommandClearHighResAngle(self.addr, Commands.CommandHighResAnglePosition)

    def setAngle(self, angle):
        angle = min(max(angle, -pi/2 +0.05), pi/2-0.05)
        decPart = ctypes.c_float(angle)
        hebi.hebiCommandSetHighResAngle(self.addr, Commands.CommandHighResAnglePosition, ctypes.c_int(), decPart)
    def setTorque(self, torque, maxTorque = 4.5, minTorque = -4.5):
        torque = min(max(float(torque), -4.5), 4.5)
        self.setField(Commands.CommandFloatTorque, torque)
    def setField(self, field, value):
        hebi.hebiCommandSetFloat(self.addr, field, ctypes.c_float(value))
    def setEnum(self, field, value):
        hebi.hebiCommandSetEnum(self.addr, field, ctypes.c_int(value))
    def getField(self, field):
        if hebi.hebiCommandHasFloat(self.addr, field):
            return hebi.hebiCommandgetFloat(self.addr, field)
        else:
            return None

class HebiModuleInfo(HebiPointer):
    def getFamily(self):
        if self.addr == 0:
            return None
        if not hebi.hebiInfoHasString(self.addr, Info.InfoStringFamily):
            return None

        buf = ctypes.create_string_buffer(256)
        # problem with C API, max string size is 256
        #bufLen = hebi.hebiInfoGetString(self.addr, Info.InfoStringFamily, buf, 1)
        #buf = ctypes.create_string_buffer(bufLen)
        bufLen = 256
        if hebi.hebiInfoGetString(self.addr, Info.InfoStringFamily, buf, bufLen):
            print('Something went wrong getting family info for module')
            return None
        return buf.value.decode()
    def getName(self):
        if self.addr == 0:
            return None
        if not hebi.hebiInfoHasString(self.addr, Info.InfoStringName):
            return None

        buf = ctypes.create_string_buffer(256)
        #bufLen = hebi.hebiInfoGetString(self.addr, Info.InfoStringName, buf, 1)
        #buf = ctypes.pointer(ctypes.create_string_buffer(bufLen))
        bufLen = 256
        if hebi.hebiInfoGetString(self.addr, Info.InfoStringName, buf, bufLen):
            print('Something went wrong getting name info for module')
            return None
        return buf.value.decode()
    def getField(self, field):
        if self.addr == 0:
            return None
        if not hebi.hebiInfoHasFloat(self.addr, field):
            return None
        return hebi.hebiInfoGetFloat(self.addr, field)

class HebiModuleFeedback(HebiPointer):
    def getPosition(self):
        if self.addr == 0:
            print('NULL pointer')
            return None
        if not hebi.hebiFeedbackHasHighResAngle(self.addr, Info.FeedbackHighResAnglePosition):
            print('No angle found')
            return None
        ipart = ctypes.pointer(ctypes.c_int())
        fpart = ctypes.pointer(ctypes.c_float())
        hebi.hebiFeedbackGetHighResAngle(self.addr, Info.FeedbackHighResAnglePosition, ipart, fpart)
        return (ipart.contents.value, fpart.contents.value)
    def getTorque(self):
        if self.addr == 0:
            return None
        if not hebi.hebiFeedbackHasFloat(self.addr, Info.FeedbackFloatTorque):
            return None
        return hebi.hebiFeedbackGetFloat(self.addr, Info.FeedbackFloatTorque)
class Gains:
    addr = None
    ones_n = lambda numModules : lambda a : [a for x in range(numModules)]
    def __init__(gains, numModules):
        ones_n = Gains.ones_n(numModules)
        gains.controlStrategy           = ones_n(4);
        gains.positionKp                = ones_n(4.0);
        gains.positionKi                = ones_n(0.01);
        gains.positionKd                = ones_n(1.0);
        gains.torqueKp                  = ones_n(1.0);
        gains.torqueKi                  = ones_n(0.0);
        gains.torqueKd                  = ones_n(0.1);
        gains.torqueMaxOutput           = ones_n(2.25);
        gains.torqueMinOutput           = [-x for x in gains.torqueMaxOutput];
        gains.positionIClamp            = ones_n(1.0);
        gains.velocityKp                = ones_n(1.0);
        gains.positionMaxOutput         = ones_n(10.0);
        gains.positionMinOutput         = ones_n(-10.0);
        gains.torqueOutputLowpassGain   = ones_n(0.5);
        gains.torqueFeedForward         = ones_n(0.15);
    def getGains(group, timeout = 1000):
        """ Static method which returns a Gains object with values equal to the gains
        in the given SEA group. group should be a HebiGroup"""
        moduleInfo = group.getModuleInfo(timeout=timeout)
        g = Gains(len(moduleInfo))
        for i, mi in enumerate(moduleInfo):
            g.positionKp[i] = mi.getField(Info.InfoFloatPositionKp)
            g.positionKi[i] = mi.getField(Info.InfoFloatPositionKi)
            g.positionKd[i] = mi.getField(Info.InfoFloatPositionKd)
            g.torqueKp[i] = mi.getField(Info.InfoFloatTorqueKp)
            g.torqueKi[i] = mi.getField(Info.InfoFloatTorqueKi)
            g.torqueKd[i] = mi.getField(Info.InfoFloatTorqueKd)
            g.torqueMaxOutput[i] = mi.getField(Info.InfoFloatTorqueMaxOutput)
            g.torqueMinOutput[i] = mi.getField(Info.InfoFloatTorqueMinOutput)
            g.positionIClamp[i] = mi.getField(Info.InfoFloatPositionIClamp)
            g.velocityKp[i] = mi.getField(Info.InfoFloatVelocityKp)
            g.positionMaxOutput[i] = mi.getField(Info.InfoFloatPositionMaxOutput)
            g.positionMinOutput[i] = mi.getField(Info.InfoFloatPositionMinOutput)
            g.torqueOutputLowpassGain[i] = mi.getField(Info.InfoFloatTorqueOutputLowpass)
            g.torqueFeedForward[i] = mi.getField(Info.InfoFloatTorqueFeedForward)

            for i in range(39):
                s = str(mi.getField(i))
                if len(s) > 4:
                    s = s[:4]
                if len(s) < 4:
                    s = '{}{}'.format(s, ' ' * (4 - len(s)))
                print(s, end=' ')
            print()

        return g
    def setGains(self, commandList):
        for i, command in enumerate(commandList):
            if command:
                self.addr = command
                HebiCommand.setField(self, Commands.CommandFloatPositionKp, self.positionKp[i])
                HebiCommand.setField(self, Commands.CommandFloatPositionKi, self.positionKi[i])
                HebiCommand.setField(self, Commands.CommandFloatPositionKd, self.positionKd[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueKp, self.torqueKp[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueKi, self.torqueKi[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueKd, self.torqueKd[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueMaxOutput, self.torqueMaxOutput[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueMinOutput, self.torqueMinOutput[i])
                HebiCommand.setField(self, Commands.CommandFloatPositionIClamp, self.positionIClamp[i])
                HebiCommand.setField(self, Commands.CommandFloatVelocityKp, self.velocityKp[i])
                HebiCommand.setField(self, Commands.CommandFloatPositionMaxOutput, self.positionMaxOutput[i])
                HebiCommand.setField(self, Commands.CommandFloatPositionMinOutput, self.positionMinOutput[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueOutputLowpass, self.torqueOutputLowpassGain[i])
                HebiCommand.setField(self, Commands.CommandFloatTorqueFeedForward, self.torqueFeedForward[i])
                HebiCommand.setEnum(self, Commands.CommandEnumControlStrategy, self.controlStrategy[i])
                self.addr = None

class HebiGroup(HebiPointer):
    infoPtr = None
    command = None
    fbkPtr = None
    rootName = ''
    rootFamily = ''
    def __init__(self, rootName, rootFamily, addr = 0):
        self.rootName = rootName
        self.rootFamily = rootFamily
        HebiPointer.__init__(self, addr)
    def __del__(self):
        self.close()
    def close(self):
        #if self.addr:
        #    hebi.hebiReleaseGroup(self.addr)
        if self.infoPtr:
            hebi.hebiGroupInfoRelease(self.infoPtr)
        if self.command:
            hebi.hebiGroupCommandRelease(self.command)
    def numModules(self):
        return hebi.hebiGroupGetNumberOfModules(self.addr)
    def sendCommand(self, command=None, release = False):
        if not command:
            if not self.command:
                return False
            command = self.command
        if hebi.hebiGroupSendCommand(self.addr, command) == 0:
            if release:
                hebi.hebiGroupCommandRelease(self.command)
                self.command = 0
            return True
        return False
    def getCommandList(self):
        cmdList = []
        if not self.command:
            self.command = hebi.hebiGroupCommandCreate(self.numModules())
        for i in range(self.numModules()):
            cmd = hebi.hebiGroupCommandGetModuleCommand(self.command, i)
            cmdList.append(cmd)
        return cmdList
    def setAngles(self, angles, send = True):
        if len(angles) != self.numModules():
            return False
        if not self.command:
            self.command = hebi.hebiGroupCommandCreate(len(angles))
        command = self.command
        if command == 0:
            return False
        for i, angle in enumerate(angles):
            if angle != None:
                moduleCommand = hebi.hebiGroupCommandGetModuleCommand(command, i)
                dummyWrapper = HebiPointer(moduleCommand)
                HebiCommand.setAngle(dummyWrapper, angle)
        if send:
            self.sendCommand()
    def setTorques(self, torques, send = True):
        if len(torques) != self.numModules():
            return False
        if not self.command:
            self.command = hebi.hebiGroupCommandCreate(len(torques))
        command = self.command
        if command == 0:
            return False
        for i, torque in enumerate(torques):
            moduleCommand = hebi.hebiGroupCommandGetModuleCommand(command, i)
            if torque != None:
                dummyWrapper = HebiPointer(moduleCommand)
                dummyWrapper.setField = lambda field, value : HebiCommand.setField(dummyWrapper, field, value)
                HebiCommand.setTorque(dummyWrapper, torque)
        if send:
            self.sendCommand()
    def getAngles(self, timeout = 15):
        moduleFeedbacks = self.getModuleFeedback(timeout = timeout)
        extractPosition = lambda moduleFbk : moduleFbk.getPosition()[1]
        return list(map(extractPosition, moduleFeedbacks))

    def getTorques(self, timeout = 15):
        moduleFeedbacks = self.getModuleFeedback(timeout = timeout)
        extractTorque = lambda moduleFbk : moduleFbk.getTorque()
        return list(map(extractTorque, moduleFeedbacks))
    def clearAngles(self):
        command = hebi.hebiGroupCommandCreate(self.numModules())
        if command == 0:
            return False
        for i in range(self.numModules()):
            moduleCommand = hebi.hebiGroupCommandGetModuleCommand(command, i)
            dummyWrapper = HebiPointer(moduleCommand)
            HebiCommand.clearAngle(dummyWrapper)
        r = self.sendCommand(command)
        hebi.hebiGroupCommandRelease(command)
        return r

    def getModules(self, timeout=1000):
        info = self.getModuleInfo(timeout=timeout)
        modules = []
        if info:
            for module in range(self.numModules()):
                modules.append((info[module].getName(), info[module].getFamily()))
        return modules
    def getModuleInfo(self, timeout=1000):
        if self.addr == 0:
            return None
        if not self.infoPtr:
            self.infoPtr = hebi.hebiGroupInfoCreate(self.numModules())
        hebi.hebiGroupRequestInfo(self.addr, self.infoPtr, timeout)
        time.sleep(0.1)
        modules = []
        for i in range(self.numModules()):
            addr = hebi.hebiGroupInfoGetModuleInfo(self.infoPtr, i)
            modules.append(HebiModuleInfo(addr))
        return modules
    def getModuleFeedback(self, timeout=15):
        if self.addr == 0:
            return None
        if not self.fbkPtr:
            self.fbkPtr = hebi.hebiGroupFeedbackCreate(self.numModules())
        recievedFeedback = False
        while not recievedFeedback:
            hebi.hebiGroupRequestFeedback(self.addr, self.fbkPtr, timeout)
            modules = []
            failed = False
            for i in range(self.numModules()):
                addr = hebi.hebiGroupFeedbackGetModuleFeedback(self.fbkPtr, i)
                position = HebiModuleFeedback(addr).getPosition()
                if position == None:
                    failed = True
                    break
                modules.append(HebiModuleFeedback(addr))
            if not failed:
                recievedFeedback = True
            else:
                time.sleep(0.1)
        return modules
    def setFeedbackFrequency(self, hz):
        if self.addr == 0:
            return None
        hebi.hebiGroupSetFeedbackFrequencyHz(self.addr, ctypes.c_float(hz))


class LookupEntryList(HebiPointer):
    def close(self):
        if self.addr:
            hebi.hebiLookupEntryListRelease(self.addr)
    def getNumEntries(self):
        return hebi.hebiLookupEntryListGetNumberOfEntries(self.addr)
    def getName(self, index):
        buf = ctypes.create_string_buffer(1)
        bufLen = hebi.hebiLookupEntryListGetName(self.addr, index, buf, 0)
        buf = ctypes.create_string_buffer(bufLen)
        if hebi.hebiLookupEntryListGetName(self.addr, index, buf, bufLen) != 0:
            print('Something went wrong getting name for module {}'.format(index))
            return None
        return buf.value.decode()
    def getFamily(self, index):
        buf = ctypes.create_string_buffer(1)
        bufLen = hebi.hebiLookupEntryListGetFamily(self.addr, index, buf, 0)
        buf = ctypes.create_string_buffer(bufLen)
        if hebi.hebiLookupEntryListGetFamily(self.addr, index, buf, bufLen) != 0:
            print('Something went wrong getting family for module {}'.format(index))
            return None
        return buf.value.decode()
    def getMac(self,index): # Not working under windows
        return hebi.hebiLookupEntryListGetMacAddress(self.addr, index)

    def getNamesAndFamilies(self):
        modules = []
        for module in range(self.getNumEntries()):
            modules.append((self.getName(module), self.getFamily(module)))
        return modules

class HebiLookup:
    def __init__(self):
        self.active = False
        self.createLookup()
    def __del__(self):
        if self.hebiLookupPtr:
            self.destroyLookup()
    def destroyLookup(self):
        if self.active:
            hebi.hebiLookupRelease(self.hebiLookupPtr)
            self.active = False
    def createLookup(self):
        if not self.active:
            self.hebiLookupPtr = hebi.hebiLookupCreate()
            if self.hebiLookupPtr:
                self.active = True
                time.sleep(0.1)
                self.printLookupTable()
            else:
                return None
    def printLookupTable(self):
            hebi.hebiPrintLookupTable(self.hebiLookupPtr)
    def getLookupEntryList(self):
        return LookupEntryList(hebi.hebiLookupCreateLookupEntryList(self.hebiLookupPtr))
    def getModuleFromName(self, name, family = '*', timeout=1000):
        if self.active:
            if family == '*':
                lookupList = self.getLookupEntryList()
                modules = lookupList.getNamesAndFamilies()
                module = lambda m : (m[0] == name)
                matchedModules = list(filter(module, modules))
                if len(matchedModules) == 0:
                    print('No module with name {} found!'.format(name))
                    return None
                name, family = matchedModules[0]
            ptrWrapper = HebiModule(name, family)
            name = name.encode()
            family = family.encode()
            ptrWrapper.addr = hebi.hebiCreateModuleFromName(self.hebiLookupPtr, name, family, timeout)
            if ptrWrapper.addr == 0:
                return None
            return ptrWrapper
        else:
            return None

    def getConnectedGroupFromName(self, name, family = '*', timeout=1000, cmdLifetime = 3000):
        if self.active:
            if family == '*':
                lookupList = self.getLookupEntryList()
                modules = lookupList.getNamesAndFamilies()
                module = lambda m : (m[0] == name)
                matchedModules = list(filter(module, modules))
                if len(matchedModules) == 0:
                    print('No module with name {} found!'.format(name))
                    return None
                name, family = matchedModules[0]
            ptrWrapper = HebiGroup(name, family)
            name = name.encode()
            family = family.encode()
            ptrWrapper.addr = hebi.hebiCreateConnectedGroupFromName(self.hebiLookupPtr, name, family, timeout)
            if ptrWrapper.addr == 0:
                return None
            hebi.hebiGroupSetCommandLifetime(ptrWrapper.addr, cmdLifetime)
            return ptrWrapper
        else:
            return None

if __name__ == '__main__':
    hebiLookup = HebiLookup()
    lt = hebiLookup.getLookupEntryList()
    module1 = lt.getNamesAndFamilies()[0]
    print(module1, lt.getNumEntries())
    group = hebiLookup.getConnectedGroupFromName('SA055', cmdLifetime = 10000)
    #g = Gains(group.numModules())
    #cmdList = group.getCommandList()
    #g.setGains(cmdList)
    #group.sendCommand()
    info  = group.getModuleInfo()
    g = Gains(2)
    cmdList = group.getCommandList()
    g.setGains(cmdList)
    group.sendCommand()
    from math import sin, cos, pi, sqrt
    import numpy as np
    group.setAngles([0,0])
    print('here')
    g0 = Gains.getGains(group.getModuleInfo())
    print(g0.positionKp)
    time.sleep(5)
    i = 0.00
    while i < 2 * pi:
        i+= 0.01
        group.setAngles([sin(i),cos(i)])
        time.sleep(0.005)
    group.setAngles([None, None])
    group.setTorques([0,0])
    for k in vars(g0):
        s0 = np.array(vars(g0)[k])
        s1 = np.array(vars(g)[k])
        if sqrt(sum((s0-s1) ** 2)) > 0:
            print('{} :\n{}\n{}\n'.format(k,vars(g0)[k],vars(g)[k]))
            #module = hebiLookup.getModuleFromName('SA079', 'SEA_D03')
    #module2 = hebiLookup.getModuleFromName('SA059', 'S5-3')
    #module3 = hebiLookup.getModuleFromName('SA037', 'SEA_D03')
    #fourGroup = hebiLookup.getConnectedGroupFromName('SA079', 'SEA_D03')
    #if None in (module, module2, module3, fourGroup):
    #    print('Module not found')
    #    exit()
    #c = HebiCommand()
    #mi = fourGroup.getModuleFeedback()
    #p = lambda f : f.getPosition()[1]
    #while True:
    #    pos = p(mi[0])
    #    print(pos)
    #    module.setAngle(pos)
    #    mi = fourGroup.getModuleFeedback()
    #    if not mi:
    #        print('oops')
    #    time.sleep(0.05)
    #angle = 0
    #from math import sin,cos
    #while True:
    #    angles = [sin(angle), 0, sin(angle)-0.3]
    #    fourGroup.setAngles(angles)
    #    angle += 0.005
    #    print(angles)
    #    time.sleep(0.01)
    ##c.setField(Commands.CommandFloatPositionKp, -0.1)
    print('\nDone.')
