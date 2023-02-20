import numpy as np
from scipy.signal import detrend
import re
def ReadData(path):
    try:
    #with open("felttest2/Record_2022-09-28_14-48-46/Record_2022-09-28_14-48-46.bin", "rb") as f:
    #with open("Record_2022-09-28_15-13-15.bin", "rb") as f:
        with open(path, "rb") as f:

            data = f.read()
            data= data[5:-5]
            stri = "DONE"
            done = False
            data_arr = []
            data_bytes = data
            while(not done):
                try:
                    ls = re.search(stri.encode(), data_bytes).end()
                    data_arr.append(data_bytes[:ls])
                    data_bytes = data_bytes[ls+4:]
                    print(len(data_bytes))
                    if(len(data_bytes) == 0):
                        done = True
                except:
                    print("Error")
                    done = True
                
            return data_arr
    
        
        
    except IOError as e:
        print(e)

def Decode(data):
    RADC = []
    for i,frame in enumerate(data):
            length = int.from_bytes(
                    frame[4:8], byteorder="little", signed=False)
            
            data_RADC = frame[8:8+length]
            data_RADC = np.frombuffer(data_RADC,dtype=np.uint16)
            data_RADC = data_RADC.reshape(3,256,512)
            data_RADC_I_raw = data_RADC[:,:,::2]
            data_RADC_Q_raw = data_RADC[:,:,1::2]
            data_RADC_I = detrend(data_RADC_I_raw, axis=2)
            data_RADC_Q = detrend(data_RADC_Q_raw, axis=2)
            
            data_RADC_I_mean = data_RADC_I
            data_RADC_Q_mean = data_RADC_Q
            data_IF = data_RADC_I_mean[:,:,:] + 1j*data_RADC_Q_mean[:,:,:]
            RADC.append(data_IF)

    return RADC

def SaveData(data, path):
    np.save(path, data)

def LoadData(path):
    return np.load(path)


def GetAndSaveData(path,savePath):
    data = ReadData(path)
    RADC = Decode(data)
    SaveData(RADC, savePath)

    return RADC