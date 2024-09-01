## Function that returns all the intervals given the Data (containing time, LP and PD signals), a time start (t_cut) from which the first neuron that spikes
## is first_neuron ('LP' or 'PD') and given minimum heigths and prominence for the spikes, as well as the maximum time distance between two spikes that
## belong to the same burst (dist_LP, dist_PD)

def Get_Intervals_Offline(Data, t_cut, first_neuron, prominence_LP_1, prominence_LP_2, prominence_PD_1, prominence_PD_2, height_LP, height_PD, dist_LP, dist_PD):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from slice_when import slice_when

    data = pd.read_csv(Data, delimiter='\t', header=None, decimal=',').dropna(axis=1)

    ## Time
    t = data.take([0], axis=1)
    t = t.to_numpy().reshape(np.shape(t.to_numpy())[0])
    t = t - t[0] * np.ones(np.shape(t))
    
    ## LP signal
    V_LP = data.take([1], axis=1)
    V_LP = V_LP.to_numpy().reshape(np.shape(V_LP.to_numpy())[0])
    
    ## PD signal
    V_PD = data.take([2], axis=1)
    V_PD = V_PD.to_numpy().reshape(np.shape(V_PD.to_numpy())[0])
    
    x_cut = np.argmin(np.abs(t - t_cut*np.ones(np.size(t))))

    ## Cut the arrays so both neurons start on a no spiking phase and PD (or LP, as requested by input) spikes first
    t = t[x_cut:]
    V_LP = V_LP[x_cut:]
    V_PD = V_PD[x_cut:]

    ## Using find_peaks with minimum height and prominence for PD
    PD_peaks = find_peaks(V_PD, height=height_PD, prominence=(prominence_PD_1,prominence_PD_2))[0].tolist()

    ## Using find_peaks with minimum height and prominence for LP
    LP_peaks = find_peaks(V_LP, height=height_LP, prominence=(prominence_LP_1,prominence_LP_2))[0].tolist()

    ## We use a function to group nearest elements from a list given a maximum distance in a cluster
    dist_PD = dist_PD / t[1]
    dist_LP = dist_LP / t[1]
    slices_LP = list(slice_when(lambda x,y: y - x > dist_LP, LP_peaks))
    slices_PD = list(slice_when(lambda x,y: y - x > dist_PD, PD_peaks))
    
    ## Define intervals
    period_LP, period_PD = [], []
    ibi_LP, ibi_PD = [], []
    burst_LP, burst_PD = [], []
    LP_PD_delay, PD_LP_delay = [], []
    LP_PD_interval, PD_LP_interval = [], []

    ## In some adquisitions, there could be a discoordination where LP spikes but does not induce spiking on PD (or viceversa)
    ## Thus, we are going to delete those alterations since they are not use for us
    ## Then, we calculate the intervals
    if first_neuron=='PD':
        indexes_to_cut = []
        i = 0
        while i < len(slices_PD)-2:
            if slices_PD[i+1][0]<slices_LP[i][0]:
                indexes_to_cut = [*indexes_to_cut, i]
                slices_PD.pop(i)
            elif slices_LP[i][0]<slices_PD[i][0]:
                indexes_to_cut = [*indexes_to_cut, i]
                slices_LP.pop(i)
            i+=1

        for i in range(len(slices_PD)-3):
            if i+1 not in indexes_to_cut:
                period_LP = [*period_LP, t[slices_LP[i+1][0]]-t[slices_LP[i][0]]]
                period_PD = [*period_PD, t[slices_PD[i+1][0]]-t[slices_PD[i][0]]]
                ibi_LP = [*ibi_LP, t[slices_LP[i+1][0]]-t[slices_LP[i][-1]]]
                ibi_PD = [*ibi_PD, t[slices_PD[i+1][0]]-t[slices_PD[i][-1]]]
                burst_LP = [*burst_LP, t[slices_LP[i][-1]]-t[slices_LP[i][0]]]
                burst_PD = [*burst_PD, t[slices_PD[i][-1]]-t[slices_PD[i][0]]]
                LP_PD_delay = [*LP_PD_delay, t[slices_PD[i+1][0]]-t[slices_LP[i][-1]]]
                PD_LP_delay = [*PD_LP_delay, t[slices_LP[i][0]]-t[slices_PD[i][-1]]]
                LP_PD_interval = [*LP_PD_interval, t[slices_PD[i+1][0]]-t[slices_LP[i][0]]]
                PD_LP_interval = [*PD_LP_interval, t[slices_LP[i][0]]-t[slices_PD[i][0]]]

    elif first_neuron=='LP':
        indexes_to_cut = []
        i = 0
        while i < len(slices_LP)-2:
            if slices_LP[i+1][0]<slices_PD[i][0]:
                indexes_to_cut = [*indexes_to_cut, i]
                slices_LP.pop(i)
            elif slices_PD[i][0]<slices_LP[i][0]:
                indexes_to_cut = [*indexes_to_cut, i]
                slices_PD.pop(i)
            i+=1
            
        for i in range(len(slices_LP)-3):
            if i+1 not in indexes_to_cut:
                period_LP = [*period_LP, t[slices_LP[i+1][0]]-t[slices_LP[i][0]]]
                period_PD = [*period_PD, t[slices_PD[i+1][0]]-t[slices_PD[i][0]]]
                ibi_LP = [*ibi_LP, t[slices_LP[i+1][0]]-t[slices_LP[i][-1]]]
                ibi_PD = [*ibi_PD, t[slices_PD[i+1][0]]-t[slices_PD[i][-1]]]
                burst_LP = [*burst_LP, t[slices_LP[i][-1]]-t[slices_LP[i][0]]]
                burst_PD = [*burst_PD, t[slices_PD[i][-1]]-t[slices_PD[i][0]]]
                LP_PD_delay = [*LP_PD_delay, t[slices_PD[i][0]]-t[slices_LP[i][-1]]]
                PD_LP_delay = [*PD_LP_delay, t[slices_LP[i+1][0]]-t[slices_PD[i][-1]]]
                LP_PD_interval = [*LP_PD_interval, t[slices_PD[i][0]]-t[slices_LP[i][0]]]
                PD_LP_interval = [*PD_LP_interval, t[slices_LP[i+1][0]]-t[slices_PD[i][0]]]


    Intervals = np.concatenate((np.array(period_LP).reshape(-1,1), np.array(period_PD).reshape(-1,1), np.array(ibi_LP).reshape(-1,1), 
                               np.array(ibi_PD).reshape(-1,1), np.array(burst_LP).reshape(-1,1), np.array(burst_PD).reshape(-1,1), 
                               np.array(LP_PD_delay).reshape(-1,1), np.array(PD_LP_delay).reshape(-1,1), np.array(LP_PD_interval).reshape(-1,1),
                               np.array(PD_LP_interval).reshape(-1,1)), axis=1)

    return t, V_PD, V_LP, slices_PD, slices_LP, Intervals

