def Get_Intervals_Online(method, Data, t_cut, spiking_height_LP, spiking_height_PD_1, spiking_height_PD_2, min_period_LP, min_period_PD, max_burst_LP, max_burst_PD):
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks
    from time import time

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
    
    ## Redefine in index
    min_period_LP = min_period_LP / t[1]
    min_period_PD = min_period_PD / t[1]
    max_burst_LP = max_burst_LP / t[1]
    max_burst_PD = max_burst_PD / t[1]

    if method=='Instant':
        t_detect_LP, t_detect_PD = None, None

        LP_state = 'ibi'
        LP_ini = []
        LP_fini = []

        PD_state = 'ibi'
        PD_ini = []
        PD_fini = []

        index_max_of_last_LP = - min_period_LP
        index_max_of_last_PD = - min_period_PD
        
        t0 = time()
        for i in range(len(V_LP)):
            ## LP detection
            ## To detect the spiking start we use a voltage threshold spiking_height_LP and an minimum distance between bursts min_period_LP
            if LP_state=='ibi' and V_LP[i]>spiking_height_LP and i-index_max_of_last_LP>min_period_LP  and PD_state=='ibi':
                idx_ini_LP = i
                LP_ini.append(i)
                LP_state = 'spiking'
                index_max_of_last_LP = i

            ## To detect the final spike we use a moving cell with width max_burst_LP that saves the maximum voltage; if that maximum voltage is lower than 
            ## no_spiking_height_LP, we state the the neuron is not longer spiking
            elif LP_state=='spiking':
                if V_LP[i]>=spiking_height_LP:
                    index_max_of_last_LP = i
                else:
                    if i - index_max_of_last_LP >= max_burst_LP and i - idx_ini_LP > 2*max_burst_LP:
                        LP_fini.append(i)
                        LP_state = 'ibi'

            ## PD detection
            ## To detect the spiking start we use a voltage threshold spiking_height_PD and an minimum distance between bursts min_period_PD
            if PD_state=='ibi' and V_PD[i]>spiking_height_PD_1 and i-index_max_of_last_PD>min_period_PD:
                idx_ini_PD = i
                PD_ini.append(i)
                PD_state = 'spiking'
                index_max_of_last_PD = i

            ## To detect the final spike we use a moving cell with width max_burst_PD that saves the a voltage (in this case, it saves either current 
            ## point if the point is higher than spiking_height_PD or the maximum); if that maximum voltage is lower than no_spiking_height_LP, we 
            ## state the the neuron is not longer spiking
            elif PD_state=='spiking':
                if V_PD[i]>=spiking_height_PD_2:
                    index_max_of_last_PD = i
                elif V_PD[i]<spiking_height_PD_1:
                    if i - index_max_of_last_PD >= max_burst_PD  and i - idx_ini_PD > 2*max_burst_PD:
                        PD_fini.append(i)
                        PD_state = 'ibi'
                    
        processing_time = [time() - t0, (time() - t0)/len(V_LP)]

            
    elif method=='Delayed':
        LP_state = 'ibi'
        LP_ini = []
        LP_fini = []

        PD_state = 'ibi'
        PD_ini = []
        PD_fini = []

        t_detect_PD = []
        t_detect_LP = []

        index_max_of_last_LP = - min_period_LP
        index_max_of_last_PD = - min_period_PD
        t0 = time()
        for i in range(len(V_LP)):
            
            ## LP detection
            if LP_state=='ibi' and V_LP[i]>spiking_height_LP and i-index_max_of_last_LP>min_period_LP and PD_state=='ibi':
                idx_ini_LP = i
                LP_ini.append(i)
                LP_state = 'spiking'
                index_max_of_last_LP = i

            elif LP_state=='spiking':
                if V_LP[i]>=spiking_height_LP:
                    index_max_of_last_LP = i
                else:
                    if i - index_max_of_last_LP >= max_burst_LP and i - idx_ini_LP > 2*max_burst_LP:
                        LP_fini.append(index_max_of_last_LP)
                        LP_state = 'ibi'
                        t_detect_LP.append(i)


            # PD detection
            if PD_state=='ibi' and V_PD[i]>spiking_height_PD_1 and i-index_max_of_last_PD>min_period_PD:
                idx_ini_PD = i
                PD_ini = [*PD_ini, i]
                PD_state = 'spiking'
                index_max_of_last_PD = i

            elif PD_state=='spiking':
                if V_PD[i]>=spiking_height_PD_2:
                    index_max_of_last_PD = i
                elif V_PD[i]<spiking_height_PD_1:
                    if i - index_max_of_last_PD >= max_burst_PD  and i - idx_ini_PD > 2*max_burst_PD:
                        peak_PD = find_peaks(V_PD[int(i-1000):i], height=spiking_height_PD_2, prominence=0.2)
                        if len(peak_PD[0])>0:
                            PD_fini.append(peak_PD[0][-1]+int(i-1000))
                        else: 
                            PD_fini.append(index_max_of_last_PD)
                        PD_state = 'ibi'
                        t_detect_PD.append(i)

                    
        processing_time = [time() - t0, (time() - t0)/len(V_LP)]
        
    ## Considering no discoordination between neurons and that we start our cycle with PD spiking:

    print('PD ini', len(PD_ini))
    print('PD fini', len(PD_fini))
    print('LP ini', len(LP_ini))
    print('LP fini', len(LP_fini))
    indexes_to_cut = []
    i = 0
    while i < len(PD_fini)-2:
        if PD_ini[i+1]<LP_ini[i]:
            indexes_to_cut = [*indexes_to_cut, i]
            PD_ini.pop(i)
            PD_fini.pop(i)
        elif LP_fini[i]<PD_ini[i]:
            indexes_to_cut = [*indexes_to_cut, i]
            LP_ini.pop(i)
            LP_fini.pop(i)
        i+=1
    
    t_inis_LP = t[LP_ini]
    t_finis_LP = t[LP_fini]
    t_inis_PD = t[PD_ini]
    t_finis_PD = t[PD_fini]
    t_detect_LP = t[t_detect_LP]
    t_detect_PD = t[t_detect_PD]
    
    ## Define intervals
    period_LP, period_PD = [], []
    ibi_LP, ibi_PD = [], []
    burst_LP, burst_PD = [], []
    LP_PD_delay, PD_LP_delay = [], []
    LP_PD_interval, PD_LP_interval = [], []
    
    for i in range(min(len(t_finis_LP),len(t_finis_PD)) - 1):
        if i+1 not in indexes_to_cut:
            period_LP = [*period_LP, t_inis_LP[i+1] - t_inis_LP[i]]
            period_PD = [*period_PD, t_inis_PD[i+1] - t_inis_PD[i]]
            ibi_LP = [*ibi_LP, t_inis_LP[i+1] - t_finis_LP[i]]
            ibi_PD = [*ibi_PD, t_inis_PD[i+1] - t_finis_PD[i]]
            burst_LP = [*burst_LP, t_finis_LP[i] - t_inis_LP[i]]
            burst_PD = [*burst_PD, t_finis_PD[i] - t_inis_PD[i]]
            LP_PD_delay = [*LP_PD_delay, t_inis_PD[i+1] - t_finis_LP[i]]
            PD_LP_delay = [*PD_LP_delay, t_inis_LP[i] - t_finis_PD[i]]
            LP_PD_interval = [*LP_PD_interval, t_inis_PD[i+1] - t_inis_LP[i]]
            PD_LP_interval = [*PD_LP_interval, t_inis_LP[i] - t_inis_PD[i]]
        
        
    Intervals = np.concatenate((np.array(period_LP).reshape(-1,1), np.array(period_PD).reshape(-1,1), np.array(ibi_LP).reshape(-1,1), 
                               np.array(ibi_PD).reshape(-1,1), np.array(burst_LP).reshape(-1,1), np.array(burst_PD).reshape(-1,1), 
                               np.array(LP_PD_delay).reshape(-1,1), np.array(PD_LP_delay).reshape(-1,1), np.array(LP_PD_interval).reshape(-1,1),
                               np.array(PD_LP_interval).reshape(-1,1)), axis=1)

   
    t_LP = np.concatenate((np.array(t_inis_LP[:min(len(t_finis_LP),len(t_finis_PD))]).reshape(-1,1), np.array(t_finis_LP[:min(len(t_finis_LP),len(t_finis_PD))]).reshape(-1,1)), axis=1)
    t_PD = np.concatenate((np.array(t_inis_PD[:min(len(t_finis_LP),len(t_finis_PD))]).reshape(-1,1), np.array(t_finis_PD[:min(len(t_finis_LP),len(t_finis_PD))]).reshape(-1,1)), axis=1)

    return t, V_PD, V_LP, t_PD, t_LP, processing_time, Intervals, t_detect_LP, t_detect_PD