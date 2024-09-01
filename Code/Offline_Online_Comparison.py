def Offline_Online_Comparison(parameters_online, parameters_offline, first_neuron):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from GetIntervalsOffline import Get_Intervals_Offline
    from Get_Intervals_Online import Get_Intervals_Online


    dict = {}

    fig, ax = plt.subplots(5,2,sharex=True, figsize=(10, 20))
    fig2, ax2 = plt.subplots(10,10, figsize=(30,30))
    
    ax = ax.flatten()
    cont = 0

    dif_inis_PD_all, dif_finis_PD_all, dif_inis_LP_all, dif_finis_LP_all, dif_inis_PD_max, dif_finis_PD_max, dif_inis_LP_max, dif_finis_LP_max, processing_time_all, mean_processing_time_all, t_detect_LP_all, t_detect_PD_all = [], [], [],[],[],[],[],[],[],[],[],[]
    for campaign in parameters_offline:
        ## Define the parameters for each campaign
        Data = '../robot/txt/'+campaign
        if first_neuron=='PD':
            t_cut = parameters_offline[campaign][-1]
        elif first_neuron=='LP':
            t_cut = parameters_offline[campaign][-2]
        print(campaign)
        prominence_LP_1 = parameters_offline[campaign][0]
        prominence_LP_2 = parameters_offline[campaign][1]
        height_LP = parameters_offline[campaign][2]
        prominence_PD_1 = parameters_offline[campaign][3]
        prominence_PD_2 = parameters_offline[campaign][4]
        height_PD = parameters_offline[campaign][5]
        dist_LP = parameters_offline[campaign][6]*t_cut
        dist_PD = parameters_offline[campaign][7]*t_cut

        ## Get Offline Analysis
        t, V_PD, V_LP, slices_PD, slices_LP, Intervals_off = Get_Intervals_Offline(Data, t_cut, first_neuron,prominence_LP_1, prominence_LP_2, prominence_PD_1, prominence_PD_2,height_LP, height_PD, dist_LP, dist_PD)

        spiking_height_LP = parameters_online[campaign][0]
        spiking_height_PD_1 = parameters_online[campaign][1]
        spiking_height_PD_2 = parameters_online[campaign][2]
        min_period_LP = parameters_online[campaign][3]*t_cut
        min_period_PD = parameters_online[campaign][4]*t_cut
        max_burst_LP = parameters_online[campaign][5]*t_cut
        max_burst_PD = parameters_online[campaign][6]*t_cut

        ## Get Online Analysis
        method = 'Delayed'
        t, V_PD, V_LP, t_PD, t_LP, processing_time, Intervals_on, t_detect_LP, t_detect_PD = Get_Intervals_Online(method, Data, t_cut, spiking_height_LP, spiking_height_PD_1, spiking_height_PD_2, min_period_LP, min_period_PD, max_burst_LP, max_burst_PD)

        ## Comparison
        t_PD_off_1 = [i[0] for i in slices_PD]
        t_PD_off_1 = t[t_PD_off_1]
        t_PD_off_2 = [i[-1] for i in slices_PD]
        t_PD_off_2 = t[t_PD_off_2]

        t_PD_off = np.concatenate((np.array(t_PD_off_1).reshape(-1,1), np.array(t_PD_off_2).reshape(-1,1)), axis=1)

        t_LP_off_1 = [i[0] for i in slices_LP]
        t_LP_off_1 = t[t_LP_off_1]
        t_LP_off_2 = [i[-1] for i in slices_LP]
        t_LP_off_2 = t[t_LP_off_2]

        t_LP_off = np.concatenate((np.array(t_LP_off_1).reshape(-1,1), np.array(t_LP_off_2).reshape(-1,1)), axis=1)

        ## Define absolute diferences between online and offline detection
        dif_inis_PD, dif_finis_PD = [], []
        if len(t_PD[:,0]) <= len(t_PD_off[:,0]):
            for i in range(len(t_PD[:,0])):
                idx = np.argmin(np.abs(t_PD[i,0] * np.ones(np.shape(t_PD_off[:,0])) - t_PD_off[:,0]))
                dif_inis_PD = [*dif_inis_PD, np.abs(t_PD[i,0] - t_PD_off[idx,0])]
                dif_finis_PD = [*dif_finis_PD, np.abs(t_PD[i,1] - t_PD_off[idx,1])]
        else:
            for i in range(len(t_PD_off[:,0])):
                idx = np.argmin(np.abs(t_PD_off[i,0] * np.ones(np.shape(t_PD[:,0])) - t_PD[:,0]))
                dif_inis_PD = [*dif_inis_PD, np.abs(t_PD_off[i,0] - t_PD[idx,0])]
                dif_finis_PD = [*dif_finis_PD, np.abs(t_PD_off[i,1] - t_PD[idx,1])]

        dif_inis_LP, dif_finis_LP = [], []
        if len(t_LP[:,0]) <= len(t_LP_off[:,0]):
            for i in range(len(t_LP[:,0])):
                idx = np.argmin(np.abs(t_LP[i,0] * np.ones(np.shape(t_LP_off[:,0])) - t_LP_off[:,0]))
                dif_inis_LP = [*dif_inis_LP, np.abs(t_LP[i,0] - t_LP_off[idx,0])]
                dif_finis_LP = [*dif_finis_LP, np.abs(t_LP[i,1] - t_LP_off[idx,1])]
        else:
            for i in range(len(t_LP_off[:,0])):
                idx = np.argmin(np.abs(t_LP_off[i,0] * np.ones(np.shape(t_LP[:,0])) - t_LP[:,0]))
                dif_inis_LP = [*dif_inis_LP, np.abs(t_LP_off[i,0] - t_LP[idx,0])]
                dif_finis_LP = [*dif_finis_LP, np.abs(t_LP_off[i,1] - t_LP[idx,1])]

        ## Calculate mean diferences
        dif_inis_PD_all = [*dif_inis_PD_all, np.sum(np.array(dif_inis_PD))/len(dif_inis_PD)]
        dif_finis_PD_all = [*dif_finis_PD_all, np.sum(np.array(dif_finis_PD))/len(dif_finis_PD)]
        dif_inis_LP_all = [*dif_inis_LP_all, np.sum(np.array(dif_inis_LP))/len(dif_inis_LP)]
        dif_finis_LP_all = [*dif_finis_LP_all, np.sum(np.array(dif_finis_LP))/len(dif_finis_LP)]

        dif_inis_PD_max = [*dif_inis_PD_max, np.max(np.array(dif_inis_PD))]
        dif_finis_PD_max = [*dif_finis_PD_max, np.max(np.array(dif_finis_PD))]
        dif_inis_LP_max = [*dif_inis_LP_max, np.max(np.array(dif_inis_LP))]
        dif_finis_LP_max = [*dif_finis_LP_max, np.max(np.array(dif_finis_LP))]

        mean_processing_time_all = [*mean_processing_time_all, processing_time[1]]
        processing_time_all = [*processing_time_all, processing_time[0]]

        dif_t_detect_LP, dif_t_detect_PD = [], []
        if len(t_detect_LP) <= len(t_LP[:,1]):
            for i in range(len(t_detect_LP)):
                idx = np.argmin(np.abs(t_detect_LP[i] * np.ones(np.shape(t_LP[:,1])) - t_LP[:,1]))
                dif_t_detect_LP = [*dif_t_detect_LP, np.abs(t_detect_LP[i] - t_LP[idx,1])]
        else:
            for i in range(len(t_LP[:,1])):
                idx = np.argmin(np.abs(t_LP[i, 1] * np.ones(np.shape(t_detect_LP)) - t_detect_LP))
                dif_t_detect_LP = [*dif_t_detect_LP, np.abs(t_detect_LP[idx] - t_LP[i,1])]

        if len(t_detect_PD) <= len(t_PD[:,1]):
            for i in range(len(t_detect_PD)):
                idx = np.argmin(np.abs(t_detect_PD[i] * np.ones(np.shape(t_PD[:,1])) - t_PD[:,1]))
                dif_t_detect_PD = [*dif_t_detect_PD, np.abs(t_detect_PD[i] - t_PD[idx,1])]
        else:
            for i in range(len(t_PD[:,1])):
                idx = np.argmin(np.abs(t_PD[i, 1] * np.ones(np.shape(t_detect_PD)) - t_detect_PD))
                dif_t_detect_PD = [*dif_t_detect_PD, np.abs(t_detect_PD[idx] - t_PD[i,1])]

        t_detect_LP_all =  [*t_detect_LP_all, np.sum(np.array(dif_t_detect_LP))/len(dif_t_detect_LP)]
        t_detect_PD_all =  [*t_detect_PD_all, np.sum(np.array(dif_t_detect_PD))/len(dif_t_detect_PD)]

        Labels = ["Start PD","End PD","Start LP","End LP"]
        Data = np.concatenate((np.array(dif_inis_PD).reshape(-1,1), np.array(dif_finis_PD).reshape(-1,1), np.array(dif_inis_LP).reshape(-1,1), np.array(dif_finis_LP).reshape(-1,1)),1)


        Data = pd.DataFrame(data=Data, columns=Labels)
        b = sns.stripplot(data = Data , color = "salmon", linewidth = 0.5, alpha = 0.5, ax=ax[cont]) 
        b = sns.boxplot(data = Data, width = 0.4, color = "lightseagreen", linewidth = 2, showfliers = False, ax=ax[cont])  
        ax[cont].title.set_text('Campaign '+str(cont+1))
        b.set_ylabel("Error (ms)", fontsize = 10)
        sns.despine(offset = 5, trim = False)
        



        data_dict = {"Periodo LP (ms)": Intervals_off[:,0],
        "Periodo PD (ms)": Intervals_off[:,1],
        "Hiperpolarización LP (ms)": Intervals_off[:,2],
        "Hiperpolarización PD (ms)": Intervals_off[:,3],
        "Burst LP (ms)": Intervals_off[:,4],
        "Burst PD (ms)": Intervals_off[:,5],
        "Interval LPPD (ms)": Intervals_off[:,6],
        "Interval PDLP (ms)": Intervals_off[:,7],
        "Delay LPPD (ms)": Intervals_off[:,8],
        "Delay PDLP (ms)": Intervals_off[:,9]}

        data_off = pd.DataFrame(data_dict)

        data_dict = {"Periodo LP (ms)": Intervals_on[:,0],
            "Periodo PD (ms)": Intervals_on[:,1],
            "Hiperpolarización LP (ms)": Intervals_on[:,2],
            "Hiperpolarización PD (ms)": Intervals_on[:,3],
            "Burst LP (ms)": Intervals_on[:,4],
            "Burst PD (ms)": Intervals_on[:,5],
            "Interval LPPD (ms)": Intervals_on[:,6],
            "Interval PDLP (ms)": Intervals_on[:,7],
            "Delay LPPD (ms)": Intervals_on[:,8],
            "Delay PDLP (ms)": Intervals_on[:,9]}

        data_on = pd.DataFrame(data_dict)
        for i in range(10):
            sns.kdeplot(data_off, x=data_off.columns[i], ax=ax2[cont, i], fill=True, legend=False, color='brown', alpha=0.5)
            sns.kdeplot(data_on, x=data_on.columns[i], ax=ax2[cont, i], fill=True, legend=False, color='steelblue', alpha=0.4)
            ax2[cont, i].yaxis.set_visible(False)

        cont = cont + 1
    
    dict['Campaign'] = ["sub1.txt", "sub4.txt", "sub6.txt", "sub8.txt", "sub10.txt", "sub12.txt", "sub13.txt", "sub14.txt", "sub15.txt", "sub16.txt"]
    dict['Mean error Start PD Burst'] = dif_inis_PD_all
    dict['Mean error End PD Burst'] = dif_finis_PD_all
    dict['Mean error Start LP Burst'] = dif_inis_LP_all
    dict['Mean error End LP Burst'] = dif_finis_LP_all
    dict['Max error Start PD Burst'] = dif_inis_PD_max
    dict['Max error End PD Burst'] = dif_finis_PD_max
    dict['Max error Start LP Burst'] = dif_inis_LP_max
    dict['Max error End LP Burst'] = dif_finis_LP_max
    dict['Time from peak to detection End Burst LP'] = t_detect_LP_all
    dict['Time from peak to detection End Burst PD'] = t_detect_PD_all
    dict['Total online processing time'] = processing_time_all
    dict['Per point online processing time'] = mean_processing_time_all



        
    fig2.tight_layout()
    plt.show()

    


    return dict

    