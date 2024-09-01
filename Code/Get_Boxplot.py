def Get_Boxplot(parameters, first_neuron):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from GetIntervalsOffline import Get_Intervals_Offline
    from Get_Slopes import Get_Slopes

    cont = 0
    for campaign in parameters:
        ## Define the parameters for each campaign
        Data = '../robot/txt/'+campaign
        t_cut = parameters[campaign][-1]
        print(campaign)
        prominence_LP_1 = parameters[campaign][0]
        prominence_LP_2 = parameters[campaign][1]
        height_LP = parameters[campaign][2]
        prominence_PD_1 = parameters[campaign][3]
        prominence_PD_2 = parameters[campaign][4]
        height_PD = parameters[campaign][5]
        dist_LP = parameters[campaign][6]*t_cut
        dist_PD = parameters[campaign][7]*t_cut

        ## Get Intervals and save all of them
        t, V_PD, V_LP, slices_PD, slices_LP, Intervals = Get_Intervals_Offline(Data, t_cut, first_neuron,prominence_LP_1, prominence_LP_2, prominence_PD_1, prominence_PD_2,height_LP, height_PD, dist_LP, dist_PD)
        if cont==0:
            all_Intervals = Intervals
            cont+=1
        else:
            all_Intervals = np.concatenate((all_Intervals, Intervals),0)

   
    ## Plot boxplot
    Labels = ["Period LP","Period PD","IBI LP","IBI PD","Burst LP","Burst PD","Interval LPPD","Interval PDLP","Delay LPPD","Delay PDLP"]
    Data = pd.DataFrame(data=all_Intervals, columns=Labels) 
    sns.set(style="ticks", rc={"figure.figsize": (15, 7), "figure.facecolor": "white", "axes.facecolor": "white"})  
    b = sns.stripplot(data = Data, color = "salmon", linewidth = 0.5, alpha = 0.2) 
    b = sns.boxplot(data = Data, width = 0.4, color = "lightseagreen", linewidth = 2, showfliers = False)  
    b.set_ylabel("Duration (ms)", fontsize = 10)
    sns.despine(offset = 5, trim = True)
    b.get_figure()





