def CompareSlopes(parameters, first_neuron, threshold=0.8):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from GetIntervalsOffline import Get_Intervals_Offline
    from Get_Slopes import Get_Slopes

    dict = {}
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

        ## Get Intervals and calculate slopes for dynamic invariants
        t, V_PD, V_LP, slices_PD, slices_LP, Intervals = Get_Intervals_Offline(Data, t_cut, first_neuron,prominence_LP_1, prominence_LP_2, prominence_PD_1, prominence_PD_2,height_LP, height_PD, dist_LP, dist_PD)
        results = Get_Slopes(Intervals, threshold)
        
        ## Save slope for each dynamic invariant
        for i,j in zip(list(results.Dataset),range(len(list(results.Dataset)))):
            if dict.get(i)==None:
                dict[i] = [[round(results.Slope[j], 3)], [round(results.R2[j], 3)], 1,[campaign[3:5]], [round(results.R2[j], 3)]]
            else:
                dict[i] = [dict.get(i)[0]+[round(results.Slope[j], 3)], dict.get(i)[1]+[round(results.R2[j], 3)], dict.get(i)[2]+1, dict.get(i)[3]+[campaign[3:5]], dict.get(i)[4]+[round(results.R2[j], 3)]]

    for i,j in zip(list(results.Dataset),range(len(list(results.Dataset)))):    
        dict[i] = [dict.get(i)[0], np.mean(np.array(dict.get(i)[1])), np.std(np.array(dict.get(i)[1])), dict.get(i)[2], dict.get(i)[3], dict.get(i)[4]]       
    ## Plot slope for each invariant and for each campaign
    fig, ax = plt.subplots()
    colors = sns.color_palette("blend:salmon,saddlebrown,lightseagreen", len(list(dict)))
    ax.set_prop_cycle('color', colors)
    for i in dict:
        ax.plot(dict.get(i)[0],'o')
    ax.legend(list(dict),bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim([0.8, 1.2])
    ax.set_ylabel('Invariants Slope')
    ax.set_xticks(range(len(dict.get(i)[4])), range(1,len(dict.get(i)[4])+1))
    ax.set_xlabel('Campaign')
    for i in dict:
        ax.plot(dict.get(i)[0],alpha=0.5)
    plt.show()

    return dict