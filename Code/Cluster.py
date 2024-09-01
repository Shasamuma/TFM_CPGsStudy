def Clustering(Data, t_cut, x_min=0, x_max=35000):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import find_peaks
    from sklearn.preprocessing import StandardScaler
    from sklearn import cluster
    from sklearn.neighbors import kneighbors_graph
    from itertools import cycle, islice
    
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
    
    # Cut the arrays so both neurons start on a no spiking phase and PD spikes first
    x_cut = np.argmin(np.abs(t - t_cut*np.ones(np.size(t))))

    t = t[x_cut:]
    V_LP = V_LP[x_cut:]
    V_PD = V_PD[x_cut:]
    
    # Plot data
    # x_min = x_cut
    # x_max = 70000
    fig, ax = plt.subplots()
    ax.plot(t[x_min:x_max], V_LP[x_min:x_max])
    ax.plot(t[x_min:x_max], V_PD[x_min:x_max])
    ax.legend(('LP data','PD data'))
    
    ######## Find peaks and classificate given the Voltage and Voltage prominence for PD neuron ########################################################
    PD_peaks_info = find_peaks(V_PD,prominence=0.1,height=-6)
    PD_peaks = PD_peaks_info[0]
    
    X_PD = np.concatenate((PD_peaks_info[1]['prominences'].reshape(-1,1), PD_peaks_info[1]['peak_heights'].reshape(-1,1)), axis=1)
    XS = StandardScaler().fit_transform(X_PD)
    
    # Define connectivity for Aglomerative Clustering
    connectivity = kneighbors_graph(XS, n_neighbors=5, include_self=False)
    # Make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    # Type classification (2 types), eg spike peak or not
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward", connectivity=connectivity)
    ward.fit(XS)
    type_PD = ward.labels_.astype(int)
    
    # Subtype: could we differenciate different types of peaks?
    ward = cluster.AgglomerativeClustering(n_clusters=4, linkage="ward", connectivity=connectivity)
    ward.fit(XS)
    subtype_PD = ward.labels_.astype(int)
    
    # Plot classification
    colors = np.array(list(islice(cycle(["#984ea3", "#e41a1c"]),int(max(type_PD) + 1),)))
    subcolors = np.array(list(islice(cycle(["#ff7f00","#4daf4a","#f781bf","#edc945"]),int(max(subtype_PD) + 1),)))
    # Add black color for outliers (if any)
    subcolors = np.append(subcolors, ["#000000"])
    colors = np.append(colors, ["#000000"])
    
    fig, ax = plt.subplots(1,2, sharey=True)
    ax[0].scatter(X_PD[:, 0], X_PD[:, 1], s=10, color=colors[type_PD])
    ax[0].set_title('Type')
    ax[1].scatter(X_PD[:, 0], X_PD[:, 1], s=10, color=subcolors[subtype_PD])
    ax[1].set_title('Subtype')
    ax[0].set_ylabel('Voltaje (mV)')
    ax[0].set_xlabel('Prominencia (mV)')
    ax[1].set_xlabel('Prominencia (mV)')
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    
    # Plot in PD signal
    subidx_PD_0 = np.sort(np.where(subtype_PD==0)[0])
    subidx_PD_1 = np.sort(np.where(subtype_PD==1)[0])
    subidx_PD_2 = np.sort(np.where(subtype_PD==2)[0])
    subidx_PD_3 = np.sort(np.where(subtype_PD==3)[0])

    subidx_PD = [subidx_PD_0, subidx_PD_1, subidx_PD_2, subidx_PD_3]

    idx_PD_0 = np.sort(np.where(type_PD==0)[0])
    idx_PD_1 = np.sort(np.where(type_PD==1)[0])

    idx_PD = [idx_PD_0, idx_PD_1]

    ## Time range for visualization
    # x_min = 0
    # x_max = 35000

    subidx_cut_0 = [i for i in np.where(PD_peaks[subidx_PD_0]>x_min)[0] if i in np.where(PD_peaks[subidx_PD_0]<x_max)[0]]
    subidx_cut_1 = [i for i in np.where(PD_peaks[subidx_PD_1]>x_min)[0] if i in np.where(PD_peaks[subidx_PD_1]<x_max)[0]]
    subidx_cut_2 = [i for i in np.where(PD_peaks[subidx_PD_2]>x_min)[0] if i in np.where(PD_peaks[subidx_PD_2]<x_max)[0]]
    subidx_cut_3 = [i for i in np.where(PD_peaks[subidx_PD_3]>x_min)[0] if i in np.where(PD_peaks[subidx_PD_3]<x_max)[0]]

    idx_cut_0 = [i for i in np.where(PD_peaks[idx_PD_0]>x_min)[0] if i in np.where(PD_peaks[idx_PD_0]<x_max)[0]]
    idx_cut_1 = [i for i in np.where(PD_peaks[idx_PD_1]>x_min)[0] if i in np.where(PD_peaks[idx_PD_1]<x_max)[0]]
    
    fig, ax = plt.subplots()
    ax.plot(t[x_min:x_max], V_PD[x_min:x_max],linewidth=1.2)
    ax.plot(t[PD_peaks[subidx_PD_0][subidx_cut_0]], V_PD[PD_peaks[subidx_PD_0][subidx_cut_0]], color=subcolors[0],marker='o',linestyle='',markeredgewidth=2)
    ax.plot(t[PD_peaks[subidx_PD_1][subidx_cut_1]], V_PD[PD_peaks[subidx_PD_1][subidx_cut_1]], color=subcolors[1],marker='o',linestyle='',markeredgewidth=2)
    ax.plot(t[PD_peaks[subidx_PD_2][subidx_cut_2]], V_PD[PD_peaks[subidx_PD_2][subidx_cut_2]], color=subcolors[2],marker='o',linestyle='',markeredgewidth=2)
    ax.plot(t[PD_peaks[subidx_PD_3][subidx_cut_3]], V_PD[PD_peaks[subidx_PD_3][subidx_cut_3]], color=subcolors[3],marker='o',linestyle='',markeredgewidth=2)
    ax.plot(t[PD_peaks[idx_PD_0][idx_cut_0]], V_PD[PD_peaks[idx_PD_0][idx_cut_0]], color=colors[0],marker='1',linestyle='',markeredgewidth=1.3)
    ax.plot(t[PD_peaks[idx_PD_1][idx_cut_1]], V_PD[PD_peaks[idx_PD_1][idx_cut_1]], color=colors[1],marker='1',linestyle='',markeredgewidth=1.3)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('Voltaje (mV)')
    ax.legend(('pd data','subgroup 1','subgroup 2','subgroup 3','subgroup 4','group 1','group 2'),bbox_to_anchor=(1.05, 1), loc='upper left')
######## Find peaks and classificate given the Voltage and Voltage prominence for LP neuron ########################################################
    LP_peaks_info = find_peaks(V_LP,prominence=0.1,height=-6)
    LP_peaks = LP_peaks_info[0]
    
    X_LP = np.concatenate((LP_peaks_info[1]['prominences'].reshape(-1,1), LP_peaks_info[1]['peak_heights'].reshape(-1,1)), axis=1)
    XS = StandardScaler().fit_transform(X_LP)
    
    # Define connectivity for Aglomerative Clustering
    connectivity = kneighbors_graph(XS, n_neighbors=5, include_self=False)
    # Make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    # Type classification (2 types), eg spike peak or not
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward", connectivity=connectivity)
    ward.fit(XS)
    type_LP = ward.labels_.astype(int)
    
    # Subtype: could we differenciate different types of peaks?
    ward = cluster.AgglomerativeClustering(n_clusters=4, linkage="ward", connectivity=connectivity)
    ward.fit(XS)
    subtype_LP = ward.labels_.astype(int)
    
    # Plot classification
    colors = np.array(list(islice(cycle(["#984ea3", "#e41a1c"]),int(max(type_LP) + 1),)))
    subcolors = np.array(list(islice(cycle(["#ff7f00","#4daf4a","#f781bf","#edc945"]),int(max(subtype_LP) + 1),)))
    # Add black color for outliers (if any)
    subcolors = np.append(subcolors, ["#000000"])
    colors = np.append(colors, ["#000000"])
    
    fig, ax = plt.subplots(1,2, sharey=True)
    ax[0].scatter(X_LP[:, 0], X_LP[:, 1], s=10, color=colors[type_LP])
    ax[0].set_title('Type')
    ax[1].scatter(X_LP[:, 0], X_LP[:, 1], s=10, color=subcolors[subtype_LP])
    ax[1].set_title('Subtype')
    ax[0].set_ylabel('Voltaje (mV)')
    ax[0].set_xlabel('Prominencia (mV)')
    ax[1].set_xlabel('Prominencia (mV)')
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    
    # Plot in LP signal
    subidx_LP_0 = np.sort(np.where(subtype_LP==0)[0])
    subidx_LP_1 = np.sort(np.where(subtype_LP==1)[0])
    subidx_LP_2 = np.sort(np.where(subtype_LP==2)[0])
    subidx_LP_3 = np.sort(np.where(subtype_LP==3)[0])

    subidx_LP = [subidx_LP_0, subidx_LP_1, subidx_LP_2, subidx_LP_3]

    idx_LP_0 = np.sort(np.where(type_LP==0)[0])
    idx_LP_1 = np.sort(np.where(type_LP==1)[0])

    idx_LP = [idx_LP_0, idx_LP_1]

    ## Time range for visualization
    # x_min = 93855
    # x_max = 93855 + 60000

    subidx_cut_0 = [i for i in np.where(LP_peaks[subidx_LP_0]>x_min)[0] if i in np.where(LP_peaks[subidx_LP_0]<x_max)[0]]
    subidx_cut_1 = [i for i in np.where(LP_peaks[subidx_LP_1]>x_min)[0] if i in np.where(LP_peaks[subidx_LP_1]<x_max)[0]]
    subidx_cut_2 = [i for i in np.where(LP_peaks[subidx_LP_2]>x_min)[0] if i in np.where(LP_peaks[subidx_LP_2]<x_max)[0]]
    subidx_cut_3 = [i for i in np.where(LP_peaks[subidx_LP_3]>x_min)[0] if i in np.where(LP_peaks[subidx_LP_3]<x_max)[0]]

    idx_cut_0 = [i for i in np.where(LP_peaks[idx_LP_0]>x_min)[0] if i in np.where(LP_peaks[idx_LP_0]<x_max)[0]]
    idx_cut_1 = [i for i in np.where(LP_peaks[idx_LP_1]>x_min)[0] if i in np.where(LP_peaks[idx_LP_1]<x_max)[0]]
    
    fig, ax = plt.subplots()
    ax.plot(t[x_min:x_max], V_LP[x_min:x_max],linewidth=1.2)
    #ax.plot(t[LP_peaks[subidx_LP_0][subidx_cut_0]], V_LP[LP_peaks[subidx_LP_0][subidx_cut_0]], color=subcolors[0],marker='o',linestyle='',markeredgewidth=2)
    #ax.plot(t[LP_peaks[subidx_LP_1][subidx_cut_1]], V_LP[LP_peaks[subidx_LP_1][subidx_cut_1]], color=subcolors[1],marker='o',linestyle='',markeredgewidth=2)
    #ax.plot(t[LP_peaks[subidx_LP_2][subidx_cut_2]], V_LP[LP_peaks[subidx_LP_2][subidx_cut_2]], color=subcolors[2],marker='o',linestyle='',markeredgewidth=2)
    #ax.plot(t[LP_peaks[subidx_LP_3][subidx_cut_3]], V_LP[LP_peaks[subidx_LP_3][subidx_cut_3]], color=subcolors[3],marker='o',linestyle='',markeredgewidth=2)
    ax.plot(t[LP_peaks[idx_LP_0][idx_cut_0]], V_LP[LP_peaks[idx_LP_0][idx_cut_0]], color=colors[0],marker='o',linestyle='',markeredgewidth=1)
    ax.plot(t[LP_peaks[idx_LP_1][idx_cut_1]], V_LP[LP_peaks[idx_LP_1][idx_cut_1]], color=colors[1],marker='o',linestyle='',markeredgewidth=1)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('Voltaje (mV)')
    ax.legend(('LP data','group 1','group 2'),bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Confirm that each subgroup is contained entirely within a group
    print('PD subgroup 1 is contained by PD group ', set(type_PD[subidx_PD_0]))
    print('PD subgroup 2 is contained by PD group ', set(type_PD[subidx_PD_1]))
    print('PD subgroup 3 is contained by PD group ', set(type_PD[subidx_PD_2]))
    print('PD subgroup 4 is contained by PD group ', set(type_PD[subidx_PD_3]))
    print('LP subgroup 1 is contained by LP group ', set(type_LP[subidx_LP_0]))
    print('LP subgroup 2 is contained by LP group ', set(type_LP[subidx_LP_1]))
    print('LP subgroup 3 is contained by LP group ', set(type_LP[subidx_LP_2]))
    print('LP subgroup 4 is contained by LP group ', set(type_LP[subidx_LP_3]))
    
    # Select the group which selects the spikes
    mn = [np.mean(X_PD[idx_PD_0,1]), np.mean(X_PD[idx_PD_1,1])]
    spike_group_PD = np.where(np.max(mn)==mn)[0][0]
    spike_idx_PD = PD_peaks[np.where(type_PD==spike_group_PD)[0]]
    spike_subgroups_PD = list(set(subtype_PD[idx_PD[spike_group_PD]]))
    
    mn = [np.mean(X_LP[idx_LP_0,1]), np.mean(X_LP[idx_LP_1,1])]
    spike_group_LP = np.where(np.max(mn)==mn)[0][0]
    spike_idx_LP = LP_peaks[np.where(type_LP==spike_group_LP)[0]]
    spike_subgroups_LP = list(set(subtype_LP[idx_LP[spike_group_LP]]))


    spike_variables_LP = np.concatenate((np.delete(X_LP[np.where(type_LP==spike_group_LP)[0],:],[0],0), (np.array(spike_idx_LP[1:]-spike_idx_LP[:-1])).reshape(-1,1)),axis=1)
  
    spike_variables_PD = np.concatenate((np.delete(X_PD[np.where(type_PD==spike_group_PD)[0],:],[0],0), (np.array(spike_idx_PD[1:]-spike_idx_PD[:-1])).reshape(-1,1)),axis=1)


    return spike_idx_LP, spike_variables_LP, spike_idx_PD, spike_variables_PD
    
   
    







