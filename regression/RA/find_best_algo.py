# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
directory_path = os.getcwd()
print("in directory: ",directory_path)
from matplotlib.pyplot import figure

data_type = ["Kin8","CCPP","Naval_Propulsion","California_Housing","WineWhite","Exponential_Function"]

"""
## Plotting routine ##
"""

# %%
def plot_errors(model_type,shift):
    data_ticks = ["K8","CCPP","NP","CH","WW","Sim"]
    data_type = ["Kin8","CCPP","Naval_Propulsion","California_Housing","WineWhite","Exponential_Function"]
    # file where the plot will be stored
    df = {} # dictionary to stores the results
    if shift:
        shift_name = "with"
    else:
        shift_name = "without"

    output_filename = "results/"+model_type + shift_name + ".pdf"
    # read the results
    for data in data_type:
        base_file_name = "/results/error_table_" + data + "_"
        df[data] = pd.read_csv(directory_path + base_file_name + model_type + '_' + shift_name +  ".csv",sep = ",")

    
    figure(figsize=(8, 6), dpi=80)
    markers = ["ko","bd","gs","rP","cv","m<","y>","bX"]
    if model_type == "NN" and (not shift):
      legends = ["CP-S","CP-CV","Res-Gauss","Emp"]
      names_model = legends[:]
      names_model[-1] = "emp coverage"
    if model_type == "NN" and shift:
      legends = ["CP-SW","CP-CVW","CP-S","Res-Gauss","Emp"]
      names_model = legends[:]
      names_model[-1] = "emp coverage"
    if model_type == "MVE_NN" and (not shift):
      legends = ["CP-S","CP-CV","MVE","Emp"]
      names_model = legends[:]
      names_model[-1] = "emp coverage"
    if model_type == "MVE_NN" and shift:
      legends = ["CP-SW","CP-CVW","CP-S","MVE","Emp"]
      names_model = legends[:]
      names_model[-1] = "emp coverage"


    leg_marker={k:v for k,v in zip(legends,markers)}
    size_marker = 10
    # locations on the x-axis where we plot the data
    id_data = np.linspace(-2,2,len(data_type))
      
    # over over the different data types
    for count_data,data in enumerate(data_type):
        # loop over the different methods
        for count_model, model in enumerate(legends):
            # we plot 1-data becuase data stores the coverage
            # alpha is the opatcity level of the markers
            if count_data == 0:
                plt.plot(id_data[count_data],1-df[data].iloc[0][count_model],leg_marker[model],label = model,markersize = size_marker,alpha = 0.5)
            else:
                # dont add the label for all the other datapoints
                plt.plot(id_data[count_data],1-df[data].loc[0][count_model],leg_marker[model],markersize = size_marker,alpha = 0.5)
        

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(id_data, data_ticks,rotation=25)
    plt.grid()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.title('Results for ' + model_type)
    plt.ylabel('miss-coverages',fontsize = 15)
    plt.xlabel('datasets',fontsize = 15)
    plt.savefig(output_filename, bbox_inches='tight')

# %%
"""
## Results ##
"""

# %%
print('plotting for iid and NN')
plot_errors(model_type = "NN",shift = False)
print('-'*50)
print('plotting for covariate shift and NN')
plot_errors(model_type = "NN", shift = True)
print('-'*50)
print('plotting for iid and MVE NN')
plot_errors(model_type = "MVE_NN",shift = False)
print('-'*50)
print('plotting for covariate shift and MVE NN')
plot_errors(model_type = "MVE_NN",shift = True)
