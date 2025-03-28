import os
import sys
import numpy as np
import pandas as pd



import os
import sys
import pandas as pd

base_path = 'C:\\Users\\Matth\\Documents\\Python_Scripts\\Ignimbrite_Project\\Location4_8\\'

only_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]


# there are drop and no_drop versions

# variations:

# drop
# nodrop

# all_phases : all phases
# phases-Z_G : no zircon or glass
# phases - Z_G_C_S : no zircon glass clino or san

# dropout - ammount of missing data : 0,.1,.2,.3,.4,.5,.6,.7,.75,.8,.9

# drop column says which element was eliminated with some dropout probability

# n_samples - 5,10,20,100,200,300,500,1000 : number of bootstrapped observations
# n_trees - 100, 200, 300, 400, 500, 600, 700

# all models were run for 100 iterations at 70% training data 30% test data

for t_file in only_files:
    t_df = pd.read_csv(base_path + '\\' + t_file)
    t_df = t_df.iloc[:,1:]
    t_df.to_csv(base_path + '\\' + t_file,index=False)
    
    
#%%


import numpy as np
import pandas as pd

base_path = 'C:/Users/Matth/Documents/Python_Scripts/Ignimbrite_Project/'

cluster_path = 'ClusteringData_All_062623_modified.xlsx'
machine_path = 'MachineLearningData_060523_ILR_norm.xlsx'


dropout_probs = [0,.1,.2,.3,.4,.5,.6,.7,.75,.8,.9]
n_trees = [100,200,300,400,500,600,700]

sample_list = [5,10,20,100,200,300,500,1000]
iter_list = [100]
save_responses = True
percent_train = .7


phases = ['all_phases','phases-Z_G','phases-Z_G_C_S']
save_location = 'Location4_8'

#%%
# one example:
    
def retrieve_data(full_conf):
    
    columns = []
    data = []
    
    locations = list(full_conf.loc[:,'index'])
    
    A = list(full_conf.precision)
    temp_col = [x + '_precision' for x in locations]
    columns.extend(temp_col)
    data.extend(A)
    
    A = list(full_conf.recall)
    temp_col = [x + '_recall' for x in locations]
    columns.extend(temp_col)
    data.extend(A)
    
    A = list(full_conf.loc[:,'local_f1'])
    temp_col = [x + '_local_f1' for x in locations]
    columns.extend(temp_col)
    data.extend(A)
    
    data.extend([full_conf.loc[0,'micro f1']])
    data.extend([full_conf.loc[0,'macro f1']])
    columns.extend(['micro_f1','macro_f1'])
    
    data.extend([phase,prob,sample,tree,full_conf.loc[0,'iteration']])
    columns.extend(['phase','dropout','samples','trees','iteration'])
    
    temp_data = pd.DataFrame(data).T
    temp_data.columns = columns
    
    return(temp_data)
#%%  

from copy import copy

aggregated_local_f1 = []
aggreagted_local_f1_2 = []
aggregated_local_f1_3 = []

for prob in dropout_probs:
    for tree in n_trees:
        for sample in sample_list:
            for phase in phases:
                try:
                    full_conf = pd.read_csv(base_path + save_location + '/' + 'drop_{}_dropout_{}_n_samples_{}_n_trees_{}.csv'.format(phase,prob,sample,tree)) 
                    #temp_data = retrieve_data(full_conf)
                    
                    if 'Unnamed: 0' in list(full_conf.columns):
                            
                        A = full_conf.groupby(['index','drop']).mean().drop('Unnamed: 0',axis=1).drop('iteration',axis=1).reset_index()
                        B = full_conf.groupby(['index','drop']).std().drop('Unnamed: 0',axis=1).drop('iteration',axis=1).reset_index()
                    else:
                        A = full_conf.groupby(['index','drop']).mean().drop('iteration',axis=1).reset_index()
                        B = full_conf.groupby(['index','drop']).std().drop('iteration',axis=1).reset_index()
                    
                    orig_columns = list(A.columns)
                    orig_columns[0] = 'true_location'
                    
                    sum_columns = [x+'_mean' for x in orig_columns]
                    std_columns = [x+'_std' for x in orig_columns]
                    
                    A.columns = sum_columns
                    B.columns = std_columns
                    
                    combined_columns = []
                    for x,y in zip(sum_columns,std_columns):
                        combined_columns.append(x)
                        combined_columns.append(y)
                    
                    C = pd.concat([A,B],axis=1)
                    
                    D = copy(C)
                    C = C[combined_columns]
                    
                    C['phase'] = phase
                    C['dropout'] = prob
                    C['samples'] = sample
                    C['trees'] = tree
                    D['phase'] = phase
                    D['dropout'] = prob
                    D['samples'] = sample
                    D['trees'] = tree
                    
                    
                    E = C.groupby('drop_mean').mean().reset_index()
                    E = E.loc[:,['drop_mean','support_mean','micro f1_mean','micro f1_std','macro f1_mean','macro f1_std','weighted f1_mean','weighted f1_std','dropout','samples','trees']]
                    E['phase'] = phase
                    aggregated_local_f1.append(C)
                    aggreagted_local_f1_2.append(D)
                    aggregated_local_f1_3.append(E)
                except:
                    print(base_path + save_location + '/' + 'drop_{}_dropout_{}_n_samples_{}_n_trees_{}.csv'.format(phase,prob,sample,tree))

agg_df = pd.concat(aggregated_local_f1).reset_index(drop=True)
agg_df2 = pd.concat(aggreagted_local_f1_2).reset_index(drop=True)
agg_df3 = pd.concat(aggregated_local_f1_3).reset_index(drop=True)

agg_df.to_csv(base_path + 'location4_8_final/' +'all_scores.csv')
agg_df2.to_csv(base_path + 'location4_8_final/' +'all_scores_alt.csv')
agg_df3.to_csv(base_path + 'location4_8_final/' +'aggregated_scores.csv')

#










