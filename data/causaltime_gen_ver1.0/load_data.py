import numpy as np
import sys
sys.path.append('../')

data_name = 'traffic'
data = np.load('./data/causaltime_gen_ver1.0/' + data_name + '/gen_data.npy')
graph = np.load('./data/causaltime_gen_ver1.0/' + data_name + '/graph.npy')

print(f"Data Name: {data_name}")
print(f'Shape of Graph H: {graph.shape} (Note that this is only the upper left corner of the adjacency matrix)')
print(f'Shape of Time-series Data: {data.shape} (Sample_num, Time_step, Node_num)')