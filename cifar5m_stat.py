import json
import numpy as np

def write_json(new_data, filename='cifar5m.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["cifar-5m"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

for label in range(10):
    merged_data = {}
    merged_data['X'] = []
    merged_data['Y'] = []

    for i in range(6):
        file = '/scratch/vvh_root/vvh1/fyw/data/cifar5m_part' + str(i) + '.npz'
        data_part = np.load(file)
        for (x, y) in zip(data_part['X'], data_part['Y']):
            if y == label:
                merged_data['X'].append(x)
                merged_data['Y'].append(y)

    merged_data['X'] = np.array(merged_data['X']) / 255
    merged_data['Y'] = np.array(merged_data['Y']) / 255
    
    mean = merged_data['X'].mean(axis = (0, 1, 2)).tolist()
    std = merged_data['X'].std(axis = (0, 1, 2)).tolist()
    
    dic = {
        'label': label,
        'mean': mean,
        'std': std
    }
    
    write_json(dic)