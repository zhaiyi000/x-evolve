import random
import json
import numpy as np

def parse_binpack(datasets, file_path, dataset_name):
    datasets[dataset_name] = {}
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
        idx = 0
        total_instances = int(lines[idx])
        idx += 1
        
        while idx < len(lines):
            instance_name = lines[idx].strip()
            idx += 1
            
            # num_items capacity param
            metadata = list(map(int, lines[idx].split()))
            # print(metadata)
            capacity, num_items, _ = metadata 
            idx += 1
            
            items = []
            for _ in range(num_items):
                if idx >= len(lines):
                    break
                items.append(int(lines[idx]))
                idx += 1
            
            # datasets
            datasets[dataset_name][instance_name] = {
                'capacity': capacity,
                'num_items': num_items,
                'items': items
            }
    
    with open('bin_packing_test_Weibull.py','a') as file:
        file.write(f'datasets[\'{dataset_name}\']=')
        json.dump(datasets[dataset_name],file)
        file.write('\n')

def gen_OR_data(data_path, num_instances, capacity, num_items):
    with open(data_path,'w') as file:
        file.write(str(num_instances)+'\n')
        for i in range(num_instances):
            file.write(f'u{num_items}_'+str(i).zfill(len(str(num_instances)))+'\n')
            file.write(str(capacity)+' '+str(num_items)+' '+'0'+'\n')
            random_integers = [random.randint(20, 100) for _ in range(num_items)]
            for item in random_integers:
                file.write(str(item)+'\n')

def gen_Weibull_data(data_path, num_instances, capacity, num_items):
    with open(data_path,'w') as file:
        file.write(str(num_instances)+'\n')
        for i in range(num_instances):
            file.write(f'u{num_items}_'+str(i).zfill(len(str(num_instances)))+'\n')
            file.write(str(capacity)+' '+str(num_items)+' '+'0'+'\n')
            shape = 3
            scale = 45
            random_values = np.random.weibull(shape, num_items) * scale
            # 1-100
            random_integers = np.clip(random_values, 1, 100).astype(int).tolist()
            for item in random_integers:
                file.write(str(item)+'\n')
    
if __name__ == '__main__':
    if_gen_data = True
    train_data_path = '/root/X_evolve/bin_packing/test_data/Weibull_100k.txt'
    num_instances = 1
    capacity = 100
    num_items = 100000
    datasets = {}
    if if_gen_data:
        # gen_OR_data(train_data_path, num_instances, capacity, num_items)
        gen_Weibull_data(train_data_path, num_instances, capacity, num_items)
    parse_binpack(datasets, train_data_path, 'Weibull_100k')
    

