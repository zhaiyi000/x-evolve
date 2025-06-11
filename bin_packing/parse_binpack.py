def parse_binpack(datasets, file_path, or_idx):
    datasets[f'OR{or_idx}'] = {}
    
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
            datasets[f'OR{or_idx}'][instance_name] = {
                'capacity': capacity,
                'num_items': num_items,
                'items': items
            }
    
def parse_binpack_Weibull(datasets, file_path, Weibull_idx):
    datasets[f'Weibull_{Weibull_idx}'] = {}
    
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
            datasets[f'Weibull_{Weibull_idx}'][instance_name] = {
                'capacity': capacity,
                'num_items': num_items,
                'items': items
            }

datasets = {}
parse_binpack(datasets, 'binpack1.txt', 1)
parse_binpack(datasets, 'binpack2.txt', 2)
parse_binpack(datasets, 'binpack3.txt', 3)
parse_binpack(datasets, 'binpack4.txt', 4)

parse_binpack_Weibull(datasets, 'Weibull_5k.txt', '5k')
parse_binpack_Weibull(datasets, 'Weibull_10k.txt', '10k')
parse_binpack_Weibull(datasets, 'Weibull_100k.txt', '100k')