def parse_binpack(datasets, file_path, or_idx):
    datasets[f'OR{or_idx}'] = {}
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # 移除空行和空白
        
        idx = 0
        total_instances = int(lines[idx])
        idx += 1
        
        while idx < len(lines):
            # 读取实例名称
            instance_name = lines[idx].strip()
            idx += 1
            
            # 读取元数据行：num_items capacity param
            metadata = list(map(int, lines[idx].split()))
            # print(metadata)
            capacity, num_items, _ = metadata  # 198参数未在输出中使用
            idx += 1
            
            # 读取物品列表
            items = []
            for _ in range(num_items):
                if idx >= len(lines):
                    break
                items.append(int(lines[idx]))
                idx += 1
            
            # 存入datasets
            datasets[f'OR{or_idx}'][instance_name] = {
                'capacity': capacity,
                'num_items': num_items,
                'items': items
            }
    

# 使用示例
datasets = {}
parse_binpack(datasets, 'binpack1.txt', 1)
parse_binpack(datasets, 'binpack2.txt', 2)
parse_binpack(datasets, 'binpack3.txt', 3)
parse_binpack(datasets, 'binpack4.txt', 4)
