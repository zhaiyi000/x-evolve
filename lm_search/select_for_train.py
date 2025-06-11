import natsort
import glob
import json
import shutil

file_path = '/root/funsearch/bin_packing_train_4_100/funsearch_llm_api/samples/*.json'
destination_path = '/root/funsearch/lm_search/llm_train_4_200'

def get_top_n_indices(lst, n):
    sorted_indices = sorted(lst, key=lambda x: x[0], reverse=True)[:n]
    return [value for index, value in sorted_indices]


def get_data(num):
    files = []
    files.extend(glob.glob(file_path))
    files = natsort.natsorted(files)

    score_list = []

    max_score = -1e10
    visited_decisions: dict[str, set[tuple[str]]] = {}

    for file in files:
        try:
            with open(file, 'r') as f:
                info = json.load(f)
        except Exception as e:
            print(f"error: {e}")
            
        score = info['score']
        if score:
            score_list.append((score, info['sample_order']))
        
    sample_list = get_top_n_indices(score_list, num)
    
    for idx in sample_list:
        file_to_copy = files[idx]
        shutil.copy(file_to_copy, destination_path)
    
if __name__ == "__main__":
    get_data(200)
        
        

        