import json

with open(f'/root/X-evolve/bin_packing/best_train_OR_newprompt.json','r') as file:
    data = json.load(file)
    for idx,item in enumerate(data):
        print(item[0])
        with open(f'/root/X-evolve/bin_packing/OR_newprompt/program_{idx}.py','w') as file:
            file.write(item[1])
print()