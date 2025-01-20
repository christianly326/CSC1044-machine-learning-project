import pandas as pd 


df = pd.read_csv('dataset/Training_set.csv')

dict = {}

dict_names = {}
for idx, label in enumerate(df['label']):
    dict[label] = dict.get(label, 0) + 1
    if label not in dict_names:
        dict_names[label] = []
    else:
        dict_names[label].append(df['filename'][idx])
lowest = max(dict.values())
highest = min(dict.values())
lowest_key = ""
highest_key = ""
for k in dict.keys():
    if lowest > dict[k]:
        lowest = dict[k]
        lowest_key = k
    if highest < dict[k]:
        highest = dict[k]
        highest_key = k
# print(k, dict[k])

# print("Lowest:\n", lowest_key, lowest)
# print("Highest:\n", highest_key, highest)
# print("Average:\n", sum(dict.values())/len(dict))

# print("Amount of classes\n", len(dict))

for k, v in dict_names.items():
    print(k, v[:10], "\n")