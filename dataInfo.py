import pandas as pd 


df = pd.read_csv('dataset/Training_set.csv')

dict = {}
for label in df['label']:
    dict[label] = dict.get(label, 0) + 1
    
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
    print(k, dict[k])

print("Lowest:\n", lowest_key, lowest)
print("Highest:\n", highest_key, highest)
print("Average:\n", sum(dict.values())/len(dict))