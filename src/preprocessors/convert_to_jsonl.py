import pickle
import ipdb
from tqdm import tqdm
import json
data = pickle.load(open("train_dataset_relevance_cut_0.pkl.pkl2","rb"))
out_file = open("train_dataset_relevance_split0.jsonl", "w")

for key, value in tqdm(data.items(), total=len(data)):
    value["query_id"] = key
    out_string=json.dumps(value)
    out_file.write(out_string.rstrip('\n') + '\n')

out_file.close()

