import pickle
import ipdb
from tqdm import tqdm
import json
import argparse

data = pickle.load(open("train_dataset_relevance_cut_0.pkl.pkl2","rb"))
out_file = open("train_dataset_relevance_split0.jsonl", "w")

def convert(data, out_file):
    for key, value in tqdm(data.items(), total=len(data)):
        value["query_id"] = key
        out_string=json.dumps(value)
        out_file.write(out_string.rstrip('\n') + '\n')

    out_file.close()

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args.add_argument("--output_file", type=str, required=True)
    args = args.parse_args()

    data = pickle.load(open(args.input_file,"rb"))
    out_file = open(args.output_file, "w")

    convert(data, out_file)


