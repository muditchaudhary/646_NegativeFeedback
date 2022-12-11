import json
import argparse
from tqdm import tqdm
from collections import defaultdict


def eval(args):

    with open(args.subset_file) as f:
        subset_ids = json.load(f)

    input_data = [line for line in open(args.input_file)]

    print("Ignoring questions with no relevant document in top-1000")

    MRR = defaultdict(list)
    valid_ids = []

    subset_mrr_dict = {"1-10": [],
                       "1-50": [],
                       "100-1000": [],
                       "500-1000": [],
                       "800-1000": []}

    for line in tqdm(input_data, desc="Evaluating", position=0, leave=True, ascii=True, ncols=90):
        data = json.loads(line)

        for rank, passage_id in enumerate(data[f"iter_1"]):
            if passage_id in data["relevant_ids"]:
                for key in subset_mrr_dict.keys():
                    if data["query_id"] in subset_ids[key]:
                        subset_mrr_dict[key].append(1/(rank+1))

                break


    for key in subset_mrr_dict:
        subset_mrr_dict[key] = sum(subset_mrr_dict[key])/len(subset_mrr_dict[key])

    print("=========RESULTS===========")
    print("MRR in subsets")
    print(subset_mrr_dict)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args.add_argument("--subset_file", type=str, required=True)
    args = args.parse_args()
    eval(args)
