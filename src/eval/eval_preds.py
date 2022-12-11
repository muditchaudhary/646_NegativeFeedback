import json
import argparse
from tqdm import tqdm
from collections import defaultdict


def eval(args):
    input_data = [line for line in open(args.input_file)]

    print("Ignoring questions with no relevant document in top-1000")

    MRR = defaultdict(list)
    valid_ids = []
    for line in tqdm(input_data, desc="Evaluating", position=0, leave=True, ascii=True, ncols=90):
        data = json.loads(line)
        relevant_ids = data["relevant_ids"]

        for iteration in range(args.max_iters):
            for rank, passage_id in enumerate(data[f"iter_{iteration + 1}"]):
                if passage_id in relevant_ids:
                    MRR[iteration + 1].append(1 / (rank + 1))
                    valid_ids.append(data["query_id"])
                    break


    for key in MRR:
        MRR[key] = sum(MRR[key])/len(MRR[key])

    print("=========RESULTS===========")
    print("MRR @ Iteration K")
    print(MRR)

    with open("./data/processed_data/valid_dev_ids.json", "w") as f:
        json.dump(valid_ids, f)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args.add_argument("--max_iters", type=int, required=True)
    args = args.parse_args()
    eval(args)
