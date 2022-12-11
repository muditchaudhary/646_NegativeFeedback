import argparse
from tqdm import tqdm
import json


def eval(args):
    input_data = [line for line in open(args.input_file)]
    subset_id_dict = {"1-10": [],
                      "1-50": [],
                      "100-1000": [],
                      "500-1000": [],
                      "800-1000": []}

    subset_mrr_dict = {"1-10": [],
                       "1-50": [],
                       "100-1000": [],
                       "500-1000": [],
                       "800-1000": []}

    with open("./data/processed_data/valid_dev_ids.json") as f:
        valid_ids = json.load(f)

    print("Ignoring questions with no relevant document in top-1000")
    for line in tqdm(input_data, desc="Evaluating", position=0, leave=True, ascii=True, ncols=90):
        data = json.loads(line)

        if data["query_id"] not in valid_ids:
            continue

        for idx, passage in enumerate(data["retrieved_passages"]):
            if passage["relevance"]:
                rank = idx + 1

                if rank <= 10:
                    subset_id_dict["1-10"].append(data["query_id"])
                    subset_mrr_dict["1-10"].append(1 / rank)

                if rank <= 50:
                    subset_id_dict["1-50"].append(data["query_id"])
                    subset_mrr_dict["1-50"].append(1 / rank)

                if rank >= 100:
                    subset_id_dict["100-1000"].append(data["query_id"])
                    subset_mrr_dict["100-1000"].append(1 / rank)

                if rank >= 500:
                    subset_id_dict["500-1000"].append(data["query_id"])
                    subset_mrr_dict["500-1000"].append(1 / rank)

                if rank >= 800:
                    subset_id_dict["800-1000"].append(data["query_id"])
                    subset_mrr_dict["800-1000"].append(1 / rank)

                break

    for key in subset_mrr_dict.keys():
        subset_mrr_dict[key] = sum(subset_mrr_dict[key]) / len(subset_mrr_dict[key])

    output_file = f"{args.input_file.split('.')[-2]}_subsets.json"
    with open(output_file, "w") as f:
        json.dump(subset_id_dict, f, indent=1)

    print(subset_mrr_dict)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args = args.parse_args()
    eval(args)
