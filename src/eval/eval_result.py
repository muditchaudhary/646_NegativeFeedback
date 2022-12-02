import json
import argparse
from tqdm.auto import tqdm
import sys


def evaluate(input_data):
    hits_at_k = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0, 200: 0, 500: 0, 700: 0, 900: 0}
    reciprocal_ranks = []
    num_questions_no_relevant = 0

    print("Ignoring questions with no relevant document in top-1000")
    for line in tqdm(input_data, desc="Evaluating", position=0, leave=True, ascii=True, ncols=90):
        data = json.loads(line)

        found = False
        for idx, passage in enumerate(data["retrieved_passages"]):
            if passage["relevance"] == True:
                rank = idx + 1
                reciprocal_ranks.append(1 / rank)
                for key in hits_at_k.keys():
                    if rank <= key:
                        hits_at_k[key] += 1

                found = True
                break

        if not found:
            num_questions_no_relevant += 1

    print("=====Evaluation Summary=====")
    print(f"Total number of questions: {len(input_data)}")
    print(f"Total number of questions with retrieved passage in top-1000: {len(reciprocal_ranks)}")
    print(f"Number of questions with no relevant passage in top-1000: {num_questions_no_relevant}")
    print(f"Mean reciprocal rank: {sum(reciprocal_ranks) / len(reciprocal_ranks)}")
    print("===Hits@K===")
    for key, value in hits_at_k.items():
        print(f"hits@{key} : {value}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args = args.parse_args()

    input_data = [line for line in open(args.input_file)]

    evaluate(input_data)
