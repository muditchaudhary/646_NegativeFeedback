"""
Ranks using DPR and also stores the embeddings in one go
"""
import pickle

import pysos
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import argparse
import os


def rank_and_encode(input_data, output_file, query_cache_folder, passage_cache_folder):
    passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-multiset-base')
    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-multiset-base')

    pool = passage_encoder.start_multi_process_pool()
    for line in tqdm(input_data, total=len(input_data), desc="Running DPR"):
        data = json.loads(line)
        passages = [passage['passage'] for passage in data['retrieved_passages']]

        passage_embeddings = passage_encoder.encode_multi_process(passages, pool)
        query_embedding = query_encoder.encode(data["query"])

        normalized_passage_embeddings = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=0, keepdims=True)

        query_embedding_out = {"embedding": query_embedding, "normalized_embedding": normalized_query_embedding}
        with open(os.path.join(query_cache_folder, f'{data["query_id"]}.embed'), "wb") as queryF:
            pickle.dump(query_embedding_out, queryF, protocol=pickle.HIGHEST_PROTOCOL)

        for idx, passage in enumerate(data["retrieved_passages"]):
            passage_embedding_out = {"embedding": passage_embeddings[idx],
                                     "normalized_embedding": normalized_passage_embeddings[idx]}
            if not os.path.exists(os.path.join(passage_cache_folder, f'{passage["passage_id"]}.embed')):
                with open(os.path.join(passage_cache_folder, f'{passage["passage_id"]}.embed'), "wb") as passageF:
                    pickle.dump(passage_embedding_out, passageF, protocol=pickle.HIGHEST_PROTOCOL)

        dpr_scores = util.dot_score(query_embedding, passage_embeddings)

        dpr_ctx = []
        for idx, ctx in enumerate(data["retrieved_passages"]):
            ctx["dpr_score"] = float(dpr_scores[0][idx])
            dpr_ctx.append(ctx)

        dpr_ctx = sorted(dpr_ctx, key=lambda x: x["dpr_score"], reverse=True)
        data["retrieved_passages"] = dpr_ctx
        out_string = json.dumps(data).rstrip('\n')
        output_file.write(out_string + "\n")


    passage_encoder.stop_multi_process_pool(pool)
    output_file.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args.add_argument("--query_cache", type=str, required=True)
    args.add_argument("--passage_cache", type=str, required=True)
    args = args.parse_args()
    output_filename = f'{args.input_file.split(".")[0]}_dpr.jsonl'

    if not os.path.exists(args.query_cache):
        os.makedirs(args.query_cache)

    if not os.path.exists(args.passage_cache):
        os.makedirs(args.passage_cache)

    input_file = open(args.input_file, encoding='utf-8')
    input_data = [line for line in input_file]
    output_file = open(output_filename, "w")

    rank_and_encode(input_data, output_file,query_cache_folder=args.query_cache, passage_cache_folder=args.passage_cache)


