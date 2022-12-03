from rank_bm25 import BM25Okapi
from string import punctuation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import argparse
from tqdm import tqdm
import json
import multiprocessing
from functools import partial

"""
Ranks using BM25 
"""
def parallel_ranker(line, stoplist, stemmer):
    data = json.loads(line)
    passages = [passage['passage'].lower() for passage in data['retrieved_passages']]
    tokenized_corpus = []
    for text in passages:
        tokenized_doc = [stemmer.stem(token) for token in word_tokenize(text) if token not in stoplist]
        tokenized_corpus.append(tokenized_doc)

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = [stemmer.stem(token) for token in word_tokenize(data["query"].lower()) if token not in stoplist]
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ctx = []
    for idx, ctx in enumerate(data["retrieved_passages"]):
        ctx["bm25_score"] = bm25_scores[idx]
        bm25_ctx.append(ctx)

    sorted_bm25_ctx = sorted(bm25_ctx, key=lambda x: x["bm25_score"], reverse=True)

    data["retrieved_passages"] = sorted_bm25_ctx
    out_string = json.dumps(data).rstrip('\n')
    return out_string


def rankBM25(input_data, output_file):
    stoplist = set(stopwords.words('english') + list(punctuation))
    ps = PorterStemmer()

    parallel_ranker_partial = partial(parallel_ranker, stoplist=stoplist, stemmer=ps)
    workers = 14
    pool = multiprocessing.Pool(workers)
    result_lines = list(tqdm(pool.imap(parallel_ranker_partial, input_data), total=len(input_data), desc="Running BM25"))
    pool.close()
    pool.join()

    assert len(input_data) == len(result_lines), "Incorrect output size"
    for line in tqdm(result_lines, total=len(result_lines), desc="Writing data"):
        output_file.write(line + '\n')
    output_file.close()
    print(f"Processed {len(result_lines)} queries")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", type=str, required=True)
    args = args.parse_args()

    input_file = open(args.input_file, encoding='utf-8')
    input_data = [line for line in input_file]
    output_filename = f'{args.input_file.split(".")[0]}_bm25.jsonl'
    output_file = open(output_filename, "w")
    rankBM25(input_data, output_file)
