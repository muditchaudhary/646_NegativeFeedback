from torch.utils.data import Dataset
import json
import os
import random
import torch
import numpy as np
import pickle
import multiprocessing
from functools import partial


class NegativeFeedbackDataset(Dataset):
    """
    Torch dataset class for Negative Feedback Query Refinement

    """

    def __init__(self, args, mode, dataset_split, workers=0):
        """
        Constructor method

        """
        self.neg_sampling_ranker = args.neg_sampling_ranker
        self.dataset_split = dataset_split
        self.mode = mode
        self.workers = workers
        self.neg_sample_rank_from = args.neg_sample_rank_from
        self.neg_sample_rank_to = args.neg_sample_rank_to + 1
        self.num_neg_samples = args.num_neg_samples
        self.cached_embeddings_folder = os.path.join(args.cached_embeddings_root, self.dataset_split)
        self.embedding_type = args.embedding_type
        if self.neg_sampling_ranker == "bm25":
            self.dataset_file = os.path.join(args.dataset_folder, f"{self.dataset_split}_bm25.jsonl")
        else:
            self.dataset_file = os.path.join(args.dataset_folder, f"{self.dataset_split}_dpr.jsonl")
        print(self.dataset_file)
        self.dataset = []
        self.prepare_dataset()

    def prepare_dataset(self):
        """
        Prepare dataset for dataloader
        1. Filters out the queries that do not have relevant passage in top-1000
        2. Only stores query_ids and passage_ids (As embeddings are already cached, no encoding of text is necessary)

        """
        try:
            dataset_IO = open(self.dataset_file, encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError("Dataset file not found")

        for line in dataset_IO:
            data = json.loads(line)
            if len(data["retrieved_passages"]) < 1000:
                continue
            found = False
            positive_passage_ids = []
            all_passage_ids = []
            for passage in data["retrieved_passages"]:
                all_passage_ids.append(passage["passage_id"])
                if passage["relevance"] and not found:
                    positive_passage_ids.append(passage["passage_id"])
                    found = True

            if found:
                datapoint = {"query_id": data["query_id"],
                             "relevant_passage_ids": positive_passage_ids,
                             "all_passage_ids": all_passage_ids}
                self.dataset.append(datapoint)

    def __len__(self):
        """
        Get length method

        returns:
        length: int
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get single item method

        returns:
        query_embedding: torch.Tensor
        relevant_passage_embeddings: torch.Tensor
        negative_passage_embeddings: torch.Tensor

        """
        datapoint = self.dataset[idx]
        query_embedding = self.get_cached_embeddings([datapoint["query_id"]], "query")[0]
        if self.mode == "train":
            negative_samples = self.sample_negatives(datapoint)
            relevant_passage_embeddings = self.get_cached_embeddings(datapoint["relevant_passage_ids"], "passage")
            negative_passage_embeddings = self.get_cached_embeddings(negative_samples, "passage")
            return torch.from_numpy(query_embedding), torch.from_numpy(relevant_passage_embeddings), torch.from_numpy(
                negative_passage_embeddings)
        elif self.mode == "eval":
            all_passage_embeddings = self.get_cached_embeddings(datapoint["all_passage_ids"], "passage")
            all_passage_ids = np.asarray(datapoint["all_passage_ids"])
            relevant_passage_ids = np.asarray(datapoint["relevant_passage_ids"])
            return torch.from_numpy(query_embedding),all_passage_embeddings, all_passage_ids, relevant_passage_ids
        else:
            raise NotImplementedError

    def sample_negatives(self, datapoint):
        """
        Negative sampler.
        1. Randomly sample self.num_neg_samples from the provided range. We use range for versatility.
        2. During train mode, we remove the positive relevant passage from the candidate set.
        3. During eval mode, we remove the positive relevant passage from the candidate set.

        returns:
        negative_samples: List
        """

        if self.mode == "train":
            negative_candidates = set(
                datapoint["all_passage_ids"][self.neg_sample_rank_from:self.neg_sample_rank_to]) - set(
                datapoint["relevant_passage_ids"])

            negative_candidates = random.choices(list(negative_candidates), k=self.num_neg_samples)
        elif self.mode == "eval":
            negative_candidates = datapoint["all_passage_ids"] - set(datapoint["relevant_passage_ids"])
        else:
            raise NotImplementedError

        return negative_candidates

    def get_cached_embeddings(self, id_list, data_type):
        """
        Gets cached embeddings from the cache folder

        returns:
        cached_embeddings: numpy.ndarray

        """
        cache_folder = os.path.join(self.cached_embeddings_folder, data_type, "dpr")

        embeddings = []

        # @dhawal :using default embeddings
        # for idx in id_list:
        #     cache_path = os.path.join(cache_folder, f"default.embed")
        #     with open(cache_path, "rb") as fEmbed:
        #         embedding = pickle.load(fEmbed)
        #         embeddings.append(embedding[self.embedding_type])
        # return np.asarray(embeddings)
        # the above code is only for debugging

        if self.workers > 0:
            # Parallelized code for IO. Only useful if a significant number of IO ops required
            global embedding_loader_parallel

            def embedding_loader_parallel(doc_id, cache_folder, embedding_type):
                cache_path = os.path.join(cache_folder, f"{doc_id}.embed")
                with open(cache_path, "rb") as fEmbed:
                    return pickle.load(fEmbed)[embedding_type]

            parallel_loader_partial = partial(embedding_loader_parallel, cache_folder=cache_folder,
                                              embedding_type=self.embedding_type)
            workers = self.workers
            pool = multiprocessing.Pool(workers)
            embeddings = pool.map(parallel_loader_partial, id_list)
            pool.close()
            pool.join()
            del embedding_loader_parallel

            assert len(embeddings) == len(id_list)
        else:
            for idx in id_list:
                cache_path = os.path.join(cache_folder, f"{idx}.embed")
                with open(cache_path, "rb") as fEmbed:
                    embedding = pickle.load(fEmbed)
                    embeddings.append(embedding[self.embedding_type])

        return np.asarray(embeddings)
