from torch.utils.data import Dataset
import json
import os
import random
import torch
import numpy as np
import pickle


class NegativeFeedbackDataset(Dataset):
    """
    Torch dataset class for Negative Feedback Query Refinement

    """
    def __init__(self, args):
        """
        Constructor method

        """
        self.neg_sampling_ranker = args.neg_sampling_ranker
        self.dataset_split = args.dataset_split
        self.model = args.mode
        self.neg_sample_rank_from = args.neg_sample_rank_from
        self.neg_sample_rank_to = args.neg_sample_rank_to + 1
        self.num_neg_samples = args.num_neg_samples
        self.cached_embeddings_folder = os.path.join(args.cached_embeddings_root, self.dataset_type)

        if self.neg_sampling_ranker == "bm25":
            self.dataset_file = os.path.join(args.dataset_folder, f"{self.dataset_split}_bm25.jsonl")
        else:
            self.dataset_file = os.path.join(args.dataset_folder, f"{self.dataset_split}_dpr.jsonl")

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
            found = False
            positive_passage_ids = []
            all_passage_ids = []
            for passage in data["retrieved_passages"]:
                all_passage_ids.append(passage["passage_id"])
                if passage["relevance"]:
                    positive_passage_id.append(passage["passage_id"])
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
        negative_samples = self.sample_negatives(datapoint)
        query_embedding = self.get_cached_embeddings([datapoint["query_id"]], "query")[0]
        relevant_passage_embeddings = self.get_cached_embeddings(datapoint["relevant_passage_ids"], "passage")
        negative_passage_embeddings = self.get_cached_embeddings(negative_samples, "passage")

        return torch.from_numpy(query_embedding), torch.from_numpy(relevant_passage_embeddings), torch.from_numpy(
            negative_passage_embeddings)

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
        else:
            negative_candidates = datapoint["all_passage_ids"][self.neg_sample_rank_from:self.neg_sample_rank_to]

        return random.choices(list(negative_candidates), k=self.num_neg_samples)

    def get_cached_embeddings(self, id_list, data_type):
        """
        Gets cached embeddings from the cache folder

        returns:
        cached_embeddings: numpy.ndarray

        """
        cache_folder = os.path.join(self.cached_embeddings_folder, data_type, "dpr")

        embeddings = []
        for idx in id_list:
            cache_path = os.path.join(cache_folder, f"{idx}.embed")
            with open(cache_path, "rb") as fEmbed:
                embedding = pickle.load(fEmbed)
                embeddings.append(embedding)

        return np.asarray(embeddings)
