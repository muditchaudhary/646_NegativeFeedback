from src.data.NegativeFeedbackDataset import NegativeFeedbackDataset
from torch.utils.data import DataLoader
import argparse
import ipdb

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_folder", type=str, required=True)
    args.add_argument("--neg_sampling_ranker", type=str, default="bm25")
    args.add_argument("--neg_sample_rank_from", type=int, required=True)
    args.add_argument("--neg_sample_rank_to", type=int, required=True)
    args.add_argument("--num_neg_samples", type=int, required=True)
    args.add_argument("--cached_embeddings_root", type=str, required=True)
    args.add_argument("--mode", type=str, required=True)
    args.add_argument("--embedding_type", type=str, default="embedding")
    args = args.parse_args()

    NFdataset = NegativeFeedbackDataset(args)
    NFDataloader = DataLoader(NFdataset, batch_size=1, shuffle=False)

    ipdb.set_trace()