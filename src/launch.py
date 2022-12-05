from src.data.NegativeFeedbackDataset import NegativeFeedbackDataset
from torch.utils.data import DataLoader
from src.trainer.trainer import Trainer
import argparse
import ipdb

if __name__ == "__main__":
    # adding defaults to make running easier
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_folder", type=str, required=True)
    args.add_argument("--neg_sampling_ranker", type=str, default="bm25")
    args.add_argument("--neg_sample_rank_from", type=int, required=True)
    args.add_argument("--neg_sample_rank_to", type=int, required=True)
    args.add_argument("--num_neg_samples", type=int, required=True) # sample 5 values from the above range
    args.add_argument("--cached_embeddings_root", type=str, required=True)
    args.add_argument("--mode", type=str, required=True)
    args.add_argument("--embedding_type", type=str, default="normalized_embedding")

    # trainer args
    args.add_argument("--eval_only", type=bool, default=False)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--warmup_steps", type=int, default=0)
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--train_batch_size", type=int, default=3)
    args.add_argument("--checkpoint", type=str, default=None)
    args.add_argument("--clip_grad", type=float, default=None)
    args.add_argument("--eval_batch_size", type=int, default=2)
    args.add_argument("--alpha1", type=float, default=1.0)
    args.add_argument("--alpha2", type=float, default=1.0)
    args.add_argument("--alpha3", type=float, default=1.0)

    args = args.parse_args()

    # NFdataset = NegativeFeedbackDataset(args, 'dev')
    # NFDataloader = DataLoader(NFdataset, batch_size=1, shuffle=False)


    # Trainer arguments
    trainer = Trainer(args)
    trainer.train()
    ipdb.set_trace()