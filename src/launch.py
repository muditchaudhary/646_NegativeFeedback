from src.data.NegativeFeedbackDataset import NegativeFeedbackDataset
from torch.utils.data import DataLoader
from src.trainer.trainer import Trainer
import argparse
import ipdb
import wandb
if __name__ == "__main__":
    # adding defaults to make running easier
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_folder", type=str, required=True)
    args.add_argument("--neg_sampling_ranker", type=str, default="bm25")
    args.add_argument("--neg_sample_rank_from", type=int, required=True)
    args.add_argument("--neg_sample_rank_to", type=int, required=True)
    args.add_argument("--num_neg_samples", type=int, required=True) # sample 5 values from the above range
    args.add_argument("--cached_embeddings_root", type=str, required=True)
    args.add_argument("--embedding_type", type=str, default="normalized_embedding")
    args.add_argument("--save_model_root", type=str, required=True)
    args.add_argument("--max_refining_iterations", type=int, default=7)
    args.add_argument("--partial_eval_steps", type=int, default=None)
    args.add_argument("--use_wandb", type=bool, default=False)
    args.add_argument("--log_cossim", type=bool, default=True)
    args.add_argument("--delta_learning", type=bool, default=False)
    args.add_argument("--save_preds_root", type=str, required=True)

    # trainer args
    args.add_argument("--eval_only", action='store_true')
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--warmup_percent", type=float, default=0.0)
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--train_batch_size", type=int, default=3)
    args.add_argument("--checkpoint", type=str, default=None)
    args.add_argument("--clip_grad", type=float, default=None)
    args.add_argument("--eval_batch_size", type=int, default=1)
    args.add_argument("--alpha1", type=float, default=1.0)
    args.add_argument("--alpha2", type=float, default=1.0)
    args.add_argument("--alpha3", type=float, default=1.0)

    args = args.parse_args()

    if args.use_wandb:
        wandb.init(project="646-NegativeFeedback")
        wandb.config.update(args)

    # Trainer arguments
    trainer = Trainer(args)

    if not args.eval_only:
        trainer.train()
    else:
        trainer.eval()
