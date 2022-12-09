import json
from src.model.RefinerModel import RefinerModel
from src.data.NegativeFeedbackDataset import NegativeFeedbackDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from src.loss.rocchio import RochhiosLoss
import torch
import transformers
import os
import numpy as np
from sentence_transformers import util
from collections import defaultdict
import wandb

class Trainer():
    """
    Class for training and eval

    """

    def __init__(self, args):
        """
        Constructor for trainer.

        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.args = args

        self.model = RefinerModel(args).to(self.device)

        self.model_name = f"save_model_{self.args.neg_sampling_ranker}_{self.args.neg_sample_rank_from}_{self.args.neg_sample_rank_to}_{self.args.num_neg_samples}"
        if self.args.use_wandb:
            self.model_name = f"saved_model_{wandb.run.name}"
        self.trainable_params = self.model.parameters()
        if self.args.checkpoint is not None:
            try:
                self.model.load_state_dict(torch.load(self.args.checkpoint))
            except:
                raise ValueError("Checkpoint config does not match model config")

        if not self.args.eval_only:
            train_dataset = NegativeFeedbackDataset(self.args, mode="train", dataset_split="train", workers=0)
            self.train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

            self.optimizer = optim.AdamW(self.trainable_params, lr=self.args.learning_rate)

            num_warmup_steps = self.args.warmup_percent * (len(self.train_dataloader) * self.args.epochs)
            self.lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                             num_warmup_steps=num_warmup_steps,
                                                                             num_training_steps=len(
                                                                                 self.train_dataloader) * self.args.epochs)

        dev_dataset = NegativeFeedbackDataset(self.args, mode="eval", dataset_split="dev", workers=12)
        self.dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                         collate_fn=lambda x: x)

    def train(self):
        """
        Training function

        """
        best_MRR = -float("inf")
        for epoch in range(self.args.epochs):
            self.model.train()
            for data in tqdm(self.train_dataloader, total=len(self.train_dataloader), ncols=50,
                             desc=f"Training epoch {epoch + 1}/{self.args.epochs}"):

                query_repr, relevant_repr, negative_repr = data
                query_repr = query_repr.to(self.device)
                relevant_repr = relevant_repr.to(self.device)
                negative_repr = negative_repr.to(self.device)
                # query_repr = query embedding, torch.Tensor; shape (train_batch_size, 1, 768)
                # relevant_repr = relevant document embedding, torch.Tensor; (train_batch_size, num relevant doc, 768)
                # negative_repr = negative document embedding, torch.Tensor; (train_batch_size, num negative samples, 768)

                refined_query_repr = self.model(query_repr, negative_repr)
                # Expected model output: Refined query representation torch.Tensor; shape (train_batch_size, 1, 768)

                loss = RochhiosLoss(self.args, refined_query_repr, query_repr, relevant_repr, negative_repr,
                                    self.device, self.args.use_wandb)

                if self.args.use_wandb:
                    wandb.log({"loss": float(loss)})

                loss.backward()

                if self.args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.args.clip_grad)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            mrr = self.eval()
            if self.args.use_wandb:
                wandb.log({f"max_MRR": float(mrr)})
            if mrr > best_MRR:
                best_MRR = mrr

                save_path = os.path.join(self.args.save_model_root, f"{self.model_name}_epoch{epoch + 1}.pt")
                torch.save(self.model, save_path)

    def eval(self):
        """
        Returns MRR

        """
        self.model.eval()
        MRR = defaultdict(list)  # {iteration: [RR@iteration]}
        eval_steps = 0
        total_eval_steps = len(
            self.dev_dataloader) if self.args.partial_eval_steps is None else self.args.partial_eval_steps

        result_dict = {}
        if self.args.eval_only:
            out_file= open(os.path.join(self.args.save_preds_root, f"{self.model_name}_preds.json"), "w")
        with torch.no_grad():
            if self.args.log_cossim:
                    stats_query_iterations = {'query': {}, 'pos' : {}, 'neg' : {}}
                    for i in range(self.args.max_refining_iterations):
                        stats_query_iterations['query'][i+1] = []
                        stats_query_iterations['pos'][i+1] = []
                        stats_query_iterations['neg'][i+1] = []
            for data in tqdm(self.dev_dataloader, total=total_eval_steps, ncols=50,
                             desc=f"Evaluating"):

                query_repr, all_passage_repr, all_passage_ids, relevant_passage_ids, query_id = data[0]
                if query_id not in result_dict:
                    result_dict[query_id] = {"relevant_ids": relevant_passage_ids.tolist()}

                initial_mean_rank = self.calc_reciprocal_rank(all_passage_ids, relevant_passage_ids)
                MRR[0].append(initial_mean_rank)
                query_repr = query_repr.unsqueeze(0).to(self.device)
                for iteration in range(self.args.max_refining_iterations):
                    negative_candidates_ids = np.copy(
                        all_passage_ids[self.args.neg_sample_rank_from:self.args.neg_sample_rank_to + 1])
                    negative_samples_repr = np.copy(
                        all_passage_repr[self.args.neg_sample_rank_from:self.args.neg_sample_rank_to + 1])
                    positive_passage_repr = np.zeros([relevant_passage_ids.shape[0], all_passage_repr.shape[1]])

                    for i, rel_id in enumerate(relevant_passage_ids):
                        delete_idx = np.where(negative_candidates_ids == rel_id)
                        passage_idx = np.where(all_passage_ids == rel_id)
                        positive_passage_repr[i] = all_passage_repr[passage_idx]
                        negative_candidates_ids = np.delete(negative_candidates_ids, delete_idx)
                        negative_samples_repr = np.delete(negative_samples_repr, delete_idx, axis=0)

                    negative_samples_repr = negative_samples_repr[
                        np.random.choice(np.arange(negative_candidates_ids.size), self.args.num_neg_samples,
                                         replace=False)]
                    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
                    negative_samples_repr_tensor = torch.from_numpy(negative_samples_repr).unsqueeze(0).to(self.device)
                    query_repr_original = query_repr.clone().detach().to(self.device)
                    query_repr = self.model(query_repr, negative_samples_repr_tensor)
                    if self.args.log_cossim:
                        query_similarity = cosine_similarity(query_repr, query_repr_original)
                        
                        positive_similarity = cosine_similarity(query_repr, torch.tensor(positive_passage_repr).to(self.device)).max() # pick the max repr
                        # if positive_passage_repr.shape[0] > 1 :
                            # print(positive_similarity)
                        negative_similarity = cosine_similarity(query_repr,  torch.tensor(negative_samples_repr).to(self.device)).mean()

                        stats_query_iterations['query'][iteration+1].append(float(query_similarity))
                        stats_query_iterations['pos'][iteration+1].append(float(positive_similarity))
                        stats_query_iterations['neg'][iteration+1].append(float(negative_similarity))


                    np_query = torch.clone(query_repr).detach().cpu().numpy()

                    sim_scores = util.cos_sim(np_query, all_passage_repr).numpy()

                    new_ranks = np.argsort(-1*sim_scores, axis=1)  # (1,1000)

                    all_passage_repr = all_passage_repr[new_ranks[0]]

                    all_passage_ids = all_passage_ids[new_ranks[0]]

                    reciprocal_rank = self.calc_reciprocal_rank(all_passage_ids, relevant_passage_ids)

                    MRR[iteration + 1].append(reciprocal_rank)

                    result_dict[query_id][f"iter_{iteration+1}"] = all_passage_ids.tolist()


                eval_steps += 1

                if self.args.eval_only:
                    out_string = json.dumps(result_dict[query_id])
                    out_file.write(out_string+'\n')

                if self.args.partial_eval_steps is not None:
                    if eval_steps > self.args.partial_eval_steps:
                        break

        for key in MRR.keys():
            mean_rr = sum(MRR[key]) / len(MRR[key])
            MRR[key] = mean_rr
            if self.args.use_wandb:
                wandb.log({f"MRR@{key}": float(mean_rr)})
        # store everything to wandb
        if self.args.log_cossim:
            print("Logging")
            if self.args.use_wandb:
                for key in stats_query_iterations['pos'].keys():
                    mean_pos = sum(stats_query_iterations['pos'][key])/len(stats_query_iterations['pos'][key])
                    mean_neg = sum(stats_query_iterations['neg'][key])/len(stats_query_iterations['neg'][key])
                    mean_query = sum(stats_query_iterations['query'][key])/len(stats_query_iterations['query'][key])
                    wandb.log({ f'query_cosine@{key}' : float(mean_query), f'pos_cosine@{key}' : float(mean_pos), f'neg_cosine@{key}' : float(mean_neg) })


        return max(list(MRR.values())[1:])

    def calc_reciprocal_rank(self, ranked_list, relevant_ids):
        for rank, passage_id in enumerate(ranked_list):
            if passage_id in relevant_ids:
                return (1 / (rank + 1))

        return 0


