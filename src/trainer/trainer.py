from src.model.RefinerModel import RefinerModel
from src.data.NegativeFeedbackDataset import NegativeFeedbackDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from src.loss.rocchio import RochhiosLoss
import torch
import transformers
import os


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

        self.model = RefinerModel(args)

        self.trainable_params = self.model.parameters()
        if self.args.checkpoint is not None:
            try:
                self.model.load_state_dict(torch.load(self.args.checkpoint))
            except:
                raise ValueError("Checkpoint config does not match model config")

        if not self.args.eval_only:
            train_dataset = NegativeFeedbackDataset(self.args, "train")
            self.train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

            self.optimizer = optim.AdamW(self.trainable_params, lr=self.args.learning_rate)
            self.lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                             num_warmup_steps=self.args.warmup_steps,
                                                                             num_training_steps=len(
                                                                                 self.train_dataloader) * self.args.epochs)

        dev_dataset = NegativeFeedbackDataset(self.args, "dev")
        self.dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)

    def train(self):
        """
        Training function

        """
        best_MRR = -float("inf")
        for epoch in range(self.args.epochs):
            for data in tqdm(self.train_dataloader, total=len(self.train_dataloader), ncols=60,
                             desc=f"Training epoch {epoch + 1}/{self.args.epochs}"):

                query_repr, relevant_repr, negative_repr = data
                # query_repr = query embedding, torch.Tensor; shape (train_batch_size, 1, 768)
                # relevant_repr = relevant document embedding, torch.Tensor; (train_batch_size, num relevant doc, 768)
                # negative_repr = negative document embedding, torch.Tensor; (train_batch_size, num negative samples, 768)

                refined_query_repr = self.model(query_repr, negative_repr)
                # Expected model output: Refined query representation torch.Tensor; shape (train_batch_size, 1, 768)

                loss = RochhiosLoss(args, refined_query_repr, query_repr, relevant_repr, negative_repr)

                loss.backward()

                if self.args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.args.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()

            mrr = self.eval()

            if mrr > best_MRR:
                best_MRR = mrr

                save_path = os.path.join(self.args.save_model_root, f"{self.args.model_identifier}_epoch{epoch + 1}.pt")
                torch.save(self.model, save_path)

    def eval(self):
        """
        Returns MRR

        """
        pass
