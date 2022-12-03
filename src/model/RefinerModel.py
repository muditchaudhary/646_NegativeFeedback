import torch

class RefinerModel(torch.nn.Module):

    def __init__(self, args):
        super(RefinerModel, self).__init__()

        pass

    def forward(self, query_repr, negative_repr):
        """
        query_repr = query embedding, torch.Tensor; shape (train_batch_size, 1, 768)
        negative_repr = negative document embedding, torch.Tensor; (train_batch_size, num negative samples, 768)

        """

        pass


