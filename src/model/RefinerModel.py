import torch

class RefinerModel(torch.nn.Module):

    def __init__(self, args):
        super(RefinerModel, self).__init__()
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()


    def forward(self, query_repr, negative_repr = None):
        """
        query_repr = query embedding, torch.Tensor; shape (train_batch_size, 1, 768)
        negative_repr = negative document embedding, torch.Tensor; (train_batch_size, num negative samples, 768)

        """
        # @dhawal, we won't be using the negative represetation for query refinement forward pass
        x = self.fc1(query_repr)
        x = self.relu(x)
        x = self.fc2(x)
        return x

