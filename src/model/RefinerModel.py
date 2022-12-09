import torch

class RefinerModel(torch.nn.Module):

    def __init__(self, args):
        super(RefinerModel, self).__init__()
        self.delta_learn = args.delta_learning
        self.fc1 = torch.nn.Linear(768 * 2, 768)
        self.fc2 = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()


    def forward(self, query_repr, negative_repr = None):
        """
        query_repr = query embedding, torch.Tensor; shape (train_batch_size, 1, 768)
        negative_repr = negative document embedding, torch.Tensor; (train_batch_size, num negative samples, 768)

        """
        # @dhawal, we won't be using the negative represetation for query refinement forward pass
        modified_repr = torch.cat([query_repr, negative_repr.mean(axis = 1)], dim = 1) # Shape : batch_size x (embedding_size * 2)
        assert modified_repr.shape[1] == query_repr.shape[-1] * 2, "Wrong Shape"
        x = self.fc1(modified_repr)
        x = self.relu(x)
        x = self.fc2(x)

        if self.delta_learn:
            refined_query = query_repr + x
            return refined_query
        else:
            return x

