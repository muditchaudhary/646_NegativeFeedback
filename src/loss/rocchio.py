import torch


def RochhiosLoss(args, refined_query_repr, query_repr, relevant_repr, negative_repr):
    # Convert Query repr to tensor
    query_repr = torch.tensor(query_repr)
    cosineLoss = torch.nn.CosineEmbeddingLoss(margin=0)
    regularized_loss = args.alpha1 * cosineLoss(refined_query_repr, query_repr,
                                                torch.tensor([1.0]))  # want to keep them similar
    batch_pos = relevant_repr.shape[1]  # get the number of queries for each batch
    batch_neg = negative_repr.shape[1]
    embedding_size = negative_repr.shape[-1]
    unsq_refine_query = torch.unsqueeze(refined_query_repr, dim=1)
    similar_repr = args.alpha2 * cosineLoss(unsq_refine_query.repeat(1, batch_pos, 1).reshape([-1,embedding_size]), relevant_repr.reshape([-1,embedding_size]), torch.tensor([1.0]))
    dissimilar_repr = args.alpha3 * cosineLoss(unsq_refine_query.repeat(1, batch_neg, 1).reshape([-1,embedding_size]), negative_repr.reshape([-1,embedding_size]), torch.tensor([-1.0]))
    return regularized_loss + similar_repr + dissimilar_repr
