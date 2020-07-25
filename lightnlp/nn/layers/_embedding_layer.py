import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    This layer is callable for word embedding and pre-trained embedding, etc
    """

    def __init__(self, vocab_size, embedding_dim, pre_trained_embeddings=None, finetune=True):
        """
        :param vocab_size: vocabulary size
        :param embedding_dim: embedding dim
        :param pre_trained_embeddings: the embedding weight, 2D tensor,
                if it has value, it will use it to build embedding, default None
        """
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.finetune = finetune

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        if pre_trained_embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(pre_trained_embeddings)
        else:
            nn.init.xavier_normal_(self.embeddings.weight)
        self.embeddings.weight.requires_grad = self.finetune

    def forward(self, sentence):
        sent_inputs = self.embeddings(sentence)  # shape: (batch, seq_len, word_vec_size)
        return sent_inputs
