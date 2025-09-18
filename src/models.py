import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionHead(nn.Module):
    """Single Attention Head.

    This class implements a single attention head which is part of the
    multi-head attention mechanism. It computes the attention for a given
    query, key, and value.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_k (int): The dimension of the key vectors.
        d_q (int): The dimension of the query vectors.
        d_v (int): The dimension of the value vectors.
    Attributes:
        wq (nn.Linear): Linear layer to project input to query vectors.
        wk (nn.Linear): Linear layer to project input to key vectors.
        wv (nn.Linear): Linear layer to project input to value vectors.
    """

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(AttentionHead, self).__init__()

        self.wq = nn.Linear(in_features=d_model, out_features=d_q)
        self.wk = nn.Linear(in_features=d_model, out_features=d_k)
        self.wv = nn.Linear(in_features=d_model, out_features=d_v)

    def scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_q).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_v).

        Returns:
            Tensor: Output tensor after applying attention.
            Tensor: Attention weights.
        """

        # The dimension of the key tensor, used to scale the scores.
        dim_k = self.wk.out_features

        # Calculate the dot product between query and the transpose of key.
        # The result is then scaled by the square root of dim_k.
        scores = (q @ torch.transpose(k, dim0=1, dim1=2)) * (1/torch.sqrt(torch.tensor(dim_k)))

        # Apply the softmax function to obtain the attention weights.
        weights = torch.softmax(scores, dim=1).transpose(1,2)

        # Compute the output by performing a weighted sum of the value tensor
        # using the attention weights.
        output = torch.bmm(weights, v)
        return output, weights

    def forward(self, x):
        """Forward pass for the attention head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        # Obtain the corresponding query, key, and value vectors of the input tensor.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, _ = self.scaled_dot_product_attention(q,k,v)

        return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.

    This class implements the multi-head attention mechanism, which allows
    the model to focus on different parts of the input sequence at each layer.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.

    Attributes:
        heads (nn.ModuleList): A list of attention heads.
        output_linear (nn.Linear): Linear layer to project concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()
        d_kqv = (d_model/num_attention_heads)
        self.error = False
        if int(d_kqv) % d_kqv != 0:
            self.error = True
            self.d_model = d_model
            self.num_attention_heads = num_attention_heads
        d_kqv = int(d_kqv)
        self.heads = nn.ModuleList(AttentionHead(d_model=d_model,d_k=d_kqv,d_q=d_kqv,d_v=d_kqv) for i in range(num_attention_heads))
        self.output_linear = nn.Linear(in_features=num_attention_heads*d_kqv, out_features=d_model)

    def forward(self, hidden_state):
        """Forward pass for the multi-head attention layer.

        Args:
            hidden_state (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        if self.error:
            raise RuntimeError (f"d_model ({self.d_model}) must be divisible by num_attention_heads ({self.num_attention_heads})")

        x = self.output_linear(torch.concat([head(hidden_state) for head in self.heads], dim=-1))
        return x
    
class FeedForward(nn.Module):
    """FeedForward module for the Transformer.

    This class implements the feed-forward network used in the Transformer
    model. It consists of two linear layers with a GELU activation in between.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        intermediate_size (int): The dimension of the intermediate layer.

    Attributes:
        linear_1 (nn.Linear): The first linear layer that projects from d_model to intermediate_size.
        linear_2 (nn.Linear): The second linear layer that projects from intermediate_size back to d_model.
        gelu (nn.GELU): GELU activation function applied after the first linear layer.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=intermediate_size)
        self.linear_2 = nn.Linear(in_features=intermediate_size, out_features=d_model)
        self.gelu = nn.GELU()
        self.layers = nn.Sequential(
            self.linear_1,
            self.gelu,
            self.linear_2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer.

    This class implements a single layer of the Transformer encoder, consisting
    of a multi-head attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization applied before the multi-head attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization applied before the feed-forward network.
        attention (MultiHeadAttention): Multi-head attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply layer normalization and then apply multi-head attention
        hidden_state = self.attention(self.layer_norm_1(x))
        
        # Apply layer normalization and then apply feed-forward network
        x = self.feed_forward(self.layer_norm_2(hidden_state))
        
        return x

class Embeddings(nn.Module):
    """Embeddings module for the Transformer.

    This module combines token embeddings and positional embeddings and applies
    layer normalization.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.

    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for token embeddings.
        position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
        layer_norm (nn.LayerNorm): Layer normalization applied after combining embeddings.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size,d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings,d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to combine token and positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined and normalized embeddings of shape (batch_size, seq_len, d_model).
        """
        # Generate position IDs based on the input sequence length
        seq_length = input_ids.shape[1]
        position_ids = torch.tensor([i for i in range(seq_length)]).unsqueeze(0).expand_as(input_ids)

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = self.layer_norm(token_embeddings+position_embeddings)

        return embeddings
    
class TransformerEncoder(nn.Module):
    """Transformer Encoder.

    This class implements the encoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer encoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer encoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer encoder layers.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int
                 ):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(vocab_size,max_position_embeddings,d_model)
        self.layers = nn.ModuleList(TransformerEncoderLayer(d_model,num_attention_heads,intermediate_size) for i in range(num_hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class ClassificationHead(nn.Module):
    """Classification head for the Transformer model.

    This class implements a classification head that can be added on top of a
    Transformer encoder to perform tasks like text classification.

    Args:
        d_model (int): The size of the hidden states (output from the Transformer encoder).
        num_classes (int): The number of classes for classification.
        dropout_prob (float): Dropout probability to apply before the final linear layer.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied before the final linear layer.
        linear (nn.Linear): Linear layer that maps the hidden states to the number of classes.
    """

    def __init__(self, d_model: int, num_classes: int, dropout_prob: float):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classification head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.linear(self.dropout(x))
        return x
    
class TransformerForSequenceClassification(nn.Module):
    """Transformer model with a classification head on top for sequence classification.

    Args:
        vocab_size (int): Vocabulary size.
        max_position_embeddings (int): Maximum number of position embeddings.
        d_model (int): Hidden size of the Transformer model.
        num_attention_heads (int): Number of attention heads in each encoder layer.
        intermediate_size (int): Intermediate size of the feed-forward network.
        num_hidden_layers (int): Number of Transformer encoder layers.
        num_classes (int): Number of classes for classification.
        dropout_prob (float): Dropout probability.

    Attributes:
        transformer_encoder (TransformerEncoder): The Transformer encoder.
        classifier (ClassificationHead): The classification head on top of the Transformer encoder.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int,
                num_classes: int, dropout_prob: float):
        super(TransformerForSequenceClassification, self).__init__()
        self.transformer_encoder = TransformerEncoder(vocab_size,max_position_embeddings,d_model,
                                                      num_attention_heads,intermediate_size,num_hidden_layers)
        self.classifier = ClassificationHead(d_model,num_classes,dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer model with classification head.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Get the hidden states from the Transformer encoder
        x = self.transformer_encoder(input_ids)

        # Use the first token's output (e.g., CLS token) for classification
        x = x[:,0,:]
        
        # Pass through the classification head
        x = self.classifier(x)
        return x
