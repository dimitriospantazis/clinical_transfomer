import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from typing import List, Union

class Embedder(nn.Module):
    """
    A custom embedder that efficiently handles both categorical (tokenized) and numerical features.

    Args:
        vocab_size (int): Size of the vocabulary for token embeddings.
        embed_dim (int, optional): Dimension of the token embeddings. Default is 32.
        input_dim_num (int, optional): Number of numerical input features. Default is 1.
        hidden_dim_num (int, optional): Hidden layer size for the numerical MLP. Default is 16.
        pad_id (int, optional): Padding token ID. Default is 0.
        use_numerical (bool, optional): Whether to process numerical inputs. Default is True.

    Example:
        >>> embedder = Embedder(vocab_size=100, embed_dim=32, use_numerical=True)
        >>> inputs = {'input_ids': torch.tensor([[1, 2, 3.5], [4, 5, 6.7]]), 
                      'type_mask': torch.tensor([[1, 1, 0], [1, 1, 0]])}
        >>> output = embedder(inputs)
        >>> print(output.shape)  # (batch_size, seq_len, embed_dim)
    """

    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 32,
                 input_dim_num: int = 1,
                 hidden_dim_num: int = 16,
                 pad_id: int = 0,
                 use_numerical: bool = True):
        super(Embedder, self).__init__()
        
        self.use_numerical = use_numerical
        
        # Embedding layer for categorical (tokenized) inputs
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        
        # MLP for numerical inputs (only initialized if numerical support is enabled)
        if self.use_numerical:
            self.numeric_mlp = nn.Sequential(
                nn.Linear(input_dim_num, hidden_dim_num),
                nn.ReLU(),
                nn.Linear(hidden_dim_num, embed_dim)
            )
        
        self.embed_dim = embed_dim
        self.pad_id = pad_id

    def forward(self, input_ids: torch.Tensor, type_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedder.
        
        Args:
            input_ids (Tensor): FloatTensor of shape (batch_size, seq_length) with token IDs (as floats) or numerical values.
            type_mask (Tensor): Tensor of shape (batch_size, seq_length) with 1s for token IDs and 0s for numerical values.
        
        Returns:
            torch.Tensor: Shape (batch_size, seq_len, embed_dim) for Transformer input.
        """
        
        # Get the device from the input tensors
        device = input_ids.device

        batch_size, seq_length = input_ids.shape
        
        # Boolean masks for tokenized and numeric values
        token_mask = type_mask.bool()        # True where input_ids are token IDs
        numeric_mask = ~token_mask           # True where input_ids are numerical values
        
        # Initialize the combined embeddings tensor with zeros
        combined_embeddings = torch.zeros(batch_size, seq_length, self.embed_dim, device=device)
        
        ### Process Categorical (Tokenized) Inputs ###
        if token_mask.any():
            # Extract token IDs
            token_ids = input_ids[token_mask].long().to(device)  # Extract only the token values
            
            # Embed the token IDs
            embedded_tokens = self.embedding(token_ids)  # Shape: (N_tokens, embed_dim)
            
            # Assign embedded tokens to the combined embeddings
            combined_embeddings[token_mask] = embedded_tokens

        ### Process Numerical Inputs (if enabled) ###
        if self.use_numerical and numeric_mask.any():
            # Extract numerical values
            numeric_values = input_ids[numeric_mask].unsqueeze(-1).float().to(device)  # Shape: (N_numeric, 1)
            
            # Pass numerical values through the MLP
            embedded_numerics = self.numeric_mlp(numeric_values)  # Shape: (N_numeric, embed_dim)
            
            # Assign embedded numerics to the combined embeddings
            combined_embeddings[numeric_mask] = embedded_numerics
        
        return combined_embeddings



# Define a class to initialize embeddings using a pre-trained BERT model. 
# Because we will embed many gene names like 'BCL2A1', 'POU3F2', 'RP11-811P12.3' etc, 
# it uses the SciBERT model from Allen AI by default.




class BertEmbedder:
    """
    Provides embeddings using a pre-trained BERT model with no training possible.
    This class is NOT an nn.Module, so its parameters are completely isolated from
    any PyTorch optimizer. That ensures they cannot be accidentally un-frozen.

    Example usage:
        >>> bert = BertEmbedder(model_name="allenai/scibert_scivocab_uncased")
        >>> bert.to("cuda")
        >>> emb = bert(["BCL2A1", "POU3F2"])
        >>> print(emb.shape)  # (2, hidden_dim)
    """

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        """
        Initialize the BERT model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained BERT model from Hugging Face,
                              e.g., 'allenai/scibert_scivocab_uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # set to evaluation mode

        # Freeze params so they can't be accidentally trained
        for param in self.model.parameters():
            param.requires_grad = False

    def to(self, device: Union[str, torch.device]) -> "BertEmbedder":
        """
        Move the BERT model to the specified device (CPU/GPU).
        Returns self so that usage can be chained, e.g. bert.to("cuda").
        """
        device = torch.device(device)
        self.model.to(device)
        return self

    def embed(self, texts: Union[str, List[str]], batch_size: int = 16) -> torch.Tensor:
        """
        Generate embeddings for a batch of texts or a single text.

        Args:
            texts (Union[str, List[str]]): A string or list of strings to embed.
            batch_size (int): Number of texts to process in each batch.

        Returns:
            torch.Tensor: A tensor of shape (num_texts, hidden_dim) containing [CLS] embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        device = next(self.model.parameters()).device
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                # Move tokens to the same device as the BERT model
                for k in encoded:
                    encoded[k] = encoded[k].to(device)

                outputs = self.model(**encoded)
                # Extract [CLS] embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embedding)

        return torch.cat(all_embeddings, dim=0)

    def __call__(self, texts: Union[str, List[str]], batch_size: int = 16) -> torch.Tensor:
        """
        Allow calling the object directly like a function:
            emb = bert_embedder(["BCL2A1", "POU3F2"])
        """
        return self.embed(texts, batch_size)



class BertInitializer:
    def __init__(self, bert_embedder, tokenizer, batch_size=64, include_special_tokens=False):
        self.bert_embedder = bert_embedder
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.include_special_tokens = include_special_tokens

    def initialize(self, embedder):
        device = embedder.embedding.weight.device
        emb_dim = embedder.embedding.embedding_dim

        # build your token list
        vocab_tokens = self.tokenizer.vocab_list
        special_tokens = []
        if self.include_special_tokens:
            special_tokens = [self.tokenizer.PAD_TOKEN, self.tokenizer.CLS_TOKEN, self.tokenizer.MASK_TOKEN]
        all_tokens = list(special_tokens) + list(vocab_tokens)

        with torch.no_grad():
            for start_idx in range(0, len(all_tokens), self.batch_size):
                batch_tokens = all_tokens[start_idx : start_idx + self.batch_size]
                batch_embs = self.bert_embedder(batch_tokens)
                if batch_embs.shape[1] != emb_dim:
                    raise ValueError(f"Expected dimension {emb_dim}, got {batch_embs.shape[1]}")

                for i, token in enumerate(batch_tokens):
                    token_id = self.tokenizer(token)
                    if not isinstance(token_id, int):
                        continue
                    embedder.embedding.weight.data[token_id] = batch_embs[i].to(device)

        print("Done initializing embedder from BERT!")





