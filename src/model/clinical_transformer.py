import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.model.embedder import Embedder

# Configuration class

class ClinicalTransformerConfig(PretrainedConfig):
    model_type = "clinical_transformer"

    def __init__(self, vocab_size_n, vocab_size_v, embed_dim_n, embed_dim_v, hidden_dim_num, hidden_size, num_layers, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        # Embedding parameters
        self.vocab_size_n = vocab_size_n
        self.vocab_size_v = vocab_size_v
        self.embed_dim_n = embed_dim_n
        self.embed_dim_v = embed_dim_v
        self.hidden_dim_num = hidden_dim_num

        # Transformer parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout


# Main model class

class ClinicalTransformer(PreTrainedModel):
    """
    Transformer model for clinical data with:
    - Custom embedder (no position embeddings)
    - Support for MLM
    - Two output heads
    """
    config_class = ClinicalTransformerConfig

    def __init__(self, config):
        super().__init__(config)

        # Embedding layers
        self.embedder_n = Embedder(config.vocab_size_n, config.embed_dim_n, use_numerical=False)
        self.embedder_v = Embedder(config.vocab_size_v, config.embed_dim_v, hidden_dim_num = config.hidden_dim_num, use_numerical=True)
        # Combined projection to hidden size.
        self.embedder = nn.Linear(config.embed_dim_n + config.embed_dim_v, config.hidden_size)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # MLM Heads
        self.mlm_head_n = nn.Linear(config.hidden_size, config.vocab_size_n) # categorical
        self.mlm_head_v = nn.Linear(config.hidden_size, config.vocab_size_v) # categorical
        self.mlm_head_v_num = nn.Linear(config.hidden_size, 1) # numerical

        # Second custom output head (e.g., classification/regression or survival task)
        self.prediction_head = nn.Linear(config.hidden_size, 1)  # Example: Predicting a scalar

        self.init_weights()

    def forward(self, inputs, labels=None):
        """
        Forward pass:
            inputs is a dictionary with the following keys:
              'input_ids_n': (batch, seq_len) token IDs for names.
              'attention_mask_n': (batch, seq_len) attention mask for names.
              'special_tokens_mask_n': (batch, seq_len) special tokens mask for names.
              'type_mask_n': (batch, seq_len) indicating categorical tokens (1) in names.
              
              'input_ids_v': (batch, seq_len) token IDs for values.
              'attention_mask_v': (batch, seq_len) attention mask for values.
              'special_tokens_mask_v': (batch, seq_len) special tokens mask for values.
              'type_mask_v': (batch, seq_len) indicating categorical tokens (1) in values.
              
              Optionally, 'labels_n' and 'labels_v' for MLM loss.
        """

        # Embed names and values separately.
        # For names:
        x_name = self.embedder_n(inputs['input_ids_n'], inputs['type_mask_n'])
        # For values:
        x_value = self.embedder_v(inputs['input_ids_v'], inputs['type_mask_v'])
        # Concatenate along the embedding dimension.
        x = self.embedder(torch.cat([x_name, x_value], dim=-1))

        # Run through the transformer encoder.
        # Use the attention mask for names (assumed to be aligned with the concatenated tokens).
        x = self.encoder(x, src_key_padding_mask=(inputs['attention_mask_n'] == 0))

        # Prediction head for downstream task (e.g. regression/survival).
        logits = self.prediction_head(x).squeeze(-1)

        # MLM Heads
        mlm_logits_n = self.mlm_head_n(x)
        mlm_logits_v = self.mlm_head_v(x)
        mlm_logits_v_num = self.mlm_head_v_num(x)

        # For numerical value head, keep only numerical values:
        # Use type_mask_v to zero out positions that are categorical.
        type_mask_expanded = inputs['type_mask_v'].unsqueeze(-1)  # shape: (batch, seq_len, 1)
        mlm_logits_v_num = mlm_logits_v_num.masked_fill(type_mask_expanded.bool(), 0)

         # Prepare outputs
        outputs = {}
        outputs['logits'] = [logits, mlm_logits_n, mlm_logits_v, mlm_logits_v_num]
        outputs['mlm_loss'] = None

        # Compute MLM Loss if labels are provided.
        if (inputs.get('labels_n') is not None) and (inputs.get('labels_v') is not None):
            
            # Categorical loss for names.
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss_n = loss_fct(mlm_logits_n.view(-1, self.config.vocab_size_n), inputs['labels_n'].view(-1).long())

            # For values, set labels to -100 where type_mask_v is 0 (exclude numerical tokens).
            labels_cat = inputs['labels_v'].masked_fill(inputs['type_mask_v'] == 0, -100)
            mlm_loss_v = loss_fct(mlm_logits_v.view(-1, self.config.vocab_size_v), labels_cat.view(-1).long())

            # Numerical loss for values.
            loss_fct = nn.MSELoss()
            labels_num = inputs['labels_v']
            mask = (inputs['type_mask_v'] == 0) & (labels_num != -100)
            mlm_loss_v_num = loss_fct(mlm_logits_v_num.squeeze(-1)[mask], labels_num[mask])

            outputs['mlm_loss'] = [mlm_loss_n, mlm_loss_v, mlm_loss_v_num]

        return outputs
