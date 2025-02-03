import torch

class Collator:
    """
    A collator for tokenized inputs that:
      1) Applies dynamic masked language modeling (MLM) with probability `mlm_probability`.
      2) Supports shuffling of name and value tokens while keeping special tokens fixed.
      3) Returns a dictionary with keys:
             'input_ids_n', 'attention_mask_n', 'special_tokens_mask_n', 'type_mask_n', 'labels_n',
             'input_ids_v', 'attention_mask_v', 'special_tokens_mask_v', 'type_mask_v', 'labels_v'
      
    The masking actions are applied in a linked fashion between the name and value inputs.
    
    Args:
        joint_tokenizer: A joint tokenizer with attributes `tokenizer_n` and `tokenizer_v`.
        mlm (bool): Whether to apply masked language modeling.
        mlm_probability (float): Probability of masking each token.
        mlm_target (str): One of ["names", "values", "both-linked"] determining which tokens to mask.
        shuffle_tokens (bool): Whether to shuffle tokens within sequences.
    """
    def __init__(self, joint_tokenizer, mlm=True, mlm_probability=0.15, 
                 mlm_target="both-linked", shuffle_tokens=False):
        self.joint_tokenizer = joint_tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.mlm_target = mlm_target.lower()
        self.shuffle_tokens = shuffle_tokens

        self.name_tok = joint_tokenizer.tokenizer_n
        self.value_tok = joint_tokenizer.tokenizer_v

    def __call__(self, batch_input):
        """
        Processes a batch of tokenized inputs, applies masking (and optionally token shuffling),
        and returns the same format as the JointTokenizer output but with added 'labels_n'
        and 'labels_v' fields.
        
        Args:
            batch_input: Either a single dictionary (for one sample) or a list of dictionaries,
              where each dictionary contains:
                  'input_ids_n', 'attention_mask_n', 'special_tokens_mask_n', 'type_mask_n',
                  'input_ids_v', 'attention_mask_v', 'special_tokens_mask_v', 'type_mask_v'
        
        Returns:
            A dictionary (if input was a single sample) or a dictionary of batched tensors,
            with the same keys as above plus:
                  'labels_n' and 'labels_v'
        """
        # If input is a single dictionary, wrap it in a list.
        is_single = False
        if isinstance(batch_input, dict):
            batch_input = [batch_input]
            is_single = True

        # Helper: determine the appropriate pad value based on the field key.
        def get_pad_value(key):
            if key == "input_ids_n":
                return self.name_tok.pad_id
            elif key == "input_ids_v":
                return self.value_tok.pad_id
            elif key in ["attention_mask_n", "attention_mask_v"]:
                return 0
            elif key in ["special_tokens_mask_n", "special_tokens_mask_v"]:
                return 1
            elif key in ["type_mask_n", "type_mask_v"]:
                return 1
            elif key in ["labels_n", "labels_v"]:
                return -100
            else:
                return 0
            
        # Helper: pad a list of sequences to the same length and convert to a tensor.
        def collate_field(key):
            field_list = [sample[key] for sample in batch_input]
            # If items are tensors, convert them to lists.
            if torch.is_tensor(field_list[0]):
                field_list = [x.tolist() for x in field_list]
            pad_val = get_pad_value(key)
            max_len = max(len(x) for x in field_list)
            padded = [x + [pad_val] * (max_len - len(x)) for x in field_list]
            dtype = torch.float if key in ["input_ids_n", "input_ids_v"] else torch.long
            return torch.tensor(padded, dtype=dtype) 

        # Collate all fields.
        collated = {key: collate_field(key) for key in batch_input[0].keys()}

        # Create label tensors for names and values (initialize with -100).
        labels_n = torch.full_like(collated["input_ids_n"], -100)
        labels_v = torch.full_like(collated["input_ids_v"], -100)
        batch_size, seq_len = collated["input_ids_n"].shape

        if self.mlm:
            # Build a probability matrix for masking.
            probability_matrix = torch.full((batch_size, seq_len), self.mlm_probability, device=collated["input_ids_n"].device)
            # Prevent special tokens from being masked using the names' special_tokens_mask.
            special_mask = collated["special_tokens_mask_n"].bool()
            probability_matrix.masked_fill_(special_mask, 0.0)

            # Ensure mlm_target is valid.
            if self.mlm_target not in ["names", "values", "both-linked"]:
                raise ValueError(f"Invalid mlm_target: {self.mlm_target}")

            # Sample masked indices (linked for both modalities).
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # Set labels: positions not masked become -100.
            if self.mlm_target in ["names", "both-linked"]:
                labels_n = collated["input_ids_n"].masked_fill(~masked_indices, -100)
            if self.mlm_target in ["values", "both-linked"]:
                labels_v = collated["input_ids_v"].masked_fill(~masked_indices, -100)

            # Decide on masking actions: 80% replace with [MASK], 10% random, 10% keep.
            mask = torch.bernoulli(torch.full((batch_size, seq_len), 0.8, device=collated["input_ids_n"].device)).bool() & masked_indices
            random_mask = torch.bernoulli(torch.full((batch_size, seq_len), 0.1, device=collated["input_ids_n"].device)).bool() & masked_indices & ~mask

            if self.mlm_target in ["names", "both-linked"]:
                collated["input_ids_n"][mask] = self.name_tok.mask_id
                random_tokens = torch.randint(3, len(self.name_tok), (batch_size, seq_len), device=collated["input_ids_n"].device)
                collated["input_ids_n"][random_mask] = random_tokens[random_mask].float()
            if self.mlm_target in ["values", "both-linked"]:
                collated["input_ids_v"][mask] = self.value_tok.mask_id
                # Cast random tokens to the same dtype as collated["input_ids_v"]
                random_tokens = torch.randint(3, len(self.value_tok), (batch_size, seq_len), device=collated["input_ids_v"].device).to(collated["input_ids_v"].dtype)
                collated["input_ids_v"][random_mask] = random_tokens[random_mask].float()

        # Add labels into the collated dictionary.
        collated["labels_n"] = labels_n
        collated["labels_v"] = labels_v

        # Optionally shuffle tokens.
        if self.shuffle_tokens:
            collated = self._shuffle_tokens(collated)

        # Return a single dictionary if the input was a single sample.
        if is_single:
            final_output = {k: v[0] for k, v in collated.items()}
            return final_output
        else:
            return collated

    def _shuffle_tokens(self, collated):
        """
        Shuffles non-special tokens within each sequence (for both name and value fields)
        while keeping special tokens fixed.
        """
        batch_size, seq_len = collated["input_ids_n"].shape

        # For each sample in the batch:
        for b in range(batch_size):
            non_special_indices = (~collated["special_tokens_mask_n"][b].bool()).nonzero(as_tuple=False).squeeze()
            if non_special_indices.numel() <= 1:
                continue
            shuffled_indices = non_special_indices[torch.randperm(non_special_indices.size(0))]
            for key in collated.keys():
                if key.endswith("_n") or key.endswith("_v"):
                    collated[key][b, non_special_indices] = collated[key][b, shuffled_indices]
        return collated
