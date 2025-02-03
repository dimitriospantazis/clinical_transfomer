import torch
import numpy as np
from typing import Any, Dict, List, Union, Optional
from typeguard import check_type


class Tokenizer:
    """
    A tokenizer that processes sequences of categorical (str) and numerical (int/float) tokens.
    
    Features:
    - Tokenizes strings using a fixed vocabulary.
    - Passes numerical inputs unchanged.
    - Supports special tokens: [PAD]=0, [CLS]=1, [MASK]=2.
    - Prepends [CLS] to each sequence.
    - Supports padding and truncation.
    - Returns attention_mask, special_tokens_mask, and type_mask.
    - Can return either PyTorch tensors or plain Python lists.
    
    Example output (when return_tensors=False) for a batch:
    [
      {
        'input_ids': [101, 7592, 1010],
        'attention_mask': [1, 1, 1],
        'special_tokens_mask': [1, 0, 0],
        'type_mask': [1, 1, 0]
      },
      { ... },
      { ... }
    ]
    
    If a single sequence is provided, a dictionary is returned.
    """

    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    MASK_TOKEN = "[MASK]"

    def __init__(self, vocabulary: List[str]):
        """Initializes the tokenizer with a vocabulary."""
        self.special_tokens = {self.PAD_TOKEN: 0, self.CLS_TOKEN: 1, self.MASK_TOKEN: 2}
        self.vocab_to_id = {word: idx for idx, word in enumerate(sorted(vocabulary), start=3)}
        self.id_to_vocab = {idx: word for word, idx in self.vocab_to_id.items()}
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

    def __call__(
        self,
        inputs: Union[List[Any], List[List[Any]]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: bool = False
    ) -> Union[Dict[str, Union[List[int], torch.Tensor]], List[Dict[str, Union[List[int], torch.Tensor]]]]:
        """
        Tokenizes input sequences with optional padding and truncation.
        
        Args:
            inputs: A single sequence (list of tokens) or a batch (list of sequences).
            padding: If True or "max_length", pad sequences to the longest or to max_length.
            truncation: Whether to truncate sequences longer than max_length.
            max_length: Maximum length to truncate/pad to if truncation or "max_length" padding is used.
            return_tensors: If True, converts the output lists to PyTorch tensors.
        
        Returns:
            If a batch of sequences is provided, returns a list of dictionaries (one per sequence),
            each with:
                - "input_ids": Token IDs.
                - "attention_mask": 1 for real tokens, 0 for padding.
                - "special_tokens_mask": 1 for special tokens, 0 otherwise.
                - "type_mask": 1 for tokens that are processed (categorical) and 0 for numerical tokens.
            
            If a single sequence is provided, returns a dictionary with the same keys.
        """
        if truncation and max_length is None:
            raise ValueError("`max_length` must be specified when `truncation=True`.")

        # Determine if the input is a single sequence.
        is_single = False
        if not isinstance(inputs[0], list):
            # If the first element is not a list, assume a single sequence.
            batch = [inputs]
            is_single = True
        else:
            batch = inputs

        tokenized_batch = []

        # Process each sequence.
        for seq in batch:
            # Prepend the [CLS] token.
            input_ids = [self.cls_id] + [self._tokenize_one(token) for token in seq]
            # type_mask: 1 for [CLS] and for tokens that are strings (categorical), 0 for numerical.
            type_mask = [1] + [1 if isinstance(token, str) else 0 for token in seq]
            special_tokens_mask = [1] + [1 if self._is_special_token(token) else 0 for token in seq]

            # Apply truncation if enabled.
            if truncation and max_length is not None and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                type_mask = type_mask[:max_length]
                special_tokens_mask = special_tokens_mask[:max_length]

            tokenized_batch.append((input_ids, type_mask, special_tokens_mask))

        # Determine maximum sequence length for padding.
        if padding:
            if isinstance(padding, str) and padding == "max_length" and max_length is not None:
                max_len = max_length
            else:
                max_len = max(len(ids) for ids, _, _ in tokenized_batch)
        else:
            max_len = None

        output = []
        for input_ids, type_mask, special_tokens_mask in tokenized_batch:
            if max_len is not None:
                pad_len = max_len - len(input_ids)
                padded_input_ids = input_ids + [self.pad_id] * pad_len
                padded_attention_mask = [1] * len(input_ids) + [0] * pad_len
                padded_type_mask = type_mask + [1] * pad_len
                padded_special_tokens_mask = special_tokens_mask + [1] * pad_len
            else:
                padded_input_ids = input_ids
                padded_attention_mask = [1] * len(input_ids)
                padded_type_mask = type_mask
                padded_special_tokens_mask = special_tokens_mask

            out_dict = {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "special_tokens_mask": padded_special_tokens_mask,
                "type_mask": padded_type_mask,
            }
            output.append(out_dict)

        if return_tensors:
            # Convert each list in each dictionary to a torch.Tensor.
            for out_dict in output:
                for key in out_dict:
                    dtype = torch.float if "input_ids" in key else torch.long
                    out_dict[key] = torch.tensor(out_dict[key], dtype=dtype) #Use float to preserve numerical values

        # If only a single sequence was provided, return the dictionary directly.
        return output[0] if is_single else output

    def decode(self, tokenized_outputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[List[Any], List[List[Any]]]:
        """
        Decodes tokenized outputs, handling both single dictionary and list of dictionaries.

        Args:
            tokenized_outputs: Either a single dictionary or a list of dictionaries, as produced by __call__.
        
        Returns:
            If a single dictionary is provided, returns a single decoded sequence (list of tokens).
            Otherwise, returns a list of decoded sequences.
        """
        single_input = False
        if isinstance(tokenized_outputs, dict):
            tokenized_outputs = [tokenized_outputs]
            single_input = True

        decoded_batch = []
        for out_dict in tokenized_outputs:
            input_ids = out_dict["input_ids"]
            type_mask = out_dict["type_mask"]

            # Convert tensors to lists if necessary.
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.detach().cpu().tolist()
            if isinstance(type_mask, torch.Tensor):
                type_mask = type_mask.detach().cpu().tolist()

            decoded_seq = [
                tok if mask == 0 else self.id_to_special.get(tok, self.id_to_vocab.get(tok, tok))
                for tok, mask in zip(input_ids, type_mask)
            ]
            decoded_batch.append(decoded_seq)

        return decoded_batch[0] if single_input else decoded_batch

    def _tokenize_one(self, token: Any) -> Union[int, float]:
        """Maps a token to its ID or returns a number unchanged."""
        if isinstance(token, str):
            return self.vocab_to_id.get(token, KeyError(f"Token '{token}' not in vocabulary."))
        return float(token) if isinstance(token, (int, float)) else TypeError(f"Unsupported type: {type(token)}")

    def _is_special_token(self, token: Any) -> bool:
        """Checks if a token is a special token."""
        return isinstance(token, str) and token in self.special_tokens

    def __len__(self) -> int:
        """Returns the number of tokens in the tokenizer's vocabulary, including special tokens."""
        return len(self.vocab_to_id) + len(self.special_tokens)

    @property
    def pad_id(self) -> int:
        return self.special_tokens[self.PAD_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.special_tokens[self.CLS_TOKEN]

    @property
    def mask_id(self) -> int:
        return self.special_tokens[self.MASK_TOKEN]



class JointTokenizer:
    """
    A tokenizer that separately tokenizes feature names and feature values.

    Input:
      - A batch of sequences, where each sequence is a list of [name, value] pairs.

    Behavior:
      - Separately tokenizes names and values using two tokenizers.
      - Returns a list of dictionaries (one per sample) with keys:
            'input_ids_n', 'attention_mask_n', 'special_tokens_mask_n', 'type_mask_n',
            'input_ids_v', 'attention_mask_v', 'special_tokens_mask_v', 'type_mask_v'
      - If a single sequence is provided, returns a dictionary (not a list).
      - Provides a decode() method that reconstructs the original sequences as lists of [name, value] pairs.
        If a single dictionary is provided, returns a single sequence.
      - Exposes properties for pad_id, cls_id, and mask_id (assumed the same for both tokenizers).
    """
    def __init__(self, tokenizer_n, tokenizer_v):
        """
        Args:
            tokenizer_n: Tokenizer for feature names.
            tokenizer_v: Tokenizer for feature values.
        """
        self.tokenizer_n = tokenizer_n
        self.tokenizer_v = tokenizer_v

    def __call__(
        self,
        inputs: Union[List[List[Any]], List[List[List[Any]]]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Tokenizes a batch of sequences, where each sequence is a list of [name, value] pairs.

        Args:
            inputs: Either a single sequence (list of [name, value] pairs) or a batch of sequences.
            padding: Whether (or how) to pad sequences.
            truncation: Whether to truncate sequences longer than max_length.
            max_length: Maximum length for truncation/padding.
            return_tensors: If True, the underlying tokenizers will return tensors.
            
        Returns:
            If a batch is provided, returns a list of dictionaries (one per sample) with keys:
                'input_ids_n', 'attention_mask_n', 'special_tokens_mask_n', 'type_mask_n',
                'input_ids_v', 'attention_mask_v', 'special_tokens_mask_v', 'type_mask_v'
            If a single sequence is provided, returns a dictionary.
        """
        # Check if a single sequence is provided.
        is_single = False
        if not isinstance(inputs[0][0], list):
            is_single = True
            inputs = [inputs]

        # Separate names and values.
        names = [[pair[0] for pair in seq] for seq in inputs]
        values = [[pair[1] for pair in seq] for seq in inputs]

        # Tokenize names and values separately.
        tokens_n = self.tokenizer_n(
            names,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
        tokens_v = self.tokenizer_v(
            values,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )

        # Assume each tokenizer returns a list of dictionaries (one per sample).
        batch_size = len(tokens_n)
        output = []
        for i in range(batch_size):
            sample_dict = {
                'input_ids_n': tokens_n[i]['input_ids'],
                'attention_mask_n': tokens_n[i]['attention_mask'],
                'special_tokens_mask_n': tokens_n[i]['special_tokens_mask'],
                'type_mask_n': tokens_n[i]['type_mask'],
                'input_ids_v': tokens_v[i]['input_ids'],
                'attention_mask_v': tokens_v[i]['attention_mask'],
                'special_tokens_mask_v': tokens_v[i]['special_tokens_mask'],
                'type_mask_v': tokens_v[i]['type_mask']
            }
            output.append(sample_dict)

        return output[0] if is_single else output

    def decode(
        self,
        tokenized_outputs: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[List[List[Any]], List[List[List[Any]]]]:
        """
        Decodes tokenized outputs back into original [name, value] pairs.

        Args:
            tokenized_outputs: Either a single dictionary (for one sample) or a list of dictionaries
                               as produced by __call__.
            
        Returns:
            If a single dictionary is provided, returns a single sequence (list of [name, value] pairs).
            Otherwise, returns a list of sequences (one per sample).
        """
        is_single = False
        if isinstance(tokenized_outputs, dict):
            is_single = True
            tokenized_outputs = [tokenized_outputs]

        decoded_batch = []
        for sample in tokenized_outputs:
            # Prepare dictionaries for decoding each modality.
            tokens_n = {
                "input_ids": sample["input_ids_n"],
                "type_mask": sample["type_mask_n"]
            }
            tokens_v = {
                "input_ids": sample["input_ids_v"],
                "type_mask": sample["type_mask_v"]
            }
            decoded_names = self.tokenizer_n.decode(tokens_n)
            decoded_values = self.tokenizer_v.decode(tokens_v)
            # Combine the decoded names and values into [name, value] pairs.
            sample_decoded = []
            for name, value in zip(decoded_names, decoded_values):
                sample_decoded.append([name, value])
            decoded_batch.append(sample_decoded)

        return decoded_batch[0] if is_single else decoded_batch

    @property
    def pad_id(self) -> int:
        # Assuming both tokenizers use the same pad_id.
        return self.tokenizer_n.pad_id

    @property
    def cls_id(self) -> int:
        return self.tokenizer_n.cls_id

    @property
    def mask_id(self) -> int:
        return self.tokenizer_n.mask_id

    def _is_single_sequence(self, x: Any) -> bool:
        """Helper to determine if x is a single sequence (i.e. not batched)."""
        return not isinstance(x[0][0], list)
