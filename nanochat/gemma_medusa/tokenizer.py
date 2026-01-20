"""
Tokenizer wrapper for Gemma 3 models.

Adapts HuggingFace Gemma tokenizer to match nanochat's tokenizer interface.
"""

import copy
from functools import lru_cache
from transformers import AutoTokenizer


class GemmaTokenizerWrapper:
    """Wraps HuggingFace Gemma tokenizer to match nanochat interface."""

    # Mapping from nanochat special tokens to Gemma equivalents
    SPECIAL_TOKEN_MAP = {
        "<|bos|>": "<bos>",
        "<|user_start|>": "<start_of_turn>user\n",
        "<|user_end|>": "<end_of_turn>\n",
        "<|assistant_start|>": "<start_of_turn>model\n",
        "<|assistant_end|>": "<end_of_turn>\n",
        # Gemma doesn't have native tool tokens, we'll use markers
        "<|python_start|>": "```python\n",
        "<|python_end|>": "\n```",
        "<|output_start|>": "\n**Output:**\n```\n",
        "<|output_end|>": "\n```\n",
    }

    def __init__(self, model_name="google/gemma-3-1b-it"):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._vocab_size = self.hf_tokenizer.vocab_size
        # Cache for special token IDs
        self._special_token_cache = {}

    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)

    def get_vocab_size(self):
        return self._vocab_size

    def get_special_tokens(self):
        """Return nanochat-style special tokens."""
        return set(self.SPECIAL_TOKEN_MAP.keys())

    def id_to_token(self, id):
        """Get token string from ID."""
        return self.hf_tokenizer.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """
        Encode a special token by exact match.
        Returns the first token ID of the mapped sequence.
        """
        if text in self.SPECIAL_TOKEN_MAP:
            gemma_text = self.SPECIAL_TOKEN_MAP[text]
            # For actual Gemma special tokens, use the tokenizer's special token handling
            if text == "<|bos|>":
                return self.hf_tokenizer.bos_token_id
            # For turn markers, encode and return the sequence
            ids = self.hf_tokenizer.encode(gemma_text, add_special_tokens=False)
            return ids[0] if ids else None
        return None

    def get_bos_token_id(self):
        """Get BOS token ID."""
        return self.hf_tokenizer.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """
        Encode text to token IDs.
        text can be a string or list of strings.
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = [self.hf_tokenizer.encode(t, add_special_tokens=False) for t in text]
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """Decode token IDs to text."""
        return self.hf_tokenizer.decode(ids, skip_special_tokens=False)

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation.
        Returns:
        - ids: list[int] of token ids
        - mask: list[int] of same length, mask=1 for tokens the Assistant trains on.
        """
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Handle system message by merging with first user message
        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        assert len(messages) >= 1, f"Conversation has less than 1 message"

        # Add BOS token
        add_tokens(self.hf_tokenizer.bos_token_id, 0)

        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages must be strings"
                # <start_of_turn>user\n{content}<end_of_turn>\n
                turn_start = self.hf_tokenizer.encode("<start_of_turn>user\n", add_special_tokens=False)
                turn_end = self.hf_tokenizer.encode("<end_of_turn>\n", add_special_tokens=False)
                content_ids = self.hf_tokenizer.encode(content, add_special_tokens=False)
                add_tokens(turn_start, 0)
                add_tokens(content_ids, 0)
                add_tokens(turn_end, 0)

            elif message["role"] == "assistant":
                # <start_of_turn>model\n
                turn_start = self.hf_tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
                turn_end = self.hf_tokenizer.encode("<end_of_turn>\n", add_special_tokens=False)
                add_tokens(turn_start, 0)

                if isinstance(content, str):
                    content_ids = self.hf_tokenizer.encode(content, add_special_tokens=False)
                    add_tokens(content_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.hf_tokenizer.encode(part["text"], add_special_tokens=False)
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            python_start = self.hf_tokenizer.encode("```python\n", add_special_tokens=False)
                            python_end = self.hf_tokenizer.encode("\n```", add_special_tokens=False)
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            output_start = self.hf_tokenizer.encode("\n**Output:**\n```\n", add_special_tokens=False)
                            output_end = self.hf_tokenizer.encode("\n```\n", add_special_tokens=False)
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")

                add_tokens(turn_end, 1)

        # Truncate to max_tokens
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Visualize tokenization with colored output."""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Render conversation priming the Assistant for completion.
        Used during RL and evaluation.
        """
        # Pop the last assistant message
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from Assistant"
        messages.pop()

        # Tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Add the assistant turn start to prime for completion
        assistant_start = self.hf_tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        ids.extend(assistant_start)
        return ids
