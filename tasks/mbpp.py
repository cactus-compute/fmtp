"""
The MBPP (Mostly Basic Python Programming) dataset.
https://huggingface.co/datasets/google-research-datasets/mbpp

Example problem instance:
- text: "Write a function to find the similar elements from the given two tuple lists."
- code: "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)"
- test_list: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", ...]
"""

from datasets import load_dataset
from tasks.common import Task


class MBPP(Task):

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], "MBPP split must be train|validation|test"
        self.split = split
        self.ds = load_dataset("google-research-datasets/mbpp", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        prompt = row['text']  # the problem description
        code = row['code']  # the reference solution
        test_list = row['test_list']  # list of assertion strings
        # Create conversation: user asks the question, assistant provides the code
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
        ]
        conversation = {
            "messages": messages,
            "test_list": test_list,  # needed during evaluation
        }
        return conversation
