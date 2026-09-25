"""Compatibility classes for the released component-pool pickle.

The pool was serialized by the original DeRy utility module under the name
``utils``. Only the two lightweight data containers required for safe loading
are retained here; feature extraction and similarity computation are not part
of this release.
"""

from blocklize import MODEL_BLOCKS, MODEL_STATS, MODEL_ZOO
from blocklize.block_meta import MODEL_INOUT_SHAPE


class Block:
    def __init__(self, model_name, block_index, node_list):
        self.model_name = str(model_name)
        self.block_index = int(block_index)
        self.node_list = list(node_list)
        self.value = 0
        self.size = 0
        self.group_id = None

    def print_split(self):
        start, end = int(self.node_list[0]), int(self.node_list[-1])
        return [
            MODEL_STATS[self.model_name]["arch"],
            MODEL_BLOCKS[self.model_name][start],
            MODEL_BLOCKS[self.model_name][end],
            MODEL_STATS[self.model_name]["backend"],
        ]

    def get_inout_size(self):
        start = MODEL_BLOCKS[self.model_name][int(self.node_list[0])]
        end = MODEL_BLOCKS[self.model_name][int(self.node_list[-1])]
        self.in_size = MODEL_INOUT_SHAPE[self.model_name]["in_size"][start]
        self.out_size = MODEL_INOUT_SHAPE[self.model_name]["out_size"][end]

    def __eq__(self, other):
        return (
            isinstance(other, Block)
            and self.model_name == other.model_name
            and self.block_index == other.block_index
            and self.node_list == other.node_list
        )


class Block_Assign:
    def __init__(self, assignment_index=None, block_split_dict=None, centers=None):
        self.block2center = {}
        self.center2block = []
        self.centers = [] if centers is None else list(centers)
