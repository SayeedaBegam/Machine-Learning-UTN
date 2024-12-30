from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import torch


class SimpleDataset(Dataset):
    def __init__(self, data_path: str):

        ####################################################################
        # YOUR CODE
        # TODO: Load the data from the csv file give by the path.
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the idx-th sample of the dataset.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sample and its label. Shapes
                are (1, 2) and (1, 1) respectively.
        """
        if idx >= len(self) or idx < 0:
            # needed for iterator stop condition
            raise IndexError

        ####################################################################
        # YOUR CODE
        # TODO: Return the idx-th sample of the dataset. Respect the shapes form
        # the docstring.
        # Hint: map the labels to interval [0, 1] as expected by the loss
        ####################################################################
        pass

        ####################################################################
        # END OF YOUR CODE
        ####################################################################

    def __len__(self) -> int:

        ####################################################################
        # YOUR CODE
        # TODO: Length of the dataset
        ####################################################################
        pass
        ####################################################################
        # END OF YOUR CODE
        ####################################################################
