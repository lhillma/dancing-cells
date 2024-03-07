from typing import Iterable
from functools import partial
import itertools
from pathlib import Path

from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import functional as fn
import numpy as np

from numba import njit

# from torch.utils.data.dataloader import multiprocessing

from tqdm import tqdm

from cell2image import image as cimg
from pathlib import Path

from typing import Iterable, Dict
from pathlib import Path
import numpy as np
import torch  # Assuming you're using PyT

from cell2image import image as cimg
from torch import Tensor

from typing import Dict
from pathlib import Path
import numpy as np
import torch  # Assuming you're using PyTorch for the Tensor type

def extract_extra_features(base_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    dict_images = {}

    # Iterate over all .vtk files in the given directory
    for path in base_path.glob("*.vtk"):
        frame = cimg.read_vtk_frame(path)
        cell_features = {}  # Dictionary for storing features for each cell_id

        for clus_id in range(0, 99):  # Assuming you're interested in cell IDs 1 through 99
            list_of_features = []
            cropped_cytoplasm = cimg.crop_cells_by_id(frame=frame, central_cell_id=clus_id, size=70)
            cropped_nucleus = cimg.crop_cells_by_id(frame=frame, central_cell_id=clus_id, size=70, cell_type=2)
            
            
            # Calculate features and append to the list
            list_of_features.append(np.sum(cropped_cytoplasm[:,:,1] > 1))
            list_of_features.append(np.sum(cropped_nucleus[:,:,1] > 1))
            list_of_features.append(np.sum(np.where(cimg.crop_cell_and_neighbours(cell_id=clus_id, frame=frame, size=70, neighbour_order=0)[:,:,1] > 1, 1, 0)))

            list_of_features.append(np.sum(np.where(cimg.crop_cell_and_neighbours(cell_id=clus_id, frame=frame, size=70, neighbour_order=1)[:,:,1] > 1, 1, 0)))

            list_of_features.append(np.sum(np.where(cimg.crop_cell_and_neighbours(cell_id=clus_id, frame=frame, size=70, neighbour_order=2)[:,:,1] > 1, 1, 0)))

            list_of_features.append(len(cimg.get_cell_neighbour_ids(   cell_ids = frame.cluster_id, neighbour_order=1, cell_id=clus_id)))
            list_of_features.append(len(cimg.get_cell_neighbour_ids(   cell_ids = frame.cluster_id, neighbour_order=2, cell_id=clus_id)))

            


            # Store the features list for the current cell_id
            cell_features[clus_id] = list_of_features

        # Convert the list of features to Tensors and assign to the outer dictionary with path as key
        dict_images[str(path)] = {cell_id: torch.tensor(features) for cell_id, features in cell_features.items()}

    return dict_images


