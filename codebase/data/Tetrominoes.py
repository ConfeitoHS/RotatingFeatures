from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset
import h5py


class TetDataset(Dataset):
    def __init__(self, opt: DictConfig, partition: str) -> None:
        """
        Initialize the 4Shapes datasets.

        This dataset class can load the following versions of the 4Shapes dataset:

        The 4Shapes dataset comprises grayscale images of dimensions 32 x 32, each containing four distinct
        white shapes (square, up/downward facing triangle, circle) on a black background. The dataset consists
        of 50,000 images in the train set, and 10,000 images for the validation and test sets, respectively.
        All pixel values fall within the range [0,1].

        The 4Shapes RGB(-D) datasets follow the same general setup, but randomly samples the color of each shape.
        To create the RGB-D variant, we incorporate a depth channel to each image and assign a unique depth value
        within the range [0,1] to every object, maintaining equal distances between them.
        By changing the configuration options, it is possible to choose the number of colors used throughout the
        dataset (opt.input.num_rand_colors) and whether the depth channel is included (opt.input.add_depth_channel).

        Args:
            opt (DictConfig): Configuration options.
            partition (str): Dataset partition ("train", "val", or "test").
        """
        super(TetDataset, self).__init__()

        self.opt = opt
        self.root_dir = Path(opt.cwd, opt.input.load_path)
        self.partition = partition

        h5_root = partition + "/"
        if opt.input.file_name == "Tetrominoes":
            if opt.input.colored:
                file_name = Path(
                    self.root_dir,
                    f"{opt.input.file_name}_Color_Lv{opt.input.condensed_level}.h5",
                )

            else:
                file_name = Path(
                    self.root_dir,
                    f"{opt.input.file_name}_Gray_Lv{opt.input.condensed_level}.h5",
                )

        else:
            raise Exception()
        h5_root += str(opt.input.num_answers)
        print(h5_root)
        # dataset = np.load(file_name)
        hfp = h5py.File(file_name, "r")
        dataset = hfp[h5_root]
        self.images = np.array(dataset["image"]).astype(np.float32) / 255.0  # C H W
        if opt.input.ceil_input:
            self.images = np.ceil(self.images)
        self.pixelwise_instance_labels = np.array(dataset["mask"][:, 0, :]).astype(
            np.float32
        )  # multi, H, W
        self.y = np.array(dataset["y_x_sh"][:, :, 0])
        self.x = np.array(dataset["y_x_sh"][:, :, 1])
        self.sh = np.array(dataset["y_x_sh"][:, :, 2])
        hfp.close()

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input image and corresponding gt_labels.
        """
        input_image = self.images[idx]
        labels = {
            "pixelwise_instance_labels": self.pixelwise_instance_labels[idx],
            "shape": self.sh[idx],
        }
        return input_image, labels
