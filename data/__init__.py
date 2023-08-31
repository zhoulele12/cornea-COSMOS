"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
from torch.utils.data import Dataset, DataLoader
from data.base_dataset import BaseDataset
from PIL import Image
from torchvision import transforms
import os


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset():
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset()
    """
    data_loader = CustomDatasetDataLoader()
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        # self.opt = opt
        # dataset_class = find_dataset_using_name(opt.dataset_mode)
        # self.dataset = dataset_class(opt)
        # print("dataset [%s] was created" % type(self.dataset).__name__)
        transform_A = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        transform_B = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.dataset = CustomDataset(root_dir_A='train/HE', root_dir_B='train/BlueWhite', transform_A=transform_A,
                                transform_B=transform_B)
        # Create the dataloader
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), 99999999)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * 32 >= 99999999:
                break
            yield data

class CustomDataset(Dataset):
    def __init__(self, root_dir_A, root_dir_B, transform_A=None, transform_B=None):
        self.root_dir_A = root_dir_A
        self.root_dir_B = root_dir_B
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.images_A = []
        self.masks_A = []
        for image_name in os.listdir(os.path.join(root_dir_A, 'images')):
            base_name, ext = os.path.splitext(image_name)
            if ext in ['.jpg', '.tif']:
                mask_name = base_name + '.jpg'
                if os.path.isfile(os.path.join(root_dir_A, 'masks', mask_name)):
                    self.images_A.append(image_name)
                    self.masks_A.append(mask_name)

        self.images_B = [f for f in os.listdir(root_dir_B) if 'Image' in f]
        self.masks_B = [f for f in os.listdir(root_dir_B) if 'Mask' in f]


    def __len__(self):
        return max(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        img_name_A = os.path.join(self.root_dir_A, 'images', self.images_A[idx % len(self.images_A)])
        mask_name_A = os.path.join(self.root_dir_A, 'masks', self.masks_A[idx % len(self.masks_A)])

        img_name_B = os.path.join(self.root_dir_B, self.images_B[idx % len(self.images_B)])
        img_basename_B = os.path.splitext(self.images_B[idx % len(self.images_B)])[0]
        mask_name_B = os.path.join(self.root_dir_B, img_basename_B.replace('Image', 'Mask') + '.gif')

        image_A = Image.open(img_name_A)
        mask_A = Image.open(mask_name_A)
        image_B = Image.open(img_name_B)
        mask_B = Image.open(mask_name_B)

        if self.transform_A:
            image_A = self.transform_A(image_A)
            mask_A = self.transform_A(mask_A)

        if self.transform_B:
            image_B = self.transform_B(image_B)
            mask_B = self.transform_B(mask_B)

        sample = {'A': {'image': image_A, 'mask': mask_A},
                  'B': {'image': image_B, 'mask': mask_B}}

        return sample
