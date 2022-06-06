from .base import AbstractDataloader
from meantime.datasets import dataset_factory
from meantime.utils import all_subclasses
from meantime.utils import import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractDataloader)

DATALOADERS = {c.code():c
               for c in all_subclasses(AbstractDataloader)
               if c.code() is not None}

### meantime data
def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader= DATALOADERS[args.dataloader_code]
    dataloader= dataloader(args, dataset)

    train_mean, val_mean, test_mean = dataloader.get_pytorch_dataloaders()
    return train_mean, val_mean, test_mean

def get_dataloader(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    return dataloader