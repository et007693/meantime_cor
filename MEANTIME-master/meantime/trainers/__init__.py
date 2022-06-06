from .base import AbstractTrainer
from meantime.utils import all_subclasses
from meantime.utils import import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractTrainer)

TRAINERS = {c.code():c
            for c in all_subclasses(AbstractTrainer)
            if c.code() is not None}

### meantime
def trainer_factory_mean(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
    
### bert_side
def trainer_factory_side(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)

