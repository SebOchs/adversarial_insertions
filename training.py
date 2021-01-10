import sys
import os
# Slurm fix
sys.path.append(os.getcwd())
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_Model import LitBERT, LitT5



def training(data_set_name, training_set, test_set, mode, batch_size=8, lr=0.00002, precision=False, accumulate_grad=8,
             ddp=False, val='', labels=0):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro",
        mode="max",
        filepath="models/" + data_set_name + '_' + mode + '_' + '{epoch}-{val_macro:.4f}',
        save_top_k=3
    )
    if mode == 'bert':
        model = LitBERT(training_set, test_set, batch_size, lr, val=val, num_labels=labels)
    if mode == 'T5':
        model = LitT5(training_set, test_set, batch_size, val=val)
    trainer = pl.Trainer(gpus=1, max_epochs=8, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=100,
                         accumulate_grad_batches=accumulate_grad, check_val_every_n_epoch=1, num_sanity_val_steps=0)
    if precision:
        trainer = pl.Trainer(gpus=1, max_epochs=8, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, precision=16, amp_level='O2',)
    if ddp:
        trainer = pl.Trainer(gpus=2, max_epochs=8, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, num_nodes=1, distributed_backend='ddp')
    if ddp and precision:
        trainer = pl.Trainer(gpus=2, max_epochs=8, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, num_nodes=1, distributed_backend='ddp',
                             precision=16, amp_level='O2')



    trainer.fit(model)
    print("finished training")


# mnli

# training("mnli", "datasets/preprocessed/bert/MNLI/train.npy", "datasets/preprocessed/bert/MNLI/dev_mm.npy", 'bert',
#         batch_size=64, precision=True, accumulate_grad=1, ddp=
# Server training
"""
training("mnli", "datasets/preprocessed/T5/MNLI/train.npy", "datasets/preprocessed/T5/MNLI/dev_mm.npy", 'T5',
         batch_size=16, precision=True, accumulate_grad=4, ddp=True)
"""
# qqp
"""
training("qqp", "datasets/preprocessed/bert/qqp/train.npy", "datasets/preprocessed/bert/qqp/dev.npy", 'bert',
         batch_size=64, precision=True, accumulate_grad=1, ddp=True, labels=2)
"""
training("qqp", "datasets/preprocessed/T5/qqp/train.npy", "datasets/preprocessed/T5/qqp/dev.npy", 'T5',
         batch_size=16, precision=True, accumulate_grad=4, ddp=True)

"""
# msrpc
training("msrpc", "datasets/preprocessed/bert/MSpara/train.npy", "datasets/preprocessed/bert/MSpara/test.npy", 'bert',
         batch_size=32, precision=True, accumulate_grad=2, labels=2)

training("msrpc", "datasets/preprocessed/T5/MSpara/train.npy", "datasets/preprocessed/T5/MSpara/test.npy", 'T5',
         batch_size=12, precision=True, accumulate_grad=4)



# RTE
training("rte", "datasets/preprocessed/bert/RTE/train.npy", "datasets/preprocessed/bert/RTE/dev.npy", 'bert',
         val="datasets/preprocessed/bert/RTE/dev.npy",
         batch_size=32, accumulate_grad=1, labels=2, precision=True)


training("rte", "datasets/preprocessed/T5/RTE/train.npy", "datasets/preprocessed/T5/RTE/dev.npy", 'T5',
         batch_size=8, precision=True, accumulate_grad=1, val="datasets/preprocessed/T5/RTE/dev.npy")

training("wic", "datasets/preprocessed/bert/wic/train.npy", "datasets/preprocessed/bert/wic/dev.npy", 'bert',
         val="datasets/preprocessed/bert/RTE/dev.npy",
         batch_size=16, accumulate_grad=1, labels=2)
"""