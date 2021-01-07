import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_BERT import LitSEBBERT

checkpoint_callback = ModelCheckpoint(
    monitor="val_macro",
    mode="max",
    filepath='models/seb_bert_{epoch}-{val_macro:.4f}',
    save_top_k=3
)
test = LitSEBBERT()
trainer = pl.Trainer(
    gpus=1,
    # num_nodes=1,
    # distributed_backend='ddp',
    max_epochs=8,
    checkpoint_callback=checkpoint_callback,
    accumulate_grad_batches=2,
    precision=16,
    amp_level='O2'
)

trainer.fit(test)

print("finished training")
