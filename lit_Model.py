import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, RandomSampler
from transformers import BertForSequenceClassification, AdamW, Adafactor, BertTokenizer, BertConfig, T5Tokenizer, \
    T5ForConditionalGeneration, T5Config
from utils import macro_f1, weighted_f1
from dataloading import MyBertDataset, MyT5Dataset
import warnings

warnings.filterwarnings("ignore")


class LitBERT(pl.LightningModule):

    def __init__(self, train, test, num_labels=3):
        super(LitBERT, self).__init__()

        config = BertConfig('bert-base-uncased', max_position_embeddings=128)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,
                                                                   max_position_embeddings=128)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=config)
        self.train = [round(len(MyBertDataset(train)) * 0.9), round(len(MyBertDataset(train)) * 0.1)]
        self.train_set_split = self.train if sum(self.train) == len(MyBertDataset(train)) else [self.train[0],
                                                                                               self.train[1] + 1]
        self.train_data, self.val_data = random_split(
            MyBertDataset(train), self.train_set_split, generator=torch.Generator().manual_seed(42))
        self.test_data = MyBertDataset(test)

    def forward(self, tok_seq):
        return self.model(input_ids=tok_seq[0], token_type_ids=tok_seq[1], attention_mask=tok_seq[2],
                              labels=tok_seq[3])


    def training_step(self, batch, batch_idx):
        text, seg, att, lab = batch
        return self.model(input_ids=text, token_type_ids=seg, attention_mask=att, labels=lab)[0].mean()



    def validation_step(self, batch, batch_idx):
        text, seg, att, lab = batch
        return {'prediction': torch.argmax(self(batch)[1]).item(), 'truth': lab.squeeze().item()}



    def validation_epoch_end(self, outputs):
        pred = [x['prediction'] for x in outputs]
        lab = [x['truth'] for x in outputs]
        acc_stack = np.stack((pred, lab), axis=0)
        acc = np.sum([1 for i in range(len(acc_stack.T)) if acc_stack[0, i] == acc_stack[1, i]]) / len(acc_stack.T)
        m_f1 = macro_f1(pred, lab)
        w_f1 = weighted_f1(pred, lab)
        print("Accuracy: " + str(acc)[:6] + ", Macro-F1: " + str(m_f1)[:6] + ", Weighted-F1 " + str(w_f1)[:6])
        self.log("val_macro", m_f1)

    # prepared for later use
    def test_step(self, batch, batch_idx):
        text, seg, att, lab = batch
        return {'prediction': torch.argmax(self(batch)[1]).item(), 'truth': lab.squeeze().item()}

    def test_epoch_end(self, outputs):
        pred = [x['prediction'] for x in outputs]
        lab = [x['truth'] for x in outputs]
        acc_stack = np.stack((pred, lab), axis=0)
        acc = np.sum([1 for i in range(len(acc_stack.T)) if acc_stack[0, i] == acc_stack[1, i]]) / len(acc_stack.T)
        m_f1 = macro_f1(pred, lab)
        w_f1 = weighted_f1(pred, lab)
        print("Accuracy: " + str(acc)[:6])
        print("Macro-F1: " + str(m_f1)[:6])
        print("Weighted-F1 " + str(w_f1)[:6])
        self.log("test_macro", m_f1)
        self.log("test_weighted", w_f1)
        self.log("test_acc", acc)
        print("Accuracy: " + str(acc)[:6] + ", Macro-F1: " + str(m_f1)[:6] + ", Weighted-F1 " + str(w_f1)[:6])

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=0.00002, correct_bias=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, num_workers=0, shuffle=True)

    def val_dataloader(self):
        """
        val_sampler = RandomSampler(self.val_data)
        """
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)


class LitT5(pl.LightningModule):

    def __init__(self, train, test):
        super(LitT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-large', n_positions=128)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        if sum([round(len(MyT5Dataset(train)) * 0.9), round(len(MyT5Dataset(train)) * 0.1)]) == len(MyT5Dataset(train)):
            train_set_split = [round(len(MyT5Dataset(train)) * 0.9), round(len(MyT5Dataset(train)) * 0.1)]
        else:
            train_set_split = [round(len(MyT5Dataset(train)) * 0.9), round(len(MyT5Dataset(train)) * 0.1) + 1]

        self.train_data, self.val_data = random_split(
            MyT5Dataset(train), train_set_split, generator=torch.Generator().manual_seed(42))
        self.test_data = MyT5Dataset(test)

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, max_length=4)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, attn, lab, lab_attn = batch
        return self.model(input_ids=text, attention_mask=attn, labels=lab)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, attn, lab, lab_attn = batch
        return {'prediction': self(text, attn), 'truth': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True)}

    def validation_epoch_end(self, outputs):
        pred = [x['prediction'] for x in outputs]
        lab = [x['truth'] for x in outputs]
        acc_stack = np.stack((pred, lab), axis=0)
        acc = np.sum([1 for i in range(len(acc_stack.T)) if acc_stack[0, i] == acc_stack[1, i]]) / len(acc_stack.T)
        m_f1 = macro_f1(pred, lab)
        w_f1 = weighted_f1(pred, lab)
        print("Accuracy: " + str(acc)[:6] + ", Macro-F1: " + str(m_f1)[:6] + ", Weighted-F1 " + str(w_f1)[:6])
        self.log("val_macro", m_f1)


    def test_step(self, batch, batch_idx):
        text, lab = batch
        return {'prediction': self(text), 'truth': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True)}

    def test_epoch_end(self, outputs):
        pred = [x['prediction'] for x in outputs]
        lab = [x['truth'] for x in outputs]
        acc_stack = np.stack((pred, lab), axis=0)
        acc = np.sum([1 for i in range(len(acc_stack.T)) if acc_stack[0, i] == acc_stack[1, i]]) / len(acc_stack.T)
        m_f1 = macro_f1(pred, lab)
        w_f1 = weighted_f1(pred, lab)
        print("Accuracy: " + str(acc)[:6])
        print("Macro-F1: " + str(m_f1)[:6])
        print("Weighted-F1 " + str(w_f1)[:6])
        self.log("test_macro", m_f1)
        self.log("test_weighted", w_f1)
        self.log("test_acc", acc)
        print("Accuracy: " + str(acc)[:6] + ", Macro-F1: " + str(m_f1)[:6] + ", Weighted-F1 " + str(w_f1)[:6])

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)
