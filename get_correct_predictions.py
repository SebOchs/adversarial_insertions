import torch
from lit_BERT import LitBERT


device = torch.device("cuda")
# Load checkpoint and get necessary objects
checkpoint = LitBERT.load_from_checkpoint("models/seb_bert_epoch=4-val_macro=0.7680.ckpt")
model = checkpoint.model
model.cuda()
model.eval()
tokenizer = checkpoint.tokenizer
# only get incorrect data instances
val_data = checkpoint.val_dataloader()
correct_predictions = []
for i in val_data:
    pred = torch.argmax(model(input_ids=i[0].to(device), token_type_ids=i[1].to(device),
                              attention_mask=i[2].to(device))[0]).item()
    if pred == i[3]:
        correct_predictions.append(i)


print("Accuracy: ", len(correct_predictions)/len(val_data))
torch.save(correct_predictions, "datasets/preprocessed/bert/sciEntsBank/correct.pt")
