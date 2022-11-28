import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from transformers import TrOCRProcessor
from transformers import AutoTokenizer
from transformers import VisionEncoderDecoderModel
from transformers import AdamW
from tqdm.notebook import tqdm
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
from transformers import default_data_collator


# ------------------------------------------ class metric --------------------------------------
cer_metric = load_metric("cer")
def compute_cer(pred_ids, labels_ids):

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer      

# ------------------------------------------ class data -----------------------------------------
class MYDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        x = feature_extractor(image, return_tensors='pt')
        for k,v in x.items():
          pixel_values = v
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length, truncation=True).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# ------------------------------------------ preparing input dataframe data --------------------------------------

data = pd.read_csv('data/IDPL-PFOD/INFO.csv') 
df = pd.DataFrame(columns =["file_name","text"])
path = 'data/IDPL-PFOD/converted/converted/'
print(path)
print(data)
i=0
x=[]
for item in os.listdir(path):
  # print(item)
  i += 1
  i_text = str(i)
  newpath = "0"* (5 - len(i_text))+i_text
  label = f"{path}/{newpath}"
  x.append(f"{newpath}.jpg")
print(x)  
df['file_name'] = x
df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
df['text']=data['true text']
# df.head()
# df.tail()

# ------------------------------------------- seprate test and train data --------------------------------------
# 1- using sklearn function to seprate test and train data
# 2- reset the indixes to start from zero
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
# ------------------------  initialize the training and evaluation datasets ----------------------------


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
processor = TrOCRProcessor(feature_extractor = feature_extractor, tokenizer = tokenizer)

train_dataset = MYDataset(root_dir='data/IDPL-PFOD/converted/converted/',
                           df=train_df,
                           processor=processor, max_target_length=128)
eval_dataset = MYDataset(root_dir='data/IDPL-PFOD/converted/converted/',
                           df=test_df,
                           processor=processor, max_target_length=128)


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))        

encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)
encoding = eval_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

# ------------------------------------------------- load data pytorch
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ------------------------------------------------- model

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224-in21k", "xlm-roberta-base")
model.to(device)


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# ---------------------------------------- optimization and training
optimizer = AdamW(model.parameters(), lr=5e-5)
for epoch in range(10):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(eval_dataloader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       print(batch["labels"])
       cer = compute_cer(outputs, batch["labels"])
       valid_cer += cer

   print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained("TrOCR_custom")