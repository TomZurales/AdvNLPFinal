import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AlbertTokenizerFast, TrainingArguments, Trainer

with open('data/bypublisher/articles.txt') as f:
    articles = f.readlines()
articles = [x.strip().lower() for x in articles] 

with open('data/bypublisher/labels.txt') as f:
    labels = f.readlines()
labels = [int(x.strip()) for x in labels]

train_texts, other_texts, train_labels, other_labels = train_test_split(articles, labels, test_size=.2)

test_texts = other_texts[len(other_texts)//2:]

test_labels = other_labels[len(other_labels)//2:]

tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v1")

test_encodings = tokenizer(test_texts, truncation=True, padding='max_length')

class HyperpartisanshipDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = HyperpartisanshipDataset(test_encodings, test_labels)

model = AutoModel.from_pretrained("models")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
)

# model.cuda()

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=test_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

trainer.evaluate()