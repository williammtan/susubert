import re
import torch
from torch.utils.data import Dataset

class BertMatcherDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=150):
        self.dataset = dataset
        self.dataset['name1'] = self.dataset['name1'].map(self.clean_text)
        self.dataset['name2'] = self.dataset['name2'].map(self.clean_text)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def clean_text(self, text):
        if text == None:
            return 'None'
        text = re.sub(r'\\.', '', text)  # Remove all \n \t etc..
        text = re.sub(r'[^\w\s]*', '', text)  # Remove anything not a digit, letter, or space
        return text.strip().lower()

    def __len__(self):
        return len(self.dataset)

    def getColumn(self, col_name):
        return self.dataset[col_name]

    def __getitem__(self, idx):
        # TODO: encode features as [COL] title [VAL] susu beruang 200ml [COL] price [VAL] 30.000 [KEY] weight [VAL] 300g

        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataset.iloc[idx, :]

        x = self.tokenizer.encode(
            text=row['name1'], text_pair=row['name2'], 
            add_special_tokens=True, 
            truncation="longest_first", 
            max_length=self.max_len, 
            return_tensors='pt', 
            pad_to_max_length=True,
        )
        y = row['match']

        return x, y

