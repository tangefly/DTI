from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer

class DTIDataset(Dataset):
    def __init__(self, csv_path, max_len=2048):
        df = pd.read_csv(csv_path)

        self.compounds = df["compound"].values
        self.proteins = df["protein"].values
        self.labels = df["label"].values

        self.max_len = max_len

        self.drug_tokenizer = AutoTokenizer.from_pretrained(
            "./tokenizers/drug",
            use_fast=True
        )

        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            "./tokenizers/protein",
            use_fast=False
        )

    def __len__(self):
        return len(self.compounds)

    def pad(self, ids, pad_id):
        if len(ids) >= self.max_len:
            return ids[:self.max_len], [1] * self.max_len

        padding_len = self.max_len - len(ids)
        return ids + [pad_id] * padding_len, [1]*len(ids) + [0]*padding_len

    def __getitem__(self, idx):
        smiles = self.compounds[idx]
        protein = self.proteins[idx]
        label = self.labels[idx]

        # drug
        tokens = self.drug_tokenizer.tokenize(smiles)
        drug_ids = self.drug_tokenizer.convert_tokens_to_ids(tokens)
        drug_ids = drug_ids[:self.max_len]

        # protein（不 padding）
        protein_enc = self.protein_tokenizer(
            protein,
            truncation=True,
            max_length=self.max_len,
            padding=False
        )

        return {
            "drug_input_ids": drug_ids,
            "protein_input_ids": protein_enc["input_ids"],
            "label": label
        }

def collate_fn(batch, drug_pad_id, protein_pad_id):
    drug_seqs = [item["drug_input_ids"] for item in batch]
    protein_seqs = [item["protein_input_ids"] for item in batch]
    labels = [item["label"] for item in batch]

    # 动态长度
    drug_max_len = max(len(seq) for seq in drug_seqs)
    protein_max_len = max(len(seq) for seq in protein_seqs)

    def pad(seqs, max_len, pad_id):
        padded = []
        masks = []
        for seq in seqs:
            pad_len = max_len - len(seq)
            padded.append(seq + [pad_id]*pad_len)
            masks.append([1]*len(seq) + [0]*pad_len)
        return torch.tensor(padded), torch.tensor(masks)

    drug_ids, drug_mask = pad(drug_seqs, drug_max_len, drug_pad_id)
    protein_ids, protein_mask = pad(protein_seqs, protein_max_len, protein_pad_id)

    return {
        "drug_input_ids": drug_ids,
        "drug_attention_mask": drug_mask,
        "protein_input_ids": protein_ids,
        "protein_attention_mask": protein_mask,
        "label": torch.tensor(labels, dtype=torch.float)
    }