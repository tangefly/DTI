import torch
import torch.nn as nn

class DTIModel(nn.Module):
    def __init__(self, drug_vocab=767, protein_vocab=33, dim=256):
        super().__init__()

        self.dim = dim

        # ===== Embedding =====
        self.drug_embedding = nn.Embedding(drug_vocab, dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab, dim, padding_idx=0)

        self.dropout = nn.Dropout(0.1)

        # ===== Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim*4,
            dropout=0.1,
            batch_first=True,
            norm_first=True   # 更稳定
        )

        self.drug_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.protein_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ===== 双向 cross attention =====
        self.cross_attn_dp = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.cross_attn_pd = nn.MultiheadAttention(dim, 8, batch_first=True)

        self.norm_drug = nn.LayerNorm(dim)
        self.norm_protein = nn.LayerNorm(dim)

        # ===== attention pooling =====
        self.pool = nn.Linear(dim, 1)

        # ===== classifier =====
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def attention_pool(self, x, mask):
        # x: (B, L, D)
        score = self.pool(x).squeeze(-1)  # (B, L)

        score = score.masked_fill(
            mask == 0,
            torch.finfo(score.dtype).min
        )
        weight = torch.softmax(score, dim=1)

        return torch.sum(x * weight.unsqueeze(-1), dim=1)

    def forward(self, batch):
        drug_ids = batch["drug_input_ids"]
        drug_mask = batch["drug_attention_mask"]

        protein_ids = batch["protein_input_ids"]
        protein_mask = batch["protein_attention_mask"]

        # ===== embedding =====
        drug = self.dropout(self.drug_embedding(drug_ids))
        protein = self.dropout(self.protein_embedding(protein_ids))

        # ===== encoder =====
        drug = self.drug_encoder(drug, src_key_padding_mask=(drug_mask == 0))
        protein = self.protein_encoder(protein, src_key_padding_mask=(protein_mask == 0))

        # ===== cross attention (双向 + residual) =====
        drug2, _ = self.cross_attn_dp(
            query=drug,
            key=protein,
            value=protein,
            key_padding_mask=(protein_mask == 0)
        )
        drug = self.norm_drug(drug + drug2)

        protein2, _ = self.cross_attn_pd(
            query=protein,
            key=drug,
            value=drug,
            key_padding_mask=(drug_mask == 0)
        )
        protein = self.norm_protein(protein + protein2)

        # ===== pooling（比 mean 强很多）=====
        drug_vec = self.attention_pool(drug, drug_mask)
        protein_vec = self.attention_pool(protein, protein_mask)

        # ===== prediction =====
        x = torch.cat([drug_vec, protein_vec], dim=-1)

        out = self.mlp(x).squeeze(-1)
        out = torch.sigmoid(out)

        return out
