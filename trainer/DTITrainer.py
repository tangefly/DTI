import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-4,
        device=None,
        grad_clip=1.0,
        use_amp=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _move_to_device(self, batch):
        return {
            k: v.to(self.device) for k, v in batch.items()
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
        # for batch in self.train_loader:
            batch = self._move_to_device(batch)

            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss = self.criterion(outputs, batch["label"])

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        all_logits = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc="Evaluating"):
        # for batch in self.val_loader:
            batch = self._move_to_device(batch)

            logits = self.model(batch)  # (B,)
            loss = self.criterion(logits, batch["label"])

            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

        # 拼接
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # 转概率
        probs = torch.sigmoid(all_logits)

        # 二值化
        preds = (probs > 0.5).int().numpy()
        labels = all_labels.int().numpy()

        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        return {
            "loss": total_loss / len(self.val_loader),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            if val_loss is not None:
                print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            else:
                print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f}")
