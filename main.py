from datasets import DTIDataset, collate_fn
from model import DTIModel
from trainer import Trainer
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, random_split


def main():
    csv_path = "./data/dataset/drugbank/drugbank.csv"

    dataset = DTIDataset(csv_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            dataset.drug_tokenizer.pad_token_id,
            dataset.protein_tokenizer.pad_token_id
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b,
            dataset.drug_tokenizer.pad_token_id,
            dataset.protein_tokenizer.pad_token_id
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = DTIModel()

    trainer = Trainer(model, train_loader, test_loader, device=device)

    # 训练
    epochs = 10
    for epoch in range(epochs):
        train_loss = trainer.train_one_epoch()
        evaluate_result = trainer.evaluate()

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, test_loss={evaluate_result["loss"]:.4f}, precision={evaluate_result["precision"]:.4f}, recall={evaluate_result["recall"]:.4f}, f1={evaluate_result["f1"]:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "dti_model.pt")
    print("Model saved.")


if __name__ == "__main__":
    main()
