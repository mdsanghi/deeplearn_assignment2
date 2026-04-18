from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


DATASET_DIR = Path(__file__).resolve().parent.parent / "aclImdb"
VOCAB_FILE = DATASET_DIR / "imdb.vocab"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("<br />", " ")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


@dataclass
class ReviewExample:
    tokens: list[str]
    label: int


def read_imdb_split(split_dir: Path, max_samples_per_label: int | None = None) -> list[ReviewExample]:
    examples: list[ReviewExample] = []
    for label_name, label_value in (("neg", 0), ("pos", 1)):
        review_paths = sorted((split_dir / label_name).glob("*.txt"))
        if max_samples_per_label is not None:
            review_paths = review_paths[:max_samples_per_label]

        for review_path in review_paths:
            text = review_path.read_text(encoding="utf-8", errors="ignore")
            tokens = tokenize(text)
            examples.append(ReviewExample(tokens=tokens, label=label_value))
    return examples


def load_vocab_from_file(vocab_path: Path, max_vocab_size: int = 20_000) -> dict[str, int]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    with vocab_path.open("r", encoding="utf-8", errors="ignore") as vocab_file:
        for line in vocab_file:
            token = line.strip().lower()
            if not token or token in vocab:
                continue
            if len(vocab) >= max_vocab_size:
                break
            vocab[token] = len(vocab)
    return vocab


class IMDBDataset(Dataset):
    def __init__(self, examples: list[ReviewExample], vocab: dict[str, int], max_length: int = 300):
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length
        self.unk_index = vocab[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        example = self.examples[index]
        token_ids = [self.vocab.get(token, self.unk_index) for token in example.tokens[: self.max_length]]
        if not token_ids:
            token_ids = [self.unk_index]
        return token_ids, example.label


def collate_batch(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_length = lengths.max().item()

    padded_sequences = torch.zeros(len(sequences), max_length, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded_sequences[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return padded_sequences, lengths, labels_tensor


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, outputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.score(outputs).squeeze(-1)
        max_length = outputs.size(1)
        mask = torch.arange(max_length, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), outputs).squeeze(1)
        return context, attention_weights


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        model_type: str = "rnn",
        bidirectional: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        self.hidden_multiplier = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if self.model_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
                nonlinearity="tanh",
            )
        elif self.model_type in {"lstm", "lstm_attention"}:
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        output_dim = hidden_dim * self.hidden_multiplier
        self.attention = AttentionPooling(output_dim) if self.model_type == "lstm_attention" else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = self.dropout(self.embedding(inputs))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.recurrent(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        attention_weights = None
        if self.attention is not None:
            features, attention_weights = self.attention(outputs, lengths)
        else:
            if isinstance(hidden, tuple):
                hidden_state = hidden[0]
            else:
                hidden_state = hidden

            if self.bidirectional:
                features = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
            else:
                features = hidden_state[-1]

        logits = self.classifier(self.dropout(features)).squeeze(1)
        return logits, attention_weights


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions) * 100,
        "precision": precision_score(labels, predictions, zero_division=0) * 100,
        "recall": recall_score(labels, predictions, zero_division=0) * 100,
        "f1": f1_score(labels, predictions, zero_division=0) * 100,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    running_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for inputs, lengths, labels in dataloader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits, _ = model(inputs, lengths)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            predictions = (torch.sigmoid(logits) >= 0.5).long()
            all_labels.extend(labels.long().cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = compute_metrics(all_labels, all_predictions)
    average_loss = running_loss / max(len(dataloader), 1)
    return average_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
) -> None:
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        all_labels: list[int] = []
        all_predictions: list[int] = []

        for inputs, lengths, labels in train_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            predictions = (torch.sigmoid(logits) >= 0.5).long()
            all_labels.extend(labels.long().cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

        train_loss = running_loss / max(len(train_loader), 1)
        train_metrics = compute_metrics(all_labels, all_predictions)
        test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )
        print(
            f"  Train -> Accuracy: {train_metrics['accuracy']:.2f}% | "
            f"Precision: {train_metrics['precision']:.2f}% | "
            f"Recall: {train_metrics['recall']:.2f}% | "
            f"F1: {train_metrics['f1']:.2f}%"
        )
        print(
            f"  Test  -> Accuracy: {test_metrics['accuracy']:.2f}% | "
            f"Precision: {test_metrics['precision']:.2f}% | "
            f"Recall: {test_metrics['recall']:.2f}% | "
            f"F1: {test_metrics['f1']:.2f}%"
        )


def print_dataset_summary(train_examples: list[ReviewExample], test_examples: list[ReviewExample], vocab: dict[str, int]) -> None:
    train_lengths = np.array([len(example.tokens) for example in train_examples], dtype=np.int32)
    test_lengths = np.array([len(example.tokens) for example in test_examples], dtype=np.int32)

    print("=" * 90)
    print("IMDB LARGE MOVIE REVIEW DATASET SUMMARY")
    print("=" * 90)
    print(f"Training samples: {len(train_examples)}")
    print(f"Test samples: {len(test_examples)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train review length -> mean: {train_lengths.mean():.2f}, median: {np.median(train_lengths):.2f}, max: {train_lengths.max()}")
    print(f"Test review length  -> mean: {test_lengths.mean():.2f}, median: {np.median(test_lengths):.2f}, max: {test_lengths.max()}")
    print("=" * 90)


def build_dataloaders(
    dataset_dir: Path,
    vocab_path: Path,
    batch_size: int,
    max_vocab_size: int,
    max_length: int,
    max_train_per_label: int | None,
    max_test_per_label: int | None,
) -> tuple[DataLoader, DataLoader, dict[str, int], list[ReviewExample], list[ReviewExample]]:
    train_examples = read_imdb_split(dataset_dir / "train", max_samples_per_label=max_train_per_label)
    test_examples = read_imdb_split(dataset_dir / "test", max_samples_per_label=max_test_per_label)
    vocab = load_vocab_from_file(vocab_path, max_vocab_size=max_vocab_size)

    train_dataset = IMDBDataset(train_examples, vocab=vocab, max_length=max_length)
    test_dataset = IMDBDataset(test_examples, vocab=vocab, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, test_loader, vocab, train_examples, test_examples


def print_model_description(args: argparse.Namespace, vocab_size: int) -> None:
    model_names = {
        "rnn": "Vanilla RNN",
        "lstm": "LSTM",
        "lstm_attention": "LSTM with Attention",
    }
    print("=" * 90)
    print("MODEL CONFIGURATION")
    print("=" * 90)
    print(f"Architecture: {model_names[args.model]}")
    print(f"Vocabulary source: {args.vocab_file}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Recurrent layers: {args.num_layers}")
    print(f"Bidirectional: {args.bidirectional}")
    print(f"Dropout: {args.dropout}")
    print(f"Max tokens per review: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMDB sentiment classification with RNN/LSTM/Attention.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to the extracted aclImdb dataset.")
    parser.add_argument("--vocab-file", type=Path, default=VOCAB_FILE, help="Path to the imdb.vocab file.")
    parser.add_argument("--model", choices=["rnn", "lstm", "lstm_attention"], default="lstm")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab-size", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--max-train-per-label", type=int, default=None)
    parser.add_argument("--max-test-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps. Auto-detected if omitted.")
    return parser.parse_args()


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    device = resolve_device(args.device)
    train_loader, test_loader, vocab, train_examples, test_examples = build_dataloaders(
        dataset_dir=args.dataset_dir,
        vocab_path=args.vocab_file,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
        max_length=args.max_length,
        max_train_per_label=args.max_train_per_label,
        max_test_per_label=args.max_test_per_label,
    )

    print_dataset_summary(train_examples, test_examples, vocab)
    print_model_description(args, vocab_size=len(vocab))
    print(f"Using device: {device}")

    model = SentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model,
        bidirectional=args.bidirectional,
        padding_idx=vocab[PAD_TOKEN],
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
