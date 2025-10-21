"""High-throughput DataLoader patterns for storage-bound training (PyTorch 2.9)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset


@dataclass
class DummyDataset(Dataset):
    length: int = 10_000

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(index)
        image = torch.randn(3, 224, 224)
        label = torch.randint(0, 10, ()).long()
        return image, label


def make_dataloader(dataset: Dataset,
                    batch_size: int = 64,
                    num_workers: int = 4,
                    *,
                    pin_memory: bool = True,
                    prefetch_factor: int = 4,
                    persistent_workers: bool = True) -> DataLoader:
    if num_workers == 0:
        prefetch_factor = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True,
    )


def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                device: torch.device,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                stream: torch.cuda.Stream | None = None) -> None:
    model.train()
    stream = stream or torch.cuda.Stream(device=device) if device.type == "cuda" else None

    for inputs, targets in loader:
        if stream is not None:
            with torch.cuda.stream(stream):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DummyDataset()
    loader = make_dataloader(dataset, batch_size=64, num_workers=8)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()
    train_epoch(model, loader, device, optimizer, criterion)
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"Epoch time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
