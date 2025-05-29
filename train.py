import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models import resnet50

# ✅ LazyFakeDataset: 메모리 아끼는 가짜 데이터셋
class LazyFakeDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)            # 각 샘플마다 랜덤 이미지 생성
        y = torch.randint(0, 100, (1,)).item()  # 0~99 사이의 정수 label
        return x, y


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    print(f"[Rank {rank}] Starting training on GPU {rank}")
    start_time = time.time()

    # 모델 정의 및 분산 래핑
    model = resnet50(num_classes=100).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ✅ Lazy 데이터셋: 100,000 샘플
    dataset = LazyFakeDataset(100000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(1, 101):  # 100 에폭
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        model.train()

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(rank, non_blocking=True)
            batch_y = batch_y.to(rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Rank {rank}] Epoch {epoch:3d} - Avg Loss: {avg_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[Rank {rank}] Training completed in {total_time:.2f} seconds")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Total GPUs available: {world_size}")

    if world_size < 2:
        print("❌ Error: Need at least 2 GPUs for multi-GPU training.")
    else:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)