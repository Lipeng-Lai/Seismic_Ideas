import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import os
import csv
from datetime import datetime
from tqdm import tqdm
from data.build_data_2Channel import *
from configs.config import get_config
from model.build import build_model
# from loss.build import build_loss
from optimizer.lr_scheduler import build_scheduler
from optimizer.optimizer import build_optimizer
from metrics.metrics import compute_mse, compute_psnr, compute_snr, compute_ssim
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def custom_dual_channel_l1_loss(pred, target):
    """
    自定义损失函数：
    L = L1(pred[:,0], target[:,0]) + L1(pred[:,1], target[:,1])
    适用于 (N, 2, H, W) 形式的输入输出。
    """
    return torch.nn.functional.l1_loss(pred[:, 0], target[:, 0]) + \
           torch.nn.functional.l1_loss(pred[:, 1], target[:, 1])


def main(config):
    # ============================================init============================================
    os.makedirs(config.train.output, exist_ok=True)
    checkpoint_path = os.path.join(config.train.output, config.model.name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(checkpoint_path, f"training_log_{timestamp}.csv")

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "MSE", "PSNR", "SNR", "SSIM", "timestamp"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ============================================init============================================

    # ============================================Data loader============================================
    image_path1 = '/home/wwd/deeplearning/data/multiple'
    image_path2 = '/home/wwd/deeplearning/data/multiple_hilbert'
    
    label_path1 = '/home/wwd/deeplearning/data/output'
    label_path2 = '/home/wwd/deeplearning/data/output_hilbert'
    full_dataset = MyDataset(image_path1, image_path2, label_path1, label_path2, transform=get_transforms(is_train=True))

    val_ratio = 0.2
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = get_transforms(is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False, pin_memory=True)
    # ============================================Data loader============================================

    # ============================================build============================================
    net = build_model(config).to(device)

    optimizer = optim.Adam(net.parameters(), lr=config.optimizer.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.optimizer.weight_decay,
        patience=config.optimizer.lr_patience,
    )

    criterion = custom_dual_channel_l1_loss
    # ============================================build============================================

    # ==========================================checkpoint resume==========================================
    start_epoch = 0
    checkpoint_files = []
    if os.path.exists(checkpoint_path):
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pth")]
    latest_checkpoint = None
    max_epoch = -1

    if checkpoint_files:
        for checkpoint_file in checkpoint_files:
            checkpoint_file_path = os.path.join(checkpoint_path, checkpoint_file)
            try:
                checkpoint = torch.load(checkpoint_file_path, map_location=device, weights_only=False)
                current_epoch = checkpoint.get("epoch", 0)
                if current_epoch > max_epoch:
                    max_epoch = current_epoch
                    latest_checkpoint = checkpoint_file_path
            except Exception as e:
                print(f"加载检查点 {checkpoint_file_path} 失败，原因：{e}")

        if latest_checkpoint:
            checkpoint = torch.load(checkpoint_file_path, map_location=device, weights_only=False)
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"已加载检查点 {latest_checkpoint}，epoch 为 {start_epoch}，恢复训练。")
    # ==========================================checkpoint resume==========================================

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = time.time()

        net.train()
        train_loss = 0.0
        total_samples = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train.epochs} [Train]", unit="batch") as pbar:
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.float32)

                output = net(batch_x)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                total_samples += batch_x.size(0)
                pbar.set_postfix({"loss": f"{loss.item()/batch_x.size(0):.6f}"})

        avg_train_loss = train_loss / total_samples

        # ==========================================val==========================================
        net.eval()

        val_loss = torch.tensor(0.0, device=device)
        val_mse = torch.tensor(0.0, device=device)
        val_psnr = torch.tensor(0.0, device=device)
        val_snr = torch.tensor(0.0, device=device)
        val_ssim = torch.tensor(0.0, device=device)
        total_val_samples = 0

        with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch") as pbar:
            for val_x, val_y in pbar:
                val_x = val_x.to(device, dtype=torch.float32)
                val_y = val_y.to(device, dtype=torch.float32)

                output = net(val_x)
                loss = criterion(output, val_y)
                batch_size = val_x.size(0)

                batch_mse = compute_mse(output, val_y)
                batch_psnr = compute_psnr(output, val_y)
                batch_snr = compute_snr(output, val_y)
                batch_ssim = compute_ssim(output, val_y)

                val_loss += loss.detach() * batch_size
                val_mse += batch_mse.detach() * batch_size
                val_psnr += batch_psnr.detach() * batch_size
                val_snr += batch_snr.detach() * batch_size
                val_ssim += batch_ssim.detach() * batch_size

                total_val_samples += batch_size
                pbar.set_postfix({"val_loss": f"{loss.item()/batch_size:.6f}"})

        avg_val_loss = val_loss / total_val_samples
        avg_val_mse = (val_mse / total_val_samples).item()
        avg_val_psnr = (val_psnr / total_val_samples).item()
        avg_val_snr = (val_snr / total_val_samples).item()
        avg_val_ssim = (val_ssim / total_val_samples).item()

        scheduler.step(avg_val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        # ==========================================val==========================================

        # ==========================================write-save==========================================
        log_data = [
            epoch+1,
            f"{avg_train_loss:.6f}",
            f"{avg_val_loss:.6f}",
            f"{current_lr:.6e}",
            f"{avg_val_mse:.6f}",
            f"{avg_val_psnr:.6f}",
            f"{avg_val_snr:.6f}",
            f"{avg_val_ssim:.6f}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_val_loss,
            }, os.path.join(checkpoint_path, "best_model.pth"))

        if (epoch+1) % config.train.save_freq == 0:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_val_loss,
            }, os.path.join(checkpoint_path, f"model_epoch_{epoch+1}.pth"))

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{config.train.epochs}|"
              f"Train Loss: {avg_train_loss:.6f}|"
              f"Val Loss: {avg_val_loss:.6f}|"
              f"MSE:{avg_val_mse:.6f}|"
              f"PSNR:{avg_val_psnr:.6f}|"
              f"SNR:{avg_val_snr:.6f}|"
              f"SSIM:{avg_val_ssim:.6f}|"
              f"lr:{current_lr:.6f}|"
              f"Time: {epoch_time:.2f}s")
        # ==========================================write-save==========================================


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = get_config(config_path)
    seed_everything(config.train.seed)
    main(config)
