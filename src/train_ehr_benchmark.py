import argparse
import os
from pathlib import Path

import numpy as np
import toml
import torch
from benchmark.datasets import MM_dataset
from benchmark.ehr_module import EHR_module
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def main(args):
    data_path = args.data_path
    saved_model_path = args.saved_model_path

    config = toml.load("config.toml")["train"]

    # load params from config
    task = config["task"]
    use_ratio = config["use_ratio"]
    best_test_only = config["best_test_only"]
    longstay_mintime = config["longstay_mintime"]  # hrs
    epochs = config["epochs"]
    use_amp = config["use_amp"]
    learning_rate = config["learning_rate"]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        use_amp = False  # override for mps
    else:
        device = "cpu"

    print(f"on the {device} device")
    print("run ehr partial experiment")
    print(f"task: {task}")
    if use_ratio:
        print("use ratio based threshold")

    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(norm_mean, norm_std),
        ]
    )

    train_dataset = MM_dataset(
        split="train",
        img_transform=img_transform,
        task=task,
        longstay_mintime=longstay_mintime,
        ehr_data_path=data_path,
    )
    test_dataset = MM_dataset(
        split="test",
        img_transform=img_transform,
        task=task,
        longstay_mintime=longstay_mintime,
        ehr_data_path=data_path,
    )
    val_dataset = MM_dataset(
        split="val",
        img_transform=img_transform,
        task=task,
        longstay_mintime=longstay_mintime,
        ehr_data_path=data_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=train_dataset.get_collate(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=test_dataset.get_collate()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.get_collate()
    )

    model = EHR_module(device=device)
    model = model.to(device)

    if use_ratio:
        ratio = (
            (train_dataset.label.sum() + val_dataset.label.sum())
            / (len(train_dataset) + len(val_dataset))
        ).item()
    else:
        ratio = None

    if best_test_only:
        best_test(model, task, device, test_loader, val_loader, ratio)
    else:
        train(
            model,
            task,
            device,
            train_loader,
            val_loader,
            test_loader,
            epochs,
            learning_rate,
            ratio,
            saved_model_path=saved_model_path,
            use_amp=use_amp,
        )
    # best_test(model, device, test_loader, val_loader, ratio)


def train_epoch(
    model, device, train_loader, optimizer, criterion, scaler=None, use_amp=False
):
    model.train()
    total_loss = []
    for batch_idx, ((input, ce_ts, le_ts, pe_ts, timestamps), _, _, label) in enumerate(
        train_loader
    ):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            demo = input.to(device)
            target = label.to(device)
            pred = model(demo, ce_ts, le_ts, pe_ts, timestamps)
            loss = criterion(pred, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # optimizer.step()

        loss_num = loss.data.item()
        total_loss.append(loss.data * len(target))
        if batch_idx % 50 == 0:
            print(f"batch [{batch_idx + 1}/{len(train_loader)}] loss: {loss_num:.3f}")

    avg_loss = torch.sum(torch.stack(total_loss)) / len(train_loader.dataset)
    return avg_loss


@torch.no_grad()
def val_epoch(model, device, val_loader, use_amp=False):
    all_targets = []
    all_preds = []
    for _, ((input, ce_ts, le_ts, pe_ts, timestamps), _, _, label) in enumerate(
        val_loader
    ):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            demo = input.to(device)
            target = label.to(device)
            pred = model(demo, ce_ts, le_ts, pe_ts, timestamps)
        all_targets.append(target)
        all_preds.append(pred.to("cpu"))
    all_targets = torch.cat(all_targets).to("cpu").float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_preds = torch.softmax(all_preds, dim=1)[:, 1].to("cpu").numpy()
    auroc = roc_auc_score(all_targets, all_preds)
    return auroc


@torch.no_grad()
def cal_threshold(model, device, test_loader, ratio=None, use_amp=False):
    model.eval()
    if ratio is None:
        return None
    all_preds = []
    for _, ((input, ce_ts, le_ts, pe_ts, timestamps), _, _, _) in enumerate(
        test_loader
    ):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            demo = input.to(device)
            # target = label.to(device)
            pred = model(demo, ce_ts, le_ts, pe_ts, timestamps)
        all_preds.append(pred.to("cpu"))
    all_preds = torch.cat(all_preds).float()
    all_probs = torch.softmax(all_preds, dim=1).to("cpu").numpy()
    pos_prob = all_probs[:, 1]
    neg_num = int(len(pos_prob) * (1 - ratio))
    partition = np.partition(pos_prob, neg_num)
    x1, x2 = np.max(partition[:neg_num]), partition[neg_num]
    return (x1 + x2) / 2


@torch.no_grad()
def test(model, task, device, test_loader, threshold=None, use_amp=False):
    model.eval()
    all_targets = []
    all_preds = []
    for _, ((input, ce_ts, le_ts, pe_ts, timestamps), _, _, label) in enumerate(
        test_loader
    ):
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            demo = input.to(device)
            target = label.to(device)
            pred = model(demo, ce_ts, le_ts, pe_ts, timestamps)
        all_targets.append(target)
        all_preds.append(pred.to("cpu"))
    all_targets = torch.cat(all_targets).to("cpu").float().numpy()
    all_preds = torch.cat(all_preds).float()
    all_probs = torch.softmax(all_preds, dim=1).to("cpu").numpy()
    if threshold is None:
        all_preds = np.argmax(all_probs, axis=1)
    else:
        all_preds = (all_probs[:, 1] >= threshold).astype("int")
    all_probs = all_probs[:, 1]
    auroc = roc_auc_score(all_targets, all_probs)
    precision, recall, t = precision_recall_curve(all_targets, all_probs)
    auprc = auc(recall, precision)
    ap = average_precision_score(all_targets, all_probs)
    report = classification_report(
        all_targets, all_preds, target_names=["negative", "positive"]
    )
    positive_num = all_preds.sum()
    return auroc, ap, auprc, report, positive_num


@torch.no_grad()
def best_test(
    model, task, device, test_loader, val_loader=None, ratio=None, saved_model_path=None
):
    model.load_state_dict(
        torch.load(os.path.join(saved_model_path, f"best_ehr_partial_model_{task}.pth"))
    )
    if val_loader is not None:
        threshold = cal_threshold(model, device, val_loader, ratio)
    else:
        threshold = None
    auroc, ap, auprc, report, positive_num = test(model, device, test_loader, threshold)
    print(f"test metric -- auroc:{auroc:.3f}")
    print(f"test metric -- ap:{ap:.3f}")
    print(f"test metric -- auprc:{auprc:.3f}")
    print(f"test metric -- predicted positive:{positive_num}")
    print(f"test metric -- report:\n{report}")
    return auroc, ap, auprc, report, positive_num


def train(
    model,
    task,
    device,
    train_loader,
    val_loader,
    test_loader,
    epoch,
    learning_rate,
    ratio=None,
    saved_model_path=None,
    use_amp=None,
):
    best_roc = 0

    if task == "mortality":
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1, 10], dtype=torch.float)
        ).to(device)
    elif task == "longstay":
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1, 1], dtype=torch.float)
        ).to(device)
    elif task == "readmission":
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1, 20], dtype=torch.float)
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler() if use_amp else None

    for epoch_idx in tqdm(range(epoch)):
        print(f"Epoch [{epoch_idx + 1}/{epoch}] ")
        epoch_loss = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            scaler=scaler,
            use_amp=use_amp,
        )
        torch.cuda.empty_cache()
        print(f"Epoch [{epoch_idx + 1}/{epoch}] loss:{epoch_loss:.3f}")

        auroc = val_epoch(model, device, val_loader, use_amp=use_amp)
        torch.cuda.empty_cache()

        if auroc > best_roc:
            print(f"new best auroc: {best_roc} -> {auroc}")
            best_roc = auroc
            print("model saved.")

            if not os.path.exists(saved_model_path):
                print(
                    f"Creating directory to save best model epoch: {saved_model_path}"
                )
                Path(saved_model_path).mkdir(parents=True)

            torch.save(
                model.state_dict(),
                os.path.join(saved_model_path, f"best_ehr_partial_model_{task}.pth"),
            )

    # model.load_state_dict(torch.load('./saved_model/best_cxr_model.pth'))
    # auroc, report, positive_num = test(model, device, test_loader)
    # torch.cuda.empty_cache()
    # print('test metric -- auroc:{:.3f}'.format(auroc))
    # print('test metric -- predicted positive:{}'.format(positive_num))
    # print('test metric -- report:\n{}'.format(report))
    # best_test(model, device, test_loader, val_loader, ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)  # required positional argument
    parser.add_argument("saved_model_path", type=str, default="./saved_models")
    args = parser.parse_args()

    main(args)
