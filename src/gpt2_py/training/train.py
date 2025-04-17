import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from aim import Run
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm, trange

from gpt2_py.data.dataloader import GPT2BookCorpusDataset
from gpt2_py.modeling.transformer import Transformer
from gpt2_py.utils.utils import AverageMeter
from transformers import GPT2TokenizerFast

def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()

def get_summary(meters: dict) -> str:
    return " ".join([f"{k}: {v.avg:5.7f}" for k, v in meters.items()])


def main(args: argparse.Namespace, config: dict) -> None:

    if not args.dry_run:
        run = Run(**config["logging"])
        run["hparams"] = config

        log_path = Path(config["logging"]["repo"])
        if not log_path.exists():
            os.makedirs(log_path.as_posix(), exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load data
    train_dataset = GPT2BookCorpusDataset(tokenizer=tokenizer, config=config["data"])
    val_dataset = GPT2BookCorpusDataset(tokenizer=tokenizer, config=config["data"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              num_workers=config["num_workers"], shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            num_workers=config["num_workers"], shuffle=False, pin_memory=True)

    # GPT-2-like Transformer
    model = Transformer(**config["model"]).to(device)

    if args.verbose:
        summary(model, input_size=(config["batch_size"], config["model"]["seq_len"]), depth=3)

    loss_fn = nn.CrossEntropyLoss(ignore_index=config["data"]["pad_idx"])
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    if config.get("weights_path"):
        weights = torch.load(config["weights_path"])
        model.load_state_dict(weights["model"])
        if config.get("recover_optim_state", False):
            optimizer.load_state_dict(weights["optimizer"])

    meters = {
        "t_loss": AverageMeter("t_loss", ":.4e"),
        "v_loss": AverageMeter("v_loss", ":.4e"),
        "t_acc": AverageMeter("t_acc", ":6.4f"),
        "v_acc": AverageMeter("v_acc", ":6.4f"),
    }

    best_val_loss = np.inf

    for i_epoch in trange(1, config["num_epochs"] + 1, desc="Epoch"):

        for meter in meters.values():
            meter.reset()

        model.train()
        for X, Y in tqdm(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            logits = logits.view(-1, logits.size(-1))
            Y = Y.view(-1)

            loss = loss_fn(logits, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc = accuracy(logits, Y)
            meters["t_loss"].update(loss.item())
            meters["t_acc"].update(acc.item())

        model.eval()
        with torch.no_grad():
            for X, Y in tqdm(val_loader):
                X = X.to(device)
                Y = Y.to(device)

                logits = model(X)
                logits = logits.view(-1, logits.size(-1))
                Y = Y.view(-1)

                loss = loss_fn(logits, Y)
                acc = accuracy(logits, Y)

                meters["v_loss"].update(loss.item())
                meters["v_acc"].update(acc.item())

        print(f"Epoch {i_epoch}/{config['num_epochs']} - {get_summary(meters)}")

        if not args.dry_run:
            run.track(meters["t_loss"].avg, "t_loss", epoch=i_epoch, context={"split": "train"})
            run.track(meters["t_acc"].avg, "t_acc", epoch=i_epoch, context={"split": "train"})
            run.track(meters["v_loss"].avg, "v_loss", epoch=i_epoch, context={"split": "val"})
            run.track(meters["v_acc"].avg, "v_acc", epoch=i_epoch, context={"split": "val"})

            if meters["v_loss"].avg < best_val_loss:
                best_val_loss = meters["v_loss"].avg
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "epoch": i_epoch,
                }, log_path / f"{run.hash}.pt")

    if not args.dry_run:
        run.report_successful_finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpt2_py/config/config.json", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    config = json.load(open(args.config))
    main(args, config)
