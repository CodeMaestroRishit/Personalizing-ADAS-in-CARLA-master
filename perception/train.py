"""Train a CNN or SNN model from a collected split directory."""

import argparse
import csv
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from perception.data import PerceptionDataset
from perception.models import ConvolutionalBaseline, SpikingConvNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--model", choices=("snn", "cnn"), required=True)
    argument_parser.add_argument("--data-root", default="data/perception")
    argument_parser.add_argument("--epochs", type=int, default=20)
    argument_parser.add_argument("--batch-size", type=int, default=32)
    argument_parser.add_argument("--learning-rate", type=float, default=0.001)
    argument_parser.add_argument("--seed", type=int, default=1337)
    argument_parser.add_argument("--num-time-steps", type=int, default=8)
    argument_parser.add_argument("--encoding", choices=("rate", "latency"), default="rate")
    argument_parser.add_argument("--beta", type=float, default=0.9)
    argument_parser.add_argument("--threshold", type=float, default=1.0)
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--checkpoint-directory", default="perception/checkpoints")
    argument_parser.add_argument("--log-directory", default="perception/logs")
    return argument_parser.parse_args()


def main():
    arguments = parse_arguments()
    set_seed(arguments.seed)
    device = torch.device(
        "cuda" if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    training_dataset = PerceptionDataset(os.path.join(arguments.data_root, "train"))
    validation_dataset = PerceptionDataset(os.path.join(arguments.data_root, "val"))
    training_loader = DataLoader(
        training_dataset, batch_size=arguments.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda"
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=arguments.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda"
    )

    if arguments.model == "snn":
        model = SpikingConvNet(
            beta=arguments.beta, threshold=arguments.threshold,
            num_time_steps=arguments.num_time_steps, encoding=arguments.encoding
        )
    else:
        model = ConvolutionalBaseline()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    os.makedirs(arguments.checkpoint_directory, exist_ok=True)
    os.makedirs(arguments.log_directory, exist_ok=True)
    checkpoint_path = os.path.join(arguments.checkpoint_directory, "{}.pt".format(arguments.model))
    log_path = os.path.join(arguments.log_directory, "{}_training.jsonl".format(arguments.model))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    best_validation_accuracy = -1.0

    with open(log_path, "w") as log_file:
        for epoch_index in range(arguments.epochs):
            epoch_started_at = time.time()
            model.train()
            training_loss_sum = 0.0
            training_sample_count = 0
            for batch_of_input_frames, class_labels in training_loader:
                batch_of_input_frames = batch_of_input_frames.to(device)
                class_labels = class_labels.to(device)
                optimizer.zero_grad()
                predictions = model(batch_of_input_frames)
                loss = loss_function(predictions, class_labels)
                loss.backward()
                optimizer.step()
                training_loss_sum += loss.item() * class_labels.size(0)
                training_sample_count += class_labels.size(0)

            model.eval()
            correct_predictions = 0
            validation_sample_count = 0
            validation_loss_sum = 0.0
            with torch.no_grad():
                for batch_of_input_frames, class_labels in validation_loader:
                    batch_of_input_frames = batch_of_input_frames.to(device)
                    class_labels = class_labels.to(device)
                    predictions = model(batch_of_input_frames)
                    validation_loss_sum += loss_function(predictions, class_labels).item() * class_labels.size(0)
                    correct_predictions += (predictions.argmax(dim=1) == class_labels).sum().item()
                    validation_sample_count += class_labels.size(0)

            epoch_record = {
                "epoch": epoch_index + 1,
                "train_loss": training_loss_sum / max(1, training_sample_count),
                "validation_loss": validation_loss_sum / max(1, validation_sample_count),
                "validation_accuracy": correct_predictions / float(max(1, validation_sample_count)),
                "seconds": time.time() - epoch_started_at,
                "seed": arguments.seed,
            }
            log_file.write(json.dumps(epoch_record) + "\n")
            log_file.flush()
            logging.info("%s", epoch_record)

            if epoch_record["validation_accuracy"] >= best_validation_accuracy:
                best_validation_accuracy = epoch_record["validation_accuracy"]
                checkpoint = {
                    "model_name": arguments.model,
                    "model_state_dict": model.state_dict(),
                    "seed": arguments.seed,
                    "num_time_steps": arguments.num_time_steps,
                    "encoding": arguments.encoding,
                    "beta": arguments.beta,
                    "threshold": arguments.threshold,
                    "best_validation_accuracy": best_validation_accuracy,
                }
                torch.save(checkpoint, checkpoint_path)

    logging.info("Saved best checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
