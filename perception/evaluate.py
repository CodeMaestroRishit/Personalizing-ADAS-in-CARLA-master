"""Evaluate CNN and SNN checkpoints, latency, sparsity, and energy proxies."""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.data import PerceptionDataset
from perception.energy import count_cnn_macs, count_parameters, count_synaptic_operations, snn_fan_out
from perception.models import ConvolutionalBaseline, SpikingConvNet


PICOJOULES_PER_MAC = 4.6
PICOJOULES_PER_ACCUMULATE = 0.9


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--data-root", default="data/perception")
    argument_parser.add_argument("--checkpoint-directory", default="perception/checkpoints")
    argument_parser.add_argument("--output-directory", default="results")
    argument_parser.add_argument("--batch-size", type=int, default=1,
                                 help="Keep at 1 for per-frame latency measurements")
    argument_parser.add_argument("--time-steps", default="1,2,4,8,16")
    argument_parser.add_argument("--warmup-runs", type=int, default=10)
    argument_parser.add_argument("--max-test-samples", type=int, default=0,
                                 help="0 evaluates every test frame")
    argument_parser.add_argument("--seed", type=int, default=1337)
    return argument_parser.parse_args()


def build_model(model_name, checkpoint):
    if model_name == "snn":
        model = SpikingConvNet(
            beta=checkpoint.get("beta", 0.9),
            threshold=checkpoint.get("threshold", 1.0),
            num_time_steps=checkpoint.get("num_time_steps", 8),
            encoding=checkpoint.get("encoding", "rate"),
        )
    else:
        model = ConvolutionalBaseline()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def confusion_and_metrics(actual_labels, predicted_labels):
    true_negative = sum(actual == 0 and predicted == 0 for actual, predicted in zip(actual_labels, predicted_labels))
    false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in zip(actual_labels, predicted_labels))
    false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in zip(actual_labels, predicted_labels))
    true_positive = sum(actual == 1 and predicted == 1 for actual, predicted in zip(actual_labels, predicted_labels))
    accuracy = (true_positive + true_negative) / float(max(1, len(actual_labels)))
    precision = true_positive / float(max(1, true_positive + false_positive))
    recall = true_positive / float(max(1, true_positive + false_negative))
    f1_score = 2.0 * precision * recall / float(max(1e-12, precision + recall))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "confusion_matrix": "[[{}, {}], [{}, {}]]".format(
            true_negative, false_positive, false_negative, true_positive
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def inference_latency_ms(model, input_frame, device, model_name, num_time_steps):
    input_frame = input_frame.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    with torch.no_grad():
        if model_name == "snn":
            model(input_frame, num_time_steps=num_time_steps)
        else:
            model(input_frame)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started_at) * 1000.0


def evaluate_configuration(model, model_name, loader, device, num_time_steps,
                           warmup_runs, max_test_samples):
    model.to(device)
    model.eval()
    for warmup_index, (input_frames, _) in enumerate(loader):
        if warmup_index >= warmup_runs:
            break
        inference_latency_ms(model, input_frames, device, model_name, num_time_steps)

    actual_labels = []
    predicted_labels = []
    latency_samples = []
    total_spike_counts = {"conv1_lif": 0.0, "conv2_lif": 0.0, "linear_lif": 0.0}
    processed_sample_count = 0
    with torch.no_grad():
        for batch_of_input_frames, class_labels in loader:
            for frame_index in range(batch_of_input_frames.size(0)):
                input_frame = batch_of_input_frames[frame_index:frame_index + 1]
                class_label = int(class_labels[frame_index].item())
                latency_samples.append(inference_latency_ms(
                    model, input_frame, device, model_name, num_time_steps
                ))
                input_frame = input_frame.to(device)
                if model_name == "snn":
                    predictions, spike_recording_per_layer = model(
                        input_frame, num_time_steps=num_time_steps,
                        return_spike_recording=True
                    )
                    for layer_name, spike_count in spike_recording_per_layer.items():
                        total_spike_counts[layer_name] += spike_count.item()
                else:
                    predictions = model(input_frame)
                actual_labels.append(class_label)
                predicted_labels.append(int(predictions.argmax(dim=1).item()))
                processed_sample_count += 1
                if max_test_samples and processed_sample_count >= max_test_samples:
                    break
            if max_test_samples and processed_sample_count >= max_test_samples:
                break

    statistics = confusion_and_metrics(actual_labels, predicted_labels)
    statistics["mean_latency_ms"] = float(np.mean(latency_samples)) if latency_samples else 0.0
    statistics["p95_latency_ms"] = float(np.percentile(latency_samples, 95)) if latency_samples else 0.0
    statistics["sample_count"] = processed_sample_count
    statistics["energy_status"] = "estimated, not measured on hardware"
    statistics["energy_reference_node_nm"] = 45
    statistics["mac_energy_pj_per_mac"] = PICOJOULES_PER_MAC
    statistics["synop_energy_pj_per_accumulate"] = PICOJOULES_PER_ACCUMULATE
    if model_name == "snn":
        fan_out_per_layer = snn_fan_out(model)
        statistics["synops_per_frame"] = count_synaptic_operations(
            total_spike_counts, fan_out_per_layer,
            batch_size=max(1, processed_sample_count), num_time_steps=1
        )
        total_neuron_slots_per_layer = {
            "conv1_lif": processed_sample_count * num_time_steps * 16 * 64 * 64,
            "conv2_lif": processed_sample_count * num_time_steps * 32 * 32 * 32,
            "linear_lif": processed_sample_count * num_time_steps * 2,
        }
        statistics["conv1_lif_spike_sparsity"] = 1.0 - total_spike_counts["conv1_lif"] / max(1, total_neuron_slots_per_layer["conv1_lif"])
        statistics["conv2_lif_spike_sparsity"] = 1.0 - total_spike_counts["conv2_lif"] / max(1, total_neuron_slots_per_layer["conv2_lif"])
        statistics["linear_lif_spike_sparsity"] = 1.0 - total_spike_counts["linear_lif"] / max(1, total_neuron_slots_per_layer["linear_lif"])
        statistics["estimated_energy_pj"] = statistics["synops_per_frame"] * PICOJOULES_PER_ACCUMULATE
    else:
        statistics["macs_per_frame"] = count_cnn_macs(model)
        statistics["estimated_energy_pj"] = statistics["macs_per_frame"] * PICOJOULES_PER_MAC
    statistics["parameter_count"] = count_parameters(model)
    return statistics


def main():
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(arguments.output_directory, exist_ok=True)
    test_dataset = PerceptionDataset(os.path.join(arguments.data_root, "test"))
    test_loader = DataLoader(test_dataset, batch_size=arguments.batch_size, shuffle=False, num_workers=0)
    time_steps_to_evaluate = [int(value.strip()) for value in arguments.time_steps.split(",") if value.strip()]
    if not time_steps_to_evaluate or any(time_step < 1 for time_step in time_steps_to_evaluate):
        raise ValueError("--time-steps must contain positive comma-separated integers")
    available_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        available_devices.append(torch.device("cuda"))
    all_result_rows = []
    accuracy_by_time_step = {"snn": [], "cnn": None}

    for model_name in ("snn", "cnn"):
        checkpoint_path = os.path.join(arguments.checkpoint_directory, "{}.pt".format(model_name))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_model(model_name, checkpoint)
        relevant_time_steps = time_steps_to_evaluate if model_name == "snn" else [1]
        for time_step_count in relevant_time_steps:
            cpu_statistics = evaluate_configuration(
                model, model_name, test_loader, torch.device("cpu"), time_step_count,
                arguments.warmup_runs, arguments.max_test_samples
            )
            accuracy_by_time_step[model_name] = cpu_statistics["accuracy"] if model_name == "cnn" else accuracy_by_time_step[model_name]
            result_row = {
                "model": model_name,
                "time_steps": time_step_count if model_name == "snn" else "static",
                "device": "cpu",
                **cpu_statistics,
            }
            all_result_rows.append(result_row)
            if model_name == "snn":
                accuracy_by_time_step["snn"].append((time_step_count, cpu_statistics["accuracy"]))
            if len(available_devices) > 1:
                gpu_statistics = evaluate_configuration(
                    model, model_name, test_loader, torch.device("cuda"), time_step_count,
                    arguments.warmup_runs, arguments.max_test_samples
                )
                all_result_rows.append({
                    "model": model_name,
                    "time_steps": time_step_count if model_name == "snn" else "static",
                    "device": "cuda",
                    **gpu_statistics,
                })

    csv_path = os.path.join(arguments.output_directory, "comparison.csv")
    column_names = sorted(set().union(*(result_row.keys() for result_row in all_result_rows)))
    with open(csv_path, "w", newline="") as result_file:
        result_writer = csv.DictWriter(result_file, fieldnames=column_names)
        result_writer.writeheader()
        result_writer.writerows(all_result_rows)

    cpu_rows = [row for row in all_result_rows if row["device"] == "cpu"]
    plt.figure()
    for model_name in ("cnn", "snn"):
        model_rows = [row for row in cpu_rows if row["model"] == model_name]
        plt.scatter(
            [row["estimated_energy_pj"] for row in model_rows],
            [row["accuracy"] for row in model_rows], label=model_name.upper()
        )
        for row in model_rows:
            label = "{} steps".format(row["time_steps"]) if model_name == "snn" else "CNN"
            plt.annotate(label, (row["estimated_energy_pj"], row["accuracy"]))
    plt.xlabel("Estimated energy per frame (pJ; not measured on hardware)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(arguments.output_directory, "accuracy_vs_estimated_energy.png"))
    plt.close()

    snn_points = sorted(accuracy_by_time_step["snn"])
    cnn_accuracy = accuracy_by_time_step["cnn"]
    plt.figure()
    plt.plot([point[0] for point in snn_points], [point[1] for point in snn_points], marker="o", label="SNN")
    if cnn_accuracy is not None:
        plt.axhline(cnn_accuracy, linestyle="--", label="CNN baseline")
    plt.xlabel("SNN simulation time steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(arguments.output_directory, "accuracy_vs_time_steps.png"))
    plt.close()
    print("Wrote {}".format(csv_path))
    print("Energy values are estimates, not measured on hardware; proxy constants: {:.1f} pJ/MAC and {:.1f} pJ/accumulate (45 nm reference figures).".format(
        PICOJOULES_PER_MAC, PICOJOULES_PER_ACCUMULATE
    ))


if __name__ == "__main__":
    main()
