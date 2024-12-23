import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch


def plot_training_metrics(
    checkpoint_path: str, net_name: str, output_dir: str = "src/data/images"
):
    """
    Generate a plot for training and validation loss, and F1 metric from a given checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        net_name (str): Name of the neural network (used for the plot title and filename).
        output_dir (str): Directory where the plot will be saved.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)  # type: ignore

    losses_val = checkpoint["val_loss"]
    losses_train = checkpoint["train_loss"]
    f1_metric_val = checkpoint["f1_metric_val"]

    # Identify the marker for the minimum validation loss
    marker_on = [losses_val.index(min(losses_val))]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 7))  # type: ignore

    # Plot train and validation loss on the left y-axis
    ax1.plot(  # type: ignore
        np.arange(1, len(losses_train) + 1),  # type: ignore
        losses_train,
        color="r",
        label="Train loss",
    )
    ax1.plot(  # type: ignore
        np.arange(1, len(losses_val) + 1),  # type: ignore
        losses_val,
        "-gD",
        label="Test loss",
        markevery=marker_on,
    )

    # Annotate the minimum loss point
    bbox = dict(boxstyle="round", fc="0.8")
    ax1.annotate(  # type: ignore
        text=f"Max F1 score after {f1_metric_val.index(max(f1_metric_val))+1} epochs, equals: {max(f1_metric_val)}",
        xy=(losses_val.index(min(losses_val)) + 1, min(losses_val)),
        xytext=(losses_val.index(min(losses_val)) + 1, min(losses_val) + 0.05),
        arrowprops=dict(facecolor="green", shrink=0.2),
        bbox=bbox,
    )

    ax1.set_xlabel("Epoch")  # type: ignore
    ax1.set_ylabel("CrossEntropyLoss")  # type: ignore
    ax1.grid()  # type: ignore

    # Create the second y-axis for F1 metric
    ax2 = ax1.twinx()
    ax2.plot(  # type: ignore
        np.arange(1, len(f1_metric_val) + 1),  # type: ignore
        f1_metric_val,
        color="blue",
        label="F1 Metric (Val)",
    )
    ax2.set_ylabel("f1_metric_val")  # type: ignore

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()  # type: ignore
    lines2, labels2 = ax2.get_legend_handles_labels()  # type: ignore
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # type: ignore

    # Set title and save the plot
    plt.title(f"Train and Val loss, F1 Metric after {len(losses_train)} epochs")  # type: ignore
    output_path = f"{output_dir}/{net_name}.png"
    plt.savefig(output_path)  # type: ignore

    print(f"Plot saved to {output_path}")
