import os
import matplotlib.pyplot as plt

def generate_performance_report(train_losses, val_losses, test_loss, metrics=None, output_dir="docs/"):
    """
    Generate a performance report including per-epoch stats, summary, metrics, and plots.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        test_loss (float): Final testing loss.
        metrics (dict, optional): Additional metrics e.g. {'mse':..,'mae':..,'r2':..}.
        output_dir (str): Directory to save the report and plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a summary file
    summary_path = os.path.join(output_dir, "performance_report.md")
    with open(summary_path, "w") as f:
        f.write("# Performance Report\n\n")
        # Per-epoch table
        if isinstance(train_losses, (list, tuple)):
            f.write("## Training & Validation Loss per Epoch\n\n")
            f.write("| Epoch | Train Loss | Val Loss |\n")
            f.write("|:-----:|:----------:|:--------:|\n")
            for i, (tr, vl) in enumerate(zip(train_losses, val_losses), start=1):
                f.write(f"| {i} | {tr:.6f} | {vl:.6f} |\n")
            f.write("\n")
        # Final test loss
        f.write("## Test Performance\n\n")
        f.write(f"**Test Loss:** {test_loss:.6f}\n\n")
        # Additional metrics
        if isinstance(metrics, dict):
            f.write("## Additional Metrics\n\n")
            for name, val in metrics.items():
                f.write(f"- **{name.upper()}:** {val:.6f}\n")
            f.write("\n")

    print(f"Performance summary saved to {summary_path}")

    # Plot training vs validation losses
    plt.figure(figsize=(8, 6))
    if isinstance(train_losses, (list, tuple)):
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, marker='o', label='Train Loss')
        plt.plot(epochs, val_losses, marker='o', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_epoch_plot.png"))
        plt.close()
        print(f"Epoch loss plot saved to {output_dir}/loss_epoch_plot.png")
    # Bar plot for final performance
    plt.figure(figsize=(6, 4))
    plt.bar(['Test'], [test_loss], color=['green'])
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, "test_loss_plot.png"))
    plt.close()
    print(f"Test loss plot saved to {output_dir}/test_loss_plot.png")
