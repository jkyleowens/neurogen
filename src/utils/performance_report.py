import os
import matplotlib.pyplot as plt

def generate_performance_report(train_loss, val_loss, test_loss, output_dir="docs/"):
    """
    Generate a performance report including plots and a summary file.

    Args:
        train_loss (float): Final training loss.
        val_loss (float): Final validation loss.
        test_loss (float): Final testing loss.
        output_dir (str): Directory to save the report and plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a summary file
    summary_path = os.path.join(output_dir, "performance_report.md")
    with open(summary_path, "w") as f:
        f.write("# Performance Report\n\n")
        f.write(f"**Training Loss:** {train_loss:.4f}\n\n")
        f.write(f"**Validation Loss:** {val_loss:.4f}\n\n")
        f.write(f"**Testing Loss:** {test_loss:.4f}\n\n")

    print(f"Performance summary saved to {summary_path}")

    # Generate a plot for losses
    plt.figure(figsize=(8, 6))
    plt.bar(["Train", "Validation", "Test"], [train_loss, val_loss, test_loss], color=["blue", "orange", "green"])
    plt.title("Model Performance")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "performance_plot.png"))
    plt.close()

    print(f"Performance plot saved to {output_dir}/performance_plot.png")
