import numpy as np
import matplotlib.pyplot as plt

class MlIoImageWriter:

    def __init__(self, validator, locator, executor):
        self.file_validator = validator
        self.executableLocator = locator
        self.processExecutor = executor

    def create_jpg_from_dot(self, input_file, out_file):
        dot_exe = self.executableLocator.get_dot_executable()
        self.file_validator.validate_file_readable(input_file)
        self.file_validator.validate_file_executable(dot_exe)
        self.file_validator.validate_file_writable(out_file)
        self.processExecutor.run([str(dot_exe), "-Tjpg",input_file, "-o", out_file], check=True)

    @staticmethod
    def create_plot(out_png, cm):
        fig, ax = plt.subplots(figsize=(4, 4))

        im = ax.imshow(cm, cmap='Greens')

        # Define label names in correct order:
        # True positives (TP) and true negatives (TN) are both accurate classifications in binary testing or modeling.
        # Where the prediction matches the actual reality
        # TN(True Negative): The model correctly predicts the negative
        # FP(False Positive)
        # FN(False Negative)
        # TP(True Positive):  The model correctly predicts the positive
        labels = np.array([["TN", "FP"], ["FN", "TP"]])

        # Add text annotations
        for (i, j), value in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{value}",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold'
            )
        ax.set_title('Confusion Matrix — Decision Tree')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Optional: tick labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
