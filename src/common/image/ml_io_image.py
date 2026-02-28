import os

import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
from src.common.image.executable_locator import ExecutableLocator


class MlIoImageWriter:

    @staticmethod
    def create_jpg_from_dot(input_file, out_file):
        dot_exe = Path(ExecutableLocator.get_dot_executable())
        print('input_file={}'.format(input_file))
        print('output_file={}'.format(out_file))
        print('dot_exe_path={}'.format(dot_exe))

        if not os.path.isfile(input_file):
            raise ValueError("File {} does not exist".format(input_file))
        if not os.path.isfile(dot_exe):
            raise ValueError("File {} does not exist".format(dot_exe))

        subprocess.run([str(dot_exe), "-Tjpg",input_file, "-o", out_file], check=True)

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
