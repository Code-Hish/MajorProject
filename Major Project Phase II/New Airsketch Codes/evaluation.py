import csv
import os
from datetime import datetime


class GestureEvaluator:
    """
    Frame-level evaluator for AirSketch pinch detection.

    Ground truth:
        1 = the user is actually pinching
        0 = the user is actually not pinching
    Prediction:
        1 = AirSketch pinch detector says pinch
        0 = AirSketch pinch detector says no pinch
    """

    def __init__(self, csv_path="evaluation_results.csv", sample_every=3):
        self.csv_path = csv_path
        self.sample_every = max(1, int(sample_every))
        self.frame_counter = 0
        self.ground_truth = None

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.samples = 0

        # Keep a detailed record so the experiment is reproducible.
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp",
            "actual",
            "predicted",
            "distance_px"
        ])

    def set_ground_truth(self, label):
        if label in (0, 1):
            self.ground_truth = int(label)

    def clear_ground_truth(self):
        self.ground_truth = None

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0
        self.samples = 0
        self.frame_counter = 0

    def add_sample(self, predicted, distance_px=None, hand_detected=True):
        """
        Add one labeled frame.

        Samples are only collected when:
        - evaluation mode is active,
        - a ground-truth label has been selected,
        - a hand is detected,
        - the sampling interval is reached.
        """
        self.frame_counter += 1

        if self.ground_truth is None:
            return False

        if not hand_detected:
            return False

        if self.frame_counter % self.sample_every != 0:
            return False

        actual = self.ground_truth
        predicted = int(bool(predicted))

        if actual == 1 and predicted == 1:
            self.tp += 1
        elif actual == 0 and predicted == 0:
            self.tn += 1
        elif actual == 0 and predicted == 1:
            self.fp += 1
        elif actual == 1 and predicted == 0:
            self.fn += 1

        self.samples += 1

        self.writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            actual,
            predicted,
            "" if distance_px is None else round(float(distance_px), 2)
        ])
        self.file.flush()

        return True

    def metrics(self):
        total = self.tp + self.tn + self.fp + self.fn

        accuracy = (self.tp + self.tn) / total if total else 0.0
        precision = (
            self.tp / (self.tp + self.fp)
            if (self.tp + self.fp) else 0.0
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if (self.tp + self.fn) else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        fpr = (
            self.fp / (self.fp + self.tn)
            if (self.fp + self.tn) else 0.0
        )
        fnr = (
            self.fn / (self.fn + self.tp)
            if (self.fn + self.tp) else 0.0
        )

        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "samples": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
        }

    def save_report(self, report_path="evaluation_report.txt"):
        m = self.metrics()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("AirSketch Pinch Detection Evaluation\n")
            f.write("=" * 42 + "\n\n")
            f.write("Confusion Matrix\n")
            f.write("                 Predicted\n")
            f.write("              No Pinch  Pinch\n")
            f.write(f"Actual No Pinch   {m['TN']:6d}  {m['FP']:5d}\n")
            f.write(f"       Pinch      {m['FN']:6d}  {m['TP']:5d}\n\n")

            f.write(f"Samples   : {m['samples']}\n")
            f.write(f"Accuracy  : {m['accuracy']:.4f} ({m['accuracy'] * 100:.2f}%)\n")
            f.write(f"Precision : {m['precision']:.4f} ({m['precision'] * 100:.2f}%)\n")
            f.write(f"Recall    : {m['recall']:.4f} ({m['recall'] * 100:.2f}%)\n")
            f.write(f"F1-score  : {m['f1']:.4f} ({m['f1'] * 100:.2f}%)\n")
            f.write(f"FPR       : {m['fpr']:.4f} ({m['fpr'] * 100:.2f}%)\n")
            f.write(f"FNR       : {m['fnr']:.4f} ({m['fnr'] * 100:.2f}%)\n")

        return m

    def close(self):
        if not self.file.closed:
            self.file.close()
