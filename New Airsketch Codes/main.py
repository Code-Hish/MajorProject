import cv2
from virtual_painter import VirtualPainter


def print_metrics(painter):
    metrics = painter.evaluator.metrics()

    print("\n" + "=" * 50)
    print("AIRSKETCH PINCH DETECTION EVALUATION")
    print("=" * 50)

    print("\nConfusion Matrix")
    print("                 Predicted")
    print("              No Pinch  Pinch")
    print(
        f"Actual No Pinch   {metrics['TN']:6d}  {metrics['FP']:5d}"
    )
    print(
        f"       Pinch      {metrics['FN']:6d}  {metrics['TP']:5d}"
    )

    print("\nMetrics")
    print(f"Samples   : {metrics['samples']}")
    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"F1-score  : {metrics['f1'] * 100:.2f}%")
    print(f"FPR       : {metrics['fpr'] * 100:.2f}%")
    print(f"FNR       : {metrics['fnr'] * 100:.2f}%")
    print("\nDetailed samples: evaluation_results.csv")
    print("Text report     : evaluation_report.txt")
    print("=" * 50)


def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    painter = VirtualPainter()

    print("\nAirSketch controls:")
    print("  E = toggle evaluation mode")
    print("  1 = actual gesture is PINCH")
    print("  0 = actual gesture is NO PINCH")
    print("  R = reset evaluation counters")
    print("  S = save evaluation report")
    print("  Q = quit")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            final_frame = painter.process_frame(frame)

            cv2.imshow(painter.window_name, final_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("e"):
                painter.evaluation_mode = not painter.evaluation_mode
                painter.evaluator.clear_ground_truth()

                print(
                    "\nEvaluation mode:",
                    "ON" if painter.evaluation_mode else "OFF"
                )

                if painter.evaluation_mode:
                    print(
                        "Press 1 while you are actually pinching, "
                        "or 0 while you are actually not pinching."
                    )

            elif key == ord("1") and painter.evaluation_mode:
                painter.evaluator.set_ground_truth(1)
                print("Ground truth set to PINCH.")

            elif key == ord("0") and painter.evaluation_mode:
                painter.evaluator.set_ground_truth(0)
                print("Ground truth set to NO PINCH.")

            elif key == ord("r") and painter.evaluation_mode:
                painter.evaluator.reset()
                print("Evaluation counters reset.")

            elif key == ord("s") and painter.evaluation_mode:
                metrics = painter.evaluator.save_report()
                print_metrics(painter)
                print("Evaluation report saved.")

            elif key == ord("q"):
                break

    finally:
        if painter.evaluator.samples > 0:
            painter.evaluator.save_report()
            print_metrics(painter)

        painter.evaluator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
