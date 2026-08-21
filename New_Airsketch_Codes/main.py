import cv2
from virtual_painter import VirtualPainter


def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    painter = VirtualPainter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            final_frame = painter.process_frame(frame)

            cv2.imshow(painter.window_name, final_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()