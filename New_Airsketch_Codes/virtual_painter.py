import cv2
import numpy as np
import math
import os

from hand_tracker import HandTracker
from ui import UI


class VirtualPainter:
    def __init__(self):

        # ==========================================
        # CANVAS
        # ==========================================

        self.canvas = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8
        )

        self.temp_canvas = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8
        )

        # ==========================================
        # COLORS
        # ==========================================

        self.colors = [
            (0, 0, 255),
            (0, 127, 255),
            (0, 255, 255),
            (110, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
            (150, 150, 150),
            (255, 255, 255),
            (57, 255, 20),
            (255, 20, 147),
            (0, 191, 255),
            (255, 255, 0)
        ]

        self.current_color = self.colors[0]

        # ==========================================
        # BRUSH SIZE
        # ==========================================

        self.brush_size = [
            5, 10, 15, 20, 25,
            30, 35, 40, 45, 50
        ]

        self.brush_size_index = 0

        self.brush_thickness = (
            self.brush_size[
                self.brush_size_index
            ]
        )

        # ==========================================
        # TOOLS
        # ==========================================

        self.tools = [
            "brush",
            "eraser",
            "rectangle",
            "circle",
            "line",
            "filled_rectangle",
            "filled_circle",
            "clear_canvas"
        ]

        self.current_tool = "brush"

        # ==========================================
        # OTHER SETTINGS
        # ==========================================

        self.save_dir = "paintings"

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

        self.window_name = "AirSketch"

        self.drawing = False

        self.prev_x = 0
        self.prev_y = 0

        self.start_x = 0
        self.start_y = 0

        self.last_ui_interaction_time = 0

        # ==========================================
        # HAND TRACKER
        # ==========================================

        self.hand_tracker = HandTracker()

        # ==========================================
        # UI
        # ==========================================

        self.ui = UI(self)

    def draw_brush(self, canvas, x, y):

        if self.prev_x == 0 and self.prev_y == 0:

            self.prev_x = x
            self.prev_y = y

        if self.current_tool == "brush":

            cv2.line(
                canvas,
                (self.prev_x, self.prev_y),
                (x, y),
                self.current_color,
                self.brush_thickness
            )

        elif self.current_tool == "eraser":

            cv2.line(
                canvas,
                (self.prev_x, self.prev_y),
                (x, y),
                (0, 0, 0),
                self.brush_thickness
            )

        self.prev_x = x
        self.prev_y = y

    def draw_shape(
        self,
        canvas,
        shape,
        x1,
        y1,
        x2,
        y2,
        color,
        thickness
    ):

        if shape == "rectangle":

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

        elif shape == "filled_rectangle":

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

        elif shape == "circle":

            radius = int(
                math.hypot(
                    x2 - x1,
                    y2 - y1
                )
            )

            cv2.circle(
                canvas,
                (x1, y1),
                radius,
                color,
                thickness
            )

        elif shape == "filled_circle":

            radius = int(
                math.hypot(
                    x2 - x1,
                    y2 - y1
                )
            )

            cv2.circle(
                canvas,
                (x1, y1),
                radius,
                color,
                -1
            )

        elif shape == "line":

            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

    def process_frame(self, frame):

        self.temp_canvas = self.canvas.copy()

        hand_data = self.hand_tracker.process(frame)

        if hand_data is not None:

            x1, y1, pinch = hand_data

            # ======================================
            # UI
            # ======================================

            if y1 < 90:

                self.ui.handle_interaction(
                    x1,
                    y1
                )

                self.drawing = False

            # ======================================
            # DRAWING
            # ======================================

            elif pinch:

                if not self.drawing:

                    self.drawing = True

                    self.prev_x = x1
                    self.prev_y = y1

                    self.start_x = x1
                    self.start_y = y1

                elif self.current_tool in [
                    "brush",
                    "eraser"
                ]:

                    self.draw_brush(
                        self.canvas,
                        x1,
                        y1
                    )

                else:

                    self.temp_canvas = (
                        self.canvas.copy()
                    )

                    self.draw_shape(
                        self.temp_canvas,
                        self.current_tool,
                        self.start_x,
                        self.start_y,
                        x1,
                        y1,
                        self.current_color,
                        self.brush_thickness
                    )

            # ======================================
            # RELEASE
            # ======================================

            else:

                if (
                    self.drawing
                    and self.current_tool in [
                        "rectangle",
                        "circle",
                        "line",
                        "filled_rectangle",
                        "filled_circle"
                    ]
                ):

                    self.draw_shape(
                        self.canvas,
                        self.current_tool,
                        self.start_x,
                        self.start_y,
                        x1,
                        y1,
                        self.current_color,
                        self.brush_thickness
                    )

                self.drawing = False

                self.prev_x = 0
                self.prev_y = 0

        # ==========================================
        # DISPLAY
        # ==========================================

        display = (
            self.temp_canvas
            if (
                self.drawing
                and self.current_tool in [
                    "rectangle",
                    "circle",
                    "line",
                    "filled_rectangle",
                    "filled_circle"
                ]
            )
            else self.canvas
        )

        mask = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2GRAY
        )

        _, mask = cv2.threshold(
            mask,
            5,
            255,
            cv2.THRESH_BINARY
        )

        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(
            frame,
            frame,
            mask=mask_inv
        )

        fg = cv2.bitwise_and(
            display,
            display,
            mask=mask
        )

        final = cv2.add(
            bg,
            fg
        )

        # ==========================================
        # DRAW UI
        # ==========================================

        final = self.ui.draw(final)

        # ==========================================
        # FINGERTIP CURSOR
        # ALWAYS DRAW LAST / ON TOP
        # ==========================================

        if hand_data is not None:

            preview_color = (
                (0, 0, 0)
                if self.current_tool == "eraser"
                else self.current_color
            )

            preview_radius = max(
                5,
                self.brush_thickness // 2
            )

            cv2.circle(
                final,
                (x1, y1),
                preview_radius,
                preview_color,
                2
            )

        return final