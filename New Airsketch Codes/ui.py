import cv2
import numpy as np
import math
import time
import os

class UI:
    def __init__(self, painter):
        self.painter = painter

        # ------------------------------------------
        # TOOLBAR
        # ------------------------------------------

        self.toolbar_height = 82

        # ------------------------------------------
        # COLORS
        # ------------------------------------------

        self.color_start_x = 185
        self.color_y = 41
        self.color_radius = 11
        self.color_spacing = 29

        # ------------------------------------------
        # TOOLS
        # ------------------------------------------

        self.tool_start_x = 560
        self.tool_y = 12
        self.tool_width = 48
        self.tool_height = 58
        self.tool_spacing = 6

        # ------------------------------------------
        # SIZE
        # ------------------------------------------

        self.size_x = 1045

        # ------------------------------------------
        # ICONS
        # ------------------------------------------

        self.icon_dir = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "icons"
        )

        self.icons = {}

        self.load_icons()

    def load_icons(self):

      icon_files = {
          "brush": "brush.png",
          "eraser": "eraser.png",
          "rectangle": "rectangle.png",
          "circle": "circle.png",
          "line": "line.png",
          "filled_rectangle": "filled_rectangle.png",
          "filled_circle": "filled_circle.png",
          "clear_canvas": "clear.png"
      }

      for tool, filename in icon_files.items():

          path = os.path.join(
              self.icon_dir,
              filename
          )

          icon = cv2.imread(
              path,
              cv2.IMREAD_UNCHANGED
          )

          if icon is None:
              print(f"Warning: Could not load {path}")
              continue

          self.icons[tool] = icon

    def draw_icon(self, frame, icon, x, y, size=30):

      if icon is None:
          return

      # Resize icon
      icon = cv2.resize(
          icon,
          (size, size),
          interpolation=cv2.INTER_AREA
      )

      h, w = icon.shape[:2]

      # Center position
      x1 = int(x - w / 2)
      y1 = int(y - h / 2)

      x2 = x1 + w
      y2 = y1 + h

      # Make sure icon is inside the frame
      if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
          return

      # PNG with alpha channel
      if icon.shape[2] == 4:

          alpha = icon[:, :, 3] / 255.0

          for c in range(3):

              frame[y1:y2, x1:x2, c] = (
                  alpha * icon[:, :, c]
                  +
                  (1 - alpha)
                  * frame[y1:y2, x1:x2, c]
              ).astype(np.uint8)

      else:

          frame[
              y1:y2,
              x1:x2
          ] = icon

    def draw(self, frame):

      # ==========================================
      # GLASS TOOLBAR
      # ==========================================

      overlay = frame.copy()

      cv2.rectangle(
          overlay,
          (0, 0),
          (1280, self.toolbar_height),
          (20, 22, 28),
          -1
      )

      # Transparency
      frame = cv2.addWeighted(
          overlay,
          0.55,
          frame,
          0.45,
          0
      )

      # Subtle bottom border
      cv2.line(
          frame,
          (0, self.toolbar_height),
          (1280, self.toolbar_height),
          (90, 90, 100),
          1
      )

      # ==========================================
      # TITLE
      # ==========================================

      cv2.putText(
          frame,
          "AIRSKETCH",
          (20, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.75,
          (245, 245, 250),
          2,
          cv2.LINE_AA
      )

      # ==========================================
      # COLOR PALETTE
      # ==========================================

      for i, color in enumerate(
          self.painter.colors
      ):

          cx = (
              self.color_start_x
              + i * self.color_spacing
          )

          # Selected ring
          if color == self.painter.current_color:

              cv2.circle(
                  frame,
                  (cx, self.color_y),
                  self.color_radius + 1,
                  (255, 255, 255),
                  2,
                  cv2.LINE_AA
              )

          cv2.circle(
              frame,
              (cx, self.color_y),
              self.color_radius,
              color,
              -1,
              cv2.LINE_AA
          )

      # ==========================================
      # TOOL BUTTONS
      # ==========================================

      for i, tool in enumerate(
          self.painter.tools
      ):

          x = (
              self.tool_start_x
              + i * (
                  self.tool_width
                  + self.tool_spacing
              )
          )

          y = self.tool_y

          # Selected tool
          if tool == self.painter.current_tool:

              cv2.rectangle(
                  frame,
                  (x, y),
                  (
                      x + self.tool_width,
                      y + self.tool_height
                  ),
                  (70, 100, 90),
                  -1
              )

              border = (130, 220, 160)

          else:

              # Mostly transparent button
              button_overlay = frame.copy()

              cv2.rectangle(
                  button_overlay,
                  (x, y),
                  (
                      x + self.tool_width,
                      y + self.tool_height
                  ),
                  (40, 42, 48),
                  -1
              )

              frame = cv2.addWeighted(
                  button_overlay,
                  0.45,
                  frame,
                  0.55,
                  0
              )

              border = (90, 90, 100)

          # Border
          cv2.rectangle(
              frame,
              (x, y),
              (
                  x + self.tool_width,
                  y + self.tool_height
              ),
              border,
              1,
              cv2.LINE_AA
          )

          # Icon
          icon = self.icons.get(tool)

          self.draw_icon(
              frame,
              icon,
              x + self.tool_width // 2,
              y + self.tool_height // 2,
              30
          )

      # ==========================================
      # SIZE SECTION
      # ==========================================

      # Vertical divider after the tools
      divider_x = self.tool_start_x + (
          len(self.painter.tools)
          * (self.tool_width + self.tool_spacing)
      ) + 12

      # SAME COLOR AS TOOL BUTTON BORDER
      divider_color = (43, 45, 51)

      cv2.line(
          frame,
          (divider_x, 0),
          (divider_x, 80),
          divider_color,
          3,
          cv2.LINE_AA
      )

      # ------------------------------------------
      # SIZE LAYOUT
      # ------------------------------------------

      size_x = divider_x + 20

      # Give the size section more horizontal space
      size_width = 250

      # Center everything vertically in toolbar
      center_y = 41

      # ------------------------------------------
      # SIZE LABEL
      # ------------------------------------------

      cv2.putText(
          frame,
          "SIZE",
          (size_x, center_y + 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.70,
          (245, 245, 250),
          1,
          cv2.LINE_AA
      )

      # ------------------------------------------
      # MINUS BUTTON
      # ------------------------------------------

      minus_x = size_x + 70

      button_width = 38
      button_height = 38

      button_y = center_y - button_height // 2

      cv2.rectangle(
          frame,
          (minus_x, button_y),
          (
              minus_x + button_width,
              button_y + button_height
          ),
          (45, 47, 54),
          -1,
          cv2.LINE_AA
      )

      cv2.rectangle(
          frame,
          (minus_x, button_y),
          (
              minus_x + button_width,
              button_y + button_height
          ),
          divider_color,
          1,
          cv2.LINE_AA
      )

      # Center "-"
      minus_text = "-"

      (text_w, text_h), _ = cv2.getTextSize(
          minus_text,
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          2
      )

      cv2.putText(
          frame,
          minus_text,
          (
              minus_x + (button_width - text_w) // 2,
              button_y + (button_height + text_h) // 2 - 2
          ),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (240, 240, 245),
          2,
          cv2.LINE_AA
      )

      # ------------------------------------------
      # CURRENT SIZE
      # ------------------------------------------

      size_text = f"{self.painter.brush_thickness}px"

      cv2.putText(
          frame,
          size_text,
          (
              minus_x + button_width + 10,
              center_y + 5
          ),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.60,
          (245, 245, 250),
          1,
          cv2.LINE_AA
      )

      # ------------------------------------------
      # PLUS BUTTON
      # ------------------------------------------

      plus_x = minus_x + button_width + 78

      cv2.rectangle(
          frame,
          (plus_x, button_y),
          (
              plus_x + button_width,
              button_y + button_height
          ),
          (45, 47, 54),
          -1,
          cv2.LINE_AA
      )

      cv2.rectangle(
          frame,
          (plus_x, button_y),
          (
              plus_x + button_width,
              button_y + button_height
          ),
          divider_color,
          1,
          cv2.LINE_AA
      )

      # Center "+"
      plus_text = "+"

      (text_w, text_h), _ = cv2.getTextSize(
          plus_text,
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          2
      )

      cv2.putText(
          frame,
          plus_text,
          (
              plus_x + (button_width - text_w) // 2,
              button_y + (button_height + text_h) // 2 - 2
          ),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (240, 240, 245),
          2,
          cv2.LINE_AA
      )

      return frame

    def handle_interaction(self, x, y):

      if (
          time.time()
          - self.painter.last_ui_interaction_time
          < 0.3
      ):
          return

      self.painter.last_ui_interaction_time = time.time()

      # ==========================================
      # COLORS
      # ==========================================

      for i, color in enumerate(
          self.painter.colors
      ):

          cx = (
              self.color_start_x
              + i * self.color_spacing
          )

          distance = math.hypot(
              x - cx,
              y - self.color_y
          )

          if distance <= self.color_radius + 10:

              self.painter.current_color = color
              return

      # ==========================================
      # TOOLS
      # ==========================================

      for i, tool in enumerate(
          self.painter.tools
      ):

          x_start = (
              self.tool_start_x
              + i * (
                  self.tool_width
                  + self.tool_spacing
              )
          )

          if (
              x_start < x
              < x_start + self.tool_width
              and
              self.tool_y < y
              < self.tool_y + self.tool_height
          ):

              if tool == "clear_canvas":

                self.painter.canvas = np.zeros(
                    (720, 1280, 3),
                    dtype=np.uint8
                )

                self.painter.drawing = False
                self.painter.prev_x = 0
                self.painter.prev_y = 0

                self.painter.current_tool = tool

                return

              else:

                  self.painter.current_tool = tool

              return

      # ==========================================
      # SIZE SECTION
      # ==========================================

      divider_x = self.tool_start_x + (
          len(self.painter.tools)
          * (self.tool_width + self.tool_spacing)
      ) + 12

      size_x = divider_x + 20

      center_y = 41

      # ------------------------------------------
      # MINUS BUTTON
      # ------------------------------------------

      minus_x = size_x + 70

      button_width = 38
      button_height = 38

      button_y = center_y - button_height // 2

      if (
          minus_x < x < minus_x + button_width
          and
          button_y < y < button_y + button_height
      ):

          self.painter.brush_size_index = max(
              0,
              self.painter.brush_size_index - 1
          )

          self.painter.brush_thickness = (
              self.painter.brush_size[
                  self.painter.brush_size_index
              ]
          )

          return

      # ------------------------------------------
      # PLUS BUTTON
      # ------------------------------------------

      plus_x = (
          minus_x
          + button_width
          + 78
      )

      if (
          plus_x < x < plus_x + button_width
          and
          button_y < y < button_y + button_height
      ):

          self.painter.brush_size_index = min(
              len(self.painter.brush_size) - 1,
              self.painter.brush_size_index + 1
          )

          self.painter.brush_thickness = (
              self.painter.brush_size[
                  self.painter.brush_size_index
              ]
          )

          return