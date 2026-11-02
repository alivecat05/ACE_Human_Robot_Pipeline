#!/usr/bin/env python3
import json
import os
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
import qrcode
import time
import threading

class NanoClock(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nano Clock + QR Code")
        self.configure(bg='black')
        self.fullscreen = False
        self.toggle_fullscreen()
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.quit())
        self.protocol("WM_DELETE_WINDOW", self.quit)

        # ---- Center container ----
        center_frame = tk.Frame(self, bg="black")
        center_frame.pack(expand=True, fill="both")

        # Inner frame to keep time + QR vertically centered
        inner = tk.Frame(center_frame, bg="black")
        inner.pack(expand=True)

        # Time label
        self.label = tk.Label(
            inner,
            text="",
            fg="#00FF00",
            bg="black",
            font=("Consolas", 60, "bold")
        )
        self.label.pack(side="top", pady=20)

        # QR Code label
        self.qr_label = tk.Label(inner, bg="black")
        self.qr_label.pack(side="top", pady=10)

        self.last_qr_text = None
        self.stop_flag = threading.Event()

        # Color cycling like color_generate.py
        self.colors = [
            (0, 0, 255),     # Blue
            (0, 255, 0),     # Green
            (255, 0, 0),     # Red
            (255, 0, 255),   # Magenta
            (255, 255, 0),   # Yellow
        ]
        self.color_names = ["Blue", "Green", "Red", "Magenta", "Yellow"]
        self.current_color_index = 0
        self.color_interval = 1.0  # seconds per color
        self.last_color_change = time.time()
        self.color_changes = []

        self.worker = threading.Thread(target=self.update_loop, daemon=True)
        self.worker.start()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)

    def update_loop(self):
        while not self.stop_flag.is_set():
            now_ns = time.time_ns()
            sec, nano = divmod(now_ns, 1_000_000_000)
            tm = time.localtime(sec)

            text = f"{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{nano:09d}"

            # Cycle text/QR color on interval
            now = time.time()
            if now - self.last_color_change >= self.color_interval:
                color_name = self.color_names[self.current_color_index]
                self.color_changes.append({
                    "timestamp": now,
                    "color_index": self.current_color_index,
                    "color_name": color_name,
                })
                self.current_color_index = (self.current_color_index + 1) % len(self.colors)
                self.last_color_change = now

            fg_color = self._rgb_to_hex(self.colors[self.current_color_index])

            # Update clock text
            self.label.after(0, lambda t=text, c=fg_color: self.label.config(text=t, fg=c))

            # Update QR only on change
            if text != self.last_qr_text:
                self.last_qr_text = text
                self.label.after(0, lambda t=text, c=fg_color: self.update_qrcode(t, c))

            time.sleep(0.02)

    def update_qrcode(self, data: str, fg_color: str):
        qr = qrcode.QRCode(
            version=1,
            box_size=8,
            border=2
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color=fg_color, back_color="black")
        img = img.resize((640, 640), Image.NEAREST)

        self.qr_img_tk = ImageTk.PhotoImage(img)
        self.qr_label.config(image=self.qr_img_tk)

    @staticmethod
    def _rgb_to_hex(rgb):
        return '#{:02X}{:02X}{:02X}'.format(*rgb)

    def quit(self):
        self.stop_flag.set()
        if hasattr(self, "worker"):
            self.worker.join(timeout=1.0)
        self.save_log()
        super().quit()

    def save_log(self):
        filename = os.path.join(
            os.path.dirname(__file__),
            f"color_ts_sync_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.color_changes, f, indent=4)
        except OSError:
            pass

if __name__ == "__main__":
    NanoClock().mainloop()
