#!/usr/bin/env python3
import tkinter as tk
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

        threading.Thread(target=self.update_loop, daemon=True).start()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)

    def update_loop(self):
        while not self.stop_flag.is_set():
            now_ns = time.time_ns()
            sec, nano = divmod(now_ns, 1_000_000_000)
            tm = time.localtime(sec)

            text = f"{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{nano:09d}"

            # Update clock text
            self.label.after(0, lambda t=text: self.label.config(text=t))

            # Update QR only on change
            if text != self.last_qr_text:
                self.last_qr_text = text
                self.label.after(0, lambda t=text: self.update_qrcode(t))

            time.sleep(0.02)

    def update_qrcode(self, data: str):
        qr = qrcode.QRCode(
            version=1,
            box_size=8,
            border=2
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="green", back_color="black")
        img = img.resize((360, 360), Image.NEAREST)

        self.qr_img_tk = ImageTk.PhotoImage(img)
        self.qr_label.config(image=self.qr_img_tk)

    def quit(self):
        self.stop_flag.set()
        super().quit()

if __name__ == "__main__":
    NanoClock().mainloop()
