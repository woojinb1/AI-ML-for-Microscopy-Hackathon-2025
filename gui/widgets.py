import tkinter as tk
from tkinter import ttk

class SmartControl(ttk.Frame):
    """
    Label + Entry + [-] Button + Slider + [+] Button을 하나로 묶은 위젯
    """
    def __init__(self, parent, label_text, var, min_val, max_val, step, **kwargs):
        super().__init__(parent, **kwargs)
        self.var = var
        self.step = step
        self.digits = 0 if isinstance(step, int) else len(str(step).split(".")[1])

        self._setup_ui(label_text, min_val, max_val)

    def _setup_ui(self, label_text, min_val, max_val):
        # Top Row: Label & Entry
        top_row = ttk.Frame(self)
        top_row.pack(fill=tk.X)
        ttk.Label(top_row, text=label_text, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        ttk.Entry(top_row, textvariable=self.var, width=6, justify='right').pack(side=tk.RIGHT)

        # Bottom Row: Controls
        bot_row = ttk.Frame(self)
        bot_row.pack(fill=tk.X, pady=2)

        ttk.Button(bot_row, text="-", width=2, command=lambda: self._change_val(-1, min_val, max_val)).pack(side=tk.LEFT)
        
        scale = ttk.Scale(bot_row, from_=min_val, to=max_val, variable=self.var, orient=tk.HORIZONTAL)
        scale.configure(command=lambda v: self._on_scale_move(v)) # 스냅 로직 연결
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(bot_row, text="+", width=2, command=lambda: self._change_val(1, min_val, max_val)).pack(side=tk.RIGHT)

    def _change_val(self, direction, min_v, max_v):
        current = self.var.get()
        new_val = round(current + (direction * self.step), self.digits)
        
        new_val = max(min_v, min(new_val, max_v))
        
        if self.digits == 0: self.var.set(int(new_val))
        else: self.var.set(new_val)

    def _on_scale_move(self, val):
        v = float(val)
        snapped = round(round(v / self.step) * self.step, self.digits)
        if self.digits == 0: self.var.set(int(snapped))
        else: self.var.set(snapped)