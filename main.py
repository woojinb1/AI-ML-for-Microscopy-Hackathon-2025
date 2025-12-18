import tkinter as tk
from gui.app import DSpacingApp
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    root = tk.Tk()
    app = DSpacingApp(root)
    root.mainloop()