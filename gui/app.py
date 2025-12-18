import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from shapely.geometry import Point
import numpy as np
import os

from core.loader import load_image_data
from core.processor import run_processing, reconstruct_from_groups
from gui.widgets import SmartControl

class DSpacingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ALMLHACK v3.2")
        self.root.geometry("1500x950")
        
        self.raw_img = None
        self.filepath = None
        self.analysis_results = {}
        self.current_view = 1
        
        # --- Parameter ---
        self.params = {
            'brightness': tk.DoubleVar(value=1.0),
            'mask_radius_dc': tk.IntVar(value=150),
            'threshold_abs': tk.IntVar(value=190),
            'mask_radius_peak': tk.IntVar(value=15),
            'tolerance_pct': tk.DoubleVar(value=8.0),
            'pixel_thresh_percent': tk.DoubleVar(value=95.0),
            'dbscan_eps': tk.DoubleVar(value=8.0),
            'dbscan_min_samples': tk.IntVar(value=60),
            'alpha_value': tk.DoubleVar(value=0.1)
        }
        
        self.pixel_size_val = tk.StringVar(value="0.026") 
        self.show_labels_var = tk.BooleanVar(value=True)

        self._setup_layout()

    def _setup_layout(self):
        # 1. Scrollable Control Panel
        left_container = ttk.Frame(self.root, width=370)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        
        canvas = tk.Canvas(left_container, width=350)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        control_frame = ttk.Frame(canvas, padding="10")
        
        canvas_window = canvas.create_window((0, 0), window=control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def on_frame_configure(event): canvas.configure(scrollregion=canvas.bbox("all"))
        def on_canvas_configure(event): canvas.itemconfig(canvas_window, width=event.width)
        control_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def _on_mouse_wheel(event):
            if event.delta: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                if event.num == 5: canvas.yview_scroll(1, "units")
                if event.num == 4: canvas.yview_scroll(-1, "units")
        
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
        canvas.bind_all("<Button-4>", _on_mouse_wheel)
        canvas.bind_all("<Button-5>", _on_mouse_wheel)

        ttk.Label(control_frame, text="Control Panel", font=("Arial", 16, "bold")).pack(pady=10)

        # 0. Preprocessing
        ttk.Label(control_frame, text="0. Preprocessing", foreground="blue").pack(anchor='w')
        SmartControl(control_frame, "Brightness (x)", self.params['brightness'], 0.1, 5.0, 0.1).pack(fill='x', pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 1. Peak Detection
        ttk.Label(control_frame, text="1. Peak Detection", foreground="blue").pack(anchor='w')
        SmartControl(control_frame, "DC Mask Radius", self.params['mask_radius_dc'], 1, 200, 1).pack(fill='x', pady=5)
        SmartControl(control_frame, "Peak Threshold", self.params['threshold_abs'], 1, 300, 1).pack(fill='x', pady=5)
        SmartControl(control_frame, "Peak Mask Radius", self.params['mask_radius_peak'], 1, 50, 1).pack(fill='x', pady=5)
        SmartControl(control_frame, "Group Tolerance", self.params['tolerance_pct'], 0.1, 20.0, 0.1).pack(fill='x', pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # 2. Reconstruction
        ttk.Label(control_frame, text="2. Reconstruction", foreground="blue").pack(anchor='w')
        SmartControl(control_frame, "Pixel Thresh %", self.params['pixel_thresh_percent'], 80.0, 99.9, 0.1).pack(fill='x', pady=5)
        SmartControl(control_frame, "DBSCAN EPS", self.params['dbscan_eps'], 1.0, 50.0, 0.5).pack(fill='x', pady=5)
        SmartControl(control_frame, "DBSCAN Min Pts", self.params['dbscan_min_samples'], 10, 500, 1).pack(fill='x', pady=5)
        SmartControl(control_frame, "Hull Alpha", self.params['alpha_value'], 0.0, 1.0, 0.01).pack(fill='x', pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 3. Calibration
        ttk.Label(control_frame, text="3. Calibration (nm)", foreground="green", font=("Arial", 11, "bold")).pack(anchor='w')
        calib_frame = ttk.Frame(control_frame)
        calib_frame.pack(fill='x', pady=5)
        
        ttk.Label(calib_frame, text="Pixel Size (nm):").pack(side='left')
        ttk.Entry(calib_frame, textvariable=self.pixel_size_val, width=10, justify='right').pack(side='left', padx=5)
        
        check_frame = ttk.Frame(control_frame)
        check_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(check_frame, text="Show Text Labels", variable=self.show_labels_var, 
                        command=lambda: self.switch_view(self.current_view)).pack(side='left')
        
        ttk.Label(control_frame, text="* Tip: Double-click Peak(View2) or Hull(View4) to delete", foreground="gray", font=("Arial", 9)).pack(anchor='w', pady=(5,0))

        # Buttons
        ttk.Button(control_frame, text="🔄 ANALYZE", command=self.run_analysis).pack(fill='x', pady=15, ipady=5)
        ttk.Button(control_frame, text="📂 Import", command=self.import_image).pack(fill='x')
        ttk.Button(control_frame, text="💾 Export", command=self.export_result).pack(fill='x')

        # 2. Display Panel
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(display_frame)
        btn_frame.pack(fill='x', pady=5)
        
        views = [("1. Original", 1), ("2. FFT Peaks", 2), ("3. IFFT+DBSCAN", 3), ("4. Hull & Area", 4)]
        for txt, val in views:
            ttk.Button(btn_frame, text=txt, command=lambda v=val: self.switch_view(v)).pack(side='left', expand=True, fill='x')

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.ax.text(0.5, 0.5, "No Image Loaded.\nPlease Click 'Import' Button.", 
                     ha='center', va='center', fontsize=20, color='gray')
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.canvas.mpl_connect('button_press_event', self.on_double_click)
        NavigationToolbar2Tk(self.canvas, display_frame).update()

    def import_image(self):
        path = filedialog.askopenfilename()
        if path:
            try:
                self.filepath = path
                self.raw_img = load_image_data(path) 
                self.run_analysis()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def run_analysis(self):
        if self.raw_img is None: return
        
        p_values = {k: v.get() for k, v in self.params.items()}
        
        try:
            px_size = float(self.pixel_size_val.get())
            if px_size <= 0: raise ValueError
            p_values['pixel_size_nm'] = px_size
        except ValueError:
            messagebox.showwarning("Input Error", "Invalid Pixel Size (nm). Must be > 0.")
            return

        self.analysis_results = run_processing(self.raw_img, p_values)
        self.switch_view(self.current_view)

    def on_double_click(self, event):
        if not event.dblclick: return
        if not self.analysis_results: return
        if event.inaxes != self.ax: return 

        # ---------------------------
        # Case A: View 2 (FFT Peaks) - 클릭한 Peak 쌍만 삭제
        # ---------------------------
        if self.current_view == 2:
            click_y, click_x = event.ydata, event.xdata
            
            groups = self.analysis_results.get('groups', {})
            mask_r = self.analysis_results.get('mask_radius_peak', 20)
            
            target_gid = -1
            target_idx = -1
            
            for gid, peaks in groups.items():
                for i, p in enumerate(peaks):
                    # 거리 계산 (p[0]=row=y, p[1]=col=x)
                    dist = np.sqrt((p[0] - click_y)**2 + (p[1] - click_x)**2)
                    if dist <= mask_r:
                        target_gid = gid
                        target_idx = i
                        break
                if target_gid != -1: break

            if target_gid != -1 and target_idx != -1:
                current_peaks = groups[target_gid]
                

                if target_idx % 2 == 0:
                    partner_idx = target_idx + 1
                else:
                    partner_idx = target_idx - 1
                

                if partner_idx < len(current_peaks):

                    indices_to_remove = sorted([target_idx, partner_idx], reverse=True)
                    
                    for idx in indices_to_remove:
                        del current_peaks[idx]

                if not current_peaks:
                    del groups[target_gid]
                    if target_gid in self.analysis_results['d_spacings']:
                        del self.analysis_results['d_spacings'][target_gid]
                

                p_values = {k: v.get() for k, v in self.params.items()}
                try:
                    p_values['pixel_size_nm'] = float(self.pixel_size_val.get())
                except: pass
                
                new_ifft, new_points, new_hull, _ = run_processing.__globals__['reconstruct_from_groups'](
                    self.analysis_results['fshift'], groups, p_values
                )
                

                self.analysis_results['canvas_ifft'] = new_ifft
                self.analysis_results['canvas_points'] = new_points
                self.analysis_results['hull_data'] = new_hull
                
                self.switch_view(2)

        # ---------------------------
        # Case B: View 4 (Hull)
        # ---------------------------
        elif self.current_view == 4:
            if 'hull_data' not in self.analysis_results: return
            
            click_point = Point(event.ydata, event.xdata) 
            hull_list = self.analysis_results['hull_data']
            deleted = False
            
            for i in range(len(hull_list) - 1, -1, -1):
                geom = hull_list[i]['geom']
                if geom.contains(click_point): 
                    del hull_list[i] 
                    deleted = True
                    break 
            
            if deleted:
                self.switch_view(4)

    def switch_view(self, view_idx):
        if not self.analysis_results: return
        self.current_view = view_idx
        self.ax.clear()
        res = self.analysis_results
        
        display_img = res.get('display_img')
        if display_img is None: display_img = self.raw_img
        
        if view_idx == 1:
            self.ax.imshow(display_img, cmap='gray')
            self.ax.axis('off')

        elif view_idx == 2:
            self.ax.imshow(res['magnitude'], cmap='gray')
            
            d_spacings = res.get('d_spacings', {})

            for gid, g_peaks in res['groups'].items():
                color = res['group_colors'][gid % 10]
                for p in g_peaks:
                    circ = Circle((p[1], p[0]), radius=res['mask_radius_peak'], edgecolor=color, facecolor='none', lw=2)
                    self.ax.add_patch(circ)
                
                if g_peaks and self.show_labels_var.get():
                    d_val = d_spacings.get(gid, 0.0)
                    tx, ty = g_peaks[0][1], g_peaks[0][0] - 20 
                    self.ax.text(tx, ty, f"G{gid}\nd={d_val:.3f}nm", color=color, fontsize=10, fontweight='bold')
            self.ax.axis('off')

        elif view_idx == 3:
            combined = res['canvas_ifft'].copy()
            pt_mask = np.any(res['canvas_points'] > 0, axis=-1)
            combined[pt_mask] = res['canvas_points'][pt_mask]
            self.ax.imshow(combined)
            self.ax.axis('off')

        elif view_idx == 4:
            self.ax.imshow(display_img, cmap='gray')
            
            total_area = 0
            for item in res['hull_data']:
                hull = item['geom']
                color = item['color']
                area = item['area']
                gid = item['id']
                total_area += area

                geoms = hull.geoms if hasattr(hull, 'geoms') else [hull]
                for poly in geoms:
                    if poly.is_empty: continue
                    poly_r, poly_c = poly.exterior.xy
                    self.ax.plot(poly_c, poly_r, color=color, linewidth=2, alpha=0.9)
                
                if self.show_labels_var.get():
                    try:
                        cx, cy = item['centroid'].x, item['centroid'].y
                        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                        self.ax.text(cy, cx, f"G{gid}\n{area:.2f}", 
                                     color='black', fontsize=8, ha='center', va='center', bbox=bbox_props)
                    except: pass
            
            self.ax.set_title(f"Concave Hull Total Area: {total_area:.2f} nm²", 
                              fontsize=12, fontweight='bold')
            self.ax.axis('off')
            
        self.canvas.draw()

    def export_result(self):
        if not self.analysis_results: return
        
        if self.filepath:
            base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        else:
            base_name = "Analysis_Result"
        
        default_name = f"{base_name}_{self.current_view}.png"
        
        filetypes = [("PNG file", "*.png"), ("PDF file", "*.pdf")]
        path = filedialog.asksaveasfilename(
            title="Save Plot", 
            filetypes=filetypes, 
            defaultextension=".png",
            initialfile=default_name
        )
        
        if path:
            self.fig.savefig(path, bbox_inches='tight')
            messagebox.showinfo("Success", f"Saved:\n{path}")