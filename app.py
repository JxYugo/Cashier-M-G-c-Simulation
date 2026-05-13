import tkinter as tk
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math


class QueueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("M/G/c Queue Simulator (Optimized)")
        self.root.geometry("1200x850")

        # ================= INPUTS =================
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        fields = [
            ("Servers (c)", "3", 0, 0),
            ("Arrivals/hr (λ)", "35", 0, 2),
            ("Service Mean (min)", "4", 1, 0),
            ("Service StdDev (σ)", "1.0", 1, 2),
            ("Max Samples", "2000", 0, 4)  # Renamed to clarify these are post-warmup samples
        ]

        self.entries = {}
        for text, default, r, c in fields:
            tk.Label(self.control_frame, text=text).grid(row=r, column=c, padx=5)
            entry = tk.Entry(self.control_frame, width=8)
            entry.insert(0, default)
            entry.grid(row=r, column=c + 1)
            self.entries[text] = entry

        tk.Button(self.control_frame, text="Run Simulation", command=self.start_sim, bg="#4CAF50", fg="white",
                  font=("Arial", 10, "bold")).grid(row=2, column=1, pady=10)
        tk.Button(self.control_frame, text="Toggle Pause", command=self.toggle_pause).grid(row=2, column=2)
        tk.Button(self.control_frame, text="Show Stats", command=self.validate).grid(row=2, column=3)

        self.canvas = tk.Canvas(root, width=1100, height=300, bg="#fdfdfd", highlightthickness=1,
                                highlightbackground="#ddd")
        self.canvas.pack(pady=5)
        self.status = tk.Label(root, text="System Idle - Ready to Simulate", font=("Arial", 11), fg="#555")
        self.status.pack()

        self.graph_win = None
        self.running = False
        self.paused = False

    def get_theoretical_wait(self):
        c = int(self.entries["Servers (c)"].get())
        lam = float(self.entries["Arrivals/hr (λ)"].get()) / 60.0
        mu_mean = float(self.entries["Service Mean (min)"].get())
        sigma = float(self.entries["Service StdDev (σ)"].get())

        rho = (lam * mu_mean) / c
        if rho >= 1: return float('inf')

        cv_s = sigma / mu_mean

        # FIX: More accurate M/M/c component for the Kingman approximation
        # Using Sakasegawa's more precise estimation for the M/M/c wait time
        wq_mmc = (rho ** (math.sqrt(2 * (c + 1)) - 1) * mu_mean) / (c * (1 - rho))

        # Adjust for General distribution (M/G/c)
        wq_mgc = wq_mmc * ((1 + cv_s ** 2) / 2)
        return wq_mgc

    def start_sim(self):
        try:
            self.num_servers = int(self.entries["Servers (c)"].get())
            self.rate = float(self.entries["Arrivals/hr (λ)"].get()) / 60.0
            self.mean = float(self.entries["Service Mean (min)"].get())
            self.std_dev = float(self.entries["Service StdDev (σ)"].get())
            self.max_samples = int(self.entries["Max Samples"].get())  # Fix: target valid sample count
        except ValueError:
            return

        self.warmup_limit = 100  # Increased warmup for better steady-state convergence
        self.time = 0
        self.dt = 0.1
        self.next_arrival = random.expovariate(self.rate)

        self.servers = [0.0] * self.num_servers
        self.queue = []
        self.wait_times = []
        self.raw_wait_times = []
        self.id_counter = 0
        self.time_hist, self.queue_hist = [], []

        self.canvas.delete("all")
        self.draw_static_elements()
        self.open_graph_window()

        self.running = True
        self.paused = False
        self.update()

    def toggle_pause(self):
        self.paused = not self.paused

    def draw_static_elements(self):
        for i in range(self.num_servers):
            x, y = 900, 30 + i * 50
            self.canvas.create_rectangle(x, y, x + 100, y + 35, fill="#e3f2fd", outline="#1976d2", width=2)
            self.canvas.create_text(x + 50, y + 17, text=f"Server {i + 1}", font=("Arial", 9))

    def create_customer(self):
        c = self.canvas.create_oval(0, 0, 24, 24, fill="#ffb74d", outline="#e65100")
        t = self.canvas.create_text(0, 0, text=str(self.id_counter), font=("Arial", 7, "bold"))
        self.queue.append(
            {"circle": c, "text": t, "arrival": self.time, "id": self.id_counter})  # Added ID for tracking
        self.id_counter += 1

    def process_servers(self):
        for i in range(len(self.servers)):
            if self.servers[i] <= self.time and self.queue:
                cust = self.queue.pop(0)
                wait = self.time - cust["arrival"]

                # FIX: Use cust["id"] to ensure warmup filtering is accurate based on customer sequence
                if cust["id"] >= self.warmup_limit:
                    self.wait_times.append(wait)

                self.raw_wait_times.append(wait)
                service_time = max(0.1, random.gauss(self.mean, self.std_dev))
                self.servers[i] = self.time + service_time
                self.canvas.delete(cust["circle"])
                self.canvas.delete(cust["text"])

    def update_queue_visuals(self):
        for i, c in enumerate(self.queue[:25]):
            x = 850 - (i * 30)
            y = 150
            self.canvas.coords(c["circle"], x, y, x + 24, y + 24)
            self.canvas.coords(c["text"], x + 12, y + 12)

    def open_graph_window(self):
        if self.graph_win is None or not self.graph_win.winfo_exists():
            self.graph_win = tk.Toplevel(self.root)
            self.graph_win.title("Live Metrics")
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 6))
            self.fig.tight_layout(pad=3.5)
            self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.graph_win)
            self.canvas_fig.get_tk_widget().pack()

    def validate(self):
        if not self.wait_times: return
        val_win = tk.Toplevel(self.root)
        val_win.title("Statistical Validation")

        sim_mean = np.mean(self.wait_times)
        theo_mean = self.get_theoretical_wait()

        c = int(self.entries["Servers (c)"].get())
        lam = float(self.entries["Arrivals/hr (λ)"].get()) / 60.0
        mu_mean = float(self.entries["Service Mean (min)"].get())
        rho = (lam * mu_mean) / c
        utilization_pct = rho * 100

        if theo_mean == float('inf') or rho >= 1:
            report = "SYSTEM UNSTABLE\nArrival rate exceeds capacity!\n\n"
            report += f"Utilization: {utilization_pct:.2f}%\n"
        else:
            # FIX: Metrics now calculated using only valid post-warmup samples
            errors = np.array(self.wait_times) - theo_mean
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)
            nrmse_val = rmse / theo_mean if theo_mean != 0 else 0
            rel_error = abs(theo_mean - sim_mean) / theo_mean

            fit = "EXCELLENT" if rel_error < 0.1 else "GOOD" if rel_error < 0.2 else "POOR"

            report = f"VALIDATION REPORT (M/G/{self.num_servers})\n"
            report += f"{'-' * 35}\n"
            report += f"Utilization (ρ): {utilization_pct:.2f}%\n"
            report += f"Theoretical Wq:  {theo_mean:.4f} min\n"
            report += f"Simulated Wq:    {sim_mean:.4f} min\n"
            report += f"{'-' * 35}\n"
            report += f"Relative Error:  {rel_error:.2%}\n"
            report += f"MSE:             {mse:.4f}\n"
            report += f"RMSE:            {rmse:.4f}\n"
            report += f"NRMSE:           {nrmse_val:.4f}\n"
            report += f"Model Fit:       {fit}\n\n"
            report += f"Valid Samples:   {len(self.wait_times)}"  # Fix: Changed label to clarify sample source

        tk.Label(val_win, text=report, font=("Courier", 11), justify="left", padx=25, pady=25).pack()

    def update(self):
        if not self.running: return

        if not self.paused:
            self.time += self.dt

            if self.time >= self.next_arrival:
                self.create_customer()
                self.next_arrival = self.time + random.expovariate(self.rate)

            self.process_servers()
            self.update_queue_visuals()

            if int(self.time / self.dt) % 5 == 0:
                self.time_hist.append(self.time)
                self.queue_hist.append(len(self.queue))
                if len(self.time_hist) > 100:
                    self.time_hist.pop(0)
                    self.queue_hist.pop(0)

                self.ax1.clear()
                self.ax1.plot(self.time_hist, self.queue_hist, color='#d32f2f')
                self.ax1.set_title("Queue Length Over Time")
                self.ax1.grid(True, alpha=0.3)

                self.ax2.clear()
                if self.raw_wait_times:
                    self.ax2.hist(self.raw_wait_times, bins=20, color='#64b5f6', edgecolor='white')
                self.ax2.set_title("Wait Time Distribution")
                self.canvas_fig.draw()

            # FIX: The exit condition now waits until we have collected the EXACT number of post-warmup samples requested
            if len(self.wait_times) >= self.max_samples:
                self.running = False
                self.status.config(text="Simulation Complete", fg="#2e7d32")
                self.validate()
                return

        # FIX: Updated status bar to show "Valid" samples count for clarity during run
        status_text = f"Clock: {self.time:.1f}m | Queue: {len(self.queue)} | Valid Samples: {len(self.wait_times)}"
        if self.id_counter <= self.warmup_limit:
            status_text += " (Warming Up...)"
            self.status.config(fg="#f57c00")
        else:
            self.status.config(fg="#1976d2")

        self.status.config(text=status_text)
        self.root.after(10, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = QueueApp(root)
    root.mainloop()