# uav/interface.py
import tkinter as tk
from threading import Thread

exit_flag = [False]

def launch_control_gui(param_refs):
    def on_stop():
        exit_flag[0] = True

    def on_reset():
        param_refs['reset_flag'][0] = True

    def update_labels():
        state_val.set(param_refs['state'][0])
        root.after(200, update_labels)

    root = tk.Tk()
    root.title("UAV Controller")
    root.geometry("300x200")

    state_val = tk.StringVar()

    tk.Button(root, text="Reset Simulation", command=on_reset).pack(pady=10)
    tk.Button(root, text="Stop UAV", command=on_stop, bg='red', fg='white').pack(pady=10)

    tk.Label(root, text="Current State:").pack()
    tk.Label(root, textvariable=state_val, font=("Arial", 14)).pack()

    update_labels()
    root.mainloop()

def start_gui(param_refs=None):
    if param_refs is None:
        Thread(target=gui_exit, daemon=True).start()
    else:
        Thread(target=lambda: launch_control_gui(param_refs), daemon=True).start()

def gui_exit():
    root = tk.Tk()
    root.title("Stop UAV")
    root.geometry("200x100")
    btn = tk.Button(root, text="STOP", font=("Arial", 20), command=lambda: exit_flag.__setitem__(0, True))
    btn.pack(expand=True)
    root.mainloop()
