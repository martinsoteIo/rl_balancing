"""
================================================================================
TENSORBOARD TO ACADEMIC PDF EXPORTER: METRIC VISUALIZATION ENGINE
================================================================================
Description:
    This module serves as a publication-pipeline utility designed to convert raw 
    TensorBoard event logs (tfevents) into high-fidelity, vector-based PDF plots 
    suitable for academic papers and theses.
    
    The software addresses the inherent noise in Reinforcement Learning training 
    data by generating dual-layer visualizations that present both the raw 
    stochastic data (as a background shadow) and the underlying trend (as a 
    smoothed trajectory).
Installation:
    Ensure that the following Python packages are installed in your environment with:
    sudo apt-get update
    sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
Key Features:
    * LaTeX Rendering Engine: Leverages the system's TeX distribution to render 
      labels and titles using 'Computer Modern' fonts and mathematical notation 
      (e.g., converting 'rew_fall_event' to $R_{fall\_event}$).
    * Dual-Layer Visualization: Simultaneously plots raw data (low opacity) to 
      show variance and EMA-smoothed data (high opacity) to show convergence.
    * Smart Outlier Rejection: Implements a percentile-based auto-scaling 
      algorithm (1st-99th percentile) to automatically crop initial training 
      spikes and focus on the relevant performance range.
    * Automated Semantic Mapping: Translates internal code variable names into 
      standardized mathematical symbols defined in a lookup dictionary.

Author: Martín Sotelo Aguirre
Version: 1.2 (Smart-Scaling & LaTeX Support)
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorboard.backend.event_processing import event_accumulator

# --- 1. CONFIGURACIÓN ACADÉMICA (LaTeX) ---
try:
    matplotlib.rcParams.update({
        "text.usetex": True,            
        "font.family": "serif",
        "font.serif": ["Computer Modern"], 
        "axes.labelsize": 20,
        "font.size": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.titlesize": 20,
        "figure.autolayout": True
    })
except Exception as e:
    print(f"Warning: LaTeX config failed ({e}). Using defaults.")

# Paleta de colores pastel
PASTEL_COLORS = ['#AEC6CF', '#FFB347', '#77DD77', '#F49AC2', '#CFCFC4', '#B39EB5']

# --- 2. FUNCIONES DE PROCESAMIENTO ---

def smooth_data(data, weight=0.95):
    """ Suavizado exponencial (EMA). """
    data = np.array(data, dtype=np.float64).ravel()
    if len(data) == 0: return np.array([])
    
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + point * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def smart_ylim(ax, values, margin=0.05):
    """
    Ajusta los límites del eje Y automáticamente usando percentiles 
    para ignorar outliers extremos (picos iniciales).
    """
    if len(values) < 10: return # No tocar si hay pocos datos
    
    # Calculamos percentiles 1% y 99% para eliminar picos locos
    y_low = np.percentile(values, 1)
    y_high = np.percentile(values, 99)
    
    # Si la gráfica es plana, no hacemos nada
    if y_high == y_low: return

    # Añadimos un pequeño margen visual (5%)
    span = y_high - y_low
    y_min = y_low - (span * margin)
    y_max = y_high + (span * margin)
    
    ax.set_ylim(y_min, y_max)

# --- 3. DICCIONARIO DE ETIQUETAS LATEX ---
METRIC_LABELS = {
    "rew_fall_event":           r"$R_{\mathrm{fall\_event}}$",
    "rew_similar_to_default":   r"$R_{\mathrm{default}}$",
    "rew_stabilize_x_velocity": r"$R_{\mathrm{stabilize}\ v_x}$",
    "rew_success_event":        r"$R_{\mathrm{success}}$",
    "rew_tracking_x":           r"$R_{\mathrm{tracking}\ x}$",
    "Loss_entropy":             r"$\mathcal{L}_{\mathrm{entropy}}$",
    "Loss_learning_rate":       r"$\alpha$ (Learning Rate)",
    "Loss_surrogate":           r"$\mathcal{L}_{\mathrm{surrogate}}$",
    "Loss_value_function":      r"$\mathcal{L}_{\mathrm{value}}$",
    "Perf_collection_time":     r"$T_{\mathrm{collection}}$ (s)",
    "Perf_learning_time":       r"$T_{\mathrm{learning}}$ (s)",
    "Perf_total_fps":           r"$\mathrm{FPS}_{\mathrm{total}}$",
    "Policy_mean_noise_std":    r"$\bar{\sigma}_{\mathrm{noise}}$",
    "Train_mean_episode_length": r"$\bar{L}_{\mathrm{episode}}$",
    "Train_mean_reward":         r"$\bar{R}_{\mathrm{mean}}$"
}

def export_metrics(experiment_name, source_base, output_base, smooth_weight=0.95):
    log_dir = os.path.join(source_base, experiment_name)
    output_dir = os.path.join(output_base, experiment_name, "plots_pdf")

    if not os.path.exists(log_dir):
        print(f"Error: Path {log_dir} not found.")
        return

    print(f"Reading logs from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    try:
        tags = ea.Tags()['scalars']
    except KeyError:
        print("No scalars found.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, tag in enumerate(tags):
        try:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events], dtype=np.float64).ravel()
            values = np.array([e.value for e in events], dtype=np.float64).ravel()
            
            steps = np.nan_to_num(steps)
            values = np.nan_to_num(values)

            if len(values) == 0: continue

            color = PASTEL_COLORS[i % len(PASTEL_COLORS)] 
            smoothed_values = smooth_data(values, weight=smooth_weight)

            fig, ax = plt.subplots(figsize=(8, 5))

            # 1. Datos Crudos (Sombra)
            ax.plot(steps, values, color=color, alpha=0.25, linewidth=1.0, zorder=1)

            # 2. Línea Suavizada (Principal)
            ax.plot(steps, smoothed_values, color=color, linewidth=1.5, zorder=2, label="Smoothed")

            # --- AUTO-ESCALADO INTELIGENTE ---
            smart_ylim(ax, values)

            # Etiquetas
            clean_tag = tag.split('/')[-1]
            if clean_tag in METRIC_LABELS:
                pretty_label = METRIC_LABELS[clean_tag]
            else:
                escaped_tag = clean_tag.replace('_', r'\_')
                pretty_label = fr"$\mathrm{{{escaped_tag}}}$"
            
            ax.set_xlabel(r"Training Iteration (Step)")
            ax.set_ylabel(pretty_label)
            ax.set_title(pretty_label)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            file_tag = tag.replace('/', '_').replace(' ', '_')
            pdf_path = os.path.join(output_dir, f"{file_tag}.pdf")
            plt.savefig(pdf_path, format="pdf", bbox_inches='tight')
            plt.close(fig)
            print(f"Exported: {pdf_path}")
            
        except Exception as e:
            print(f"FAILED to export {tag}: {e}")
            continue 

if __name__ == "__main__":
    SOURCE_PATH = os.path.expanduser("~/workspace/src/rl_policies/logs/")
    OUTPUT_PATH = os.path.expanduser("~/workspace/src/rl_policies/rl_policies/juggling_platform_2d/utils/Tensorboard/")
    TARGET_EXPERIMENT = "Experimento003" 

    export_metrics(TARGET_EXPERIMENT, SOURCE_PATH, OUTPUT_PATH, smooth_weight=0.95)