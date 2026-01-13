import os
import pickle
import torch
import numpy as np
import matplotlib
# Backend no interactivo para evitar conflictos con Qt/XCB
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from rl_policies.balancing_platform_2d.envs.x_pos_env_genesis_mass_friction import XPos2DEnv

# --- Configuración Tipografía LaTeX ---
plt.rcParams.update({
    "text.usetex": False,            
    "mathtext.fontset": "cm",        # Fuente Computer Modern (LaTeX)
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "svg.fonttype": "none", 
})

def run_full_sweep(exp_name="Experimento003", ckpt=9999):
    gs.init(logging_level="warning")

    # 1. Carga de configuración y entorno
    log_dir = f"/home/admin/workspace/src/rl_policies/logs/{exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env = XPos2DEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # 2. Carga del Runner y la Política
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    
    runner.alg.actor_critic.eval()
    policy = runner.get_inference_policy(device=gs.device)

    # 3. Configuración del Barrido
    targets = np.arange(-0.08, 0.081, 0.01)
    T_max = 5.0  
    dt = 0.02
    max_steps = int(T_max / dt)
    t_eval_start = 2.0 # Tiempo donde comienza la evaluación de precisión
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(targets)))

    print(f"--- Iniciando Barrido Completo: {len(targets)} trayectorias ---")

    for idx, target_x in enumerate(targets):
        time_series, pos_series = [], []
        
        obs, _ = env.reset(is_train=False)
        target_tensor = torch.tensor([[target_x]], device=gs.device, dtype=torch.float)
        
        for step in range(max_steps):
            with torch.no_grad():
                obs_float = obs.float()
                env.commands = target_tensor
                actions = policy(obs_float)
                obs, _, resets, _ = env.step(actions, is_train=False)
                
                ball_x = env.ball_pos[0, 0].item()
                time_series.append(step * dt)
                pos_series.append(ball_x)
                
                if resets.any(): break

        # Graficar trayectoria
        ax.plot(time_series, pos_series, color=colors[idx], lw=1.5, alpha=0.8, label=f'${target_x:+.2f}$ m')
        
        # Bandas de tolerancia del 5%
        tolerance = max(abs(target_x) * 0.05, 0.002)
        ax.axhline(y=target_x, color=colors[idx], linestyle='-', alpha=0.1, lw=0.8)
        ax.axhline(y=target_x + tolerance, color=colors[idx], linestyle='--', alpha=0.2, lw=0.5)
        ax.axhline(y=target_x - tolerance, color=colors[idx], linestyle='--', alpha=0.2, lw=0.5)

    # --- NUEVO: Línea discontinua vertical en t=2 ---
    ax.axvline(x=t_eval_start, color='gray', linestyle='--', lw=2, alpha=0.6, label=r'$t_{start} = 2.0$s')
    # Etiqueta sobre la línea para mayor claridad
    ax.text(t_eval_start + 0.05, ax.get_ylim()[0] + 0.005, r'$\mathrm{Steady\ State\ Start}$', 
            rotation=90, verticalalignment='bottom', color='gray', fontsize=10)

    # Estética LaTeX
    ax.set_title(rf'$\mathbf{{Transient\ Response\ Sweep:\ {exp_name}}}$', pad=20, fontsize=14)
    ax.set_xlabel(r'$\mathrm{Time}\ t\ \mathrm{[s]}$')
    ax.set_ylabel(r'$\mathrm{Ball\ Position}\ x\ \mathrm{[m]}$')
    ax.set_xlim([0, T_max])
    ax.set_ylim([-0.09, 0.09])
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title=r'$x_{target}$')

    output_path = os.path.join(log_dir, f"full_sweep_pastel_{exp_name}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"✅ Proceso finalizado. PDF guardado en: {output_path}")

if __name__ == "__main__":
    run_full_sweep(exp_name="Experimento003", ckpt=9999)