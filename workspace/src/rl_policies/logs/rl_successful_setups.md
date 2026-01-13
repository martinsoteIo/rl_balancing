# Reinforcement Learning – Ball-and-Plate Experiment Logs

This repository contains the trained weights, evaluation scripts, and environment configurations for a Ball-and-Plate robotic system. These logs represent the successful evolution of the control policies using **Genesis** and **MuJoCo**.

## How to Evaluate

To run any of the experiments listed below, use the main evaluation script with the corresponding experiment name:

```bash
juggling_2d_eval_telop.py --exp_name <EXPERIMENT_NAME> --ckpt <CHECKPOINT_NUMBER>
juggling_3d_eval_telop.py --exp_name <EXPERIMENT_NAME> --ckpt <CHECKPOINT_NUMBER>

## Validated Experiments

### **EXPERIMENT 002: 2D Stabilization (X-axis)**

- **Log Name:** Experimento002  
- **Environment Name:** `XPos2DEnv`  
- **Environment File:** `x_pos_env_genesis.py`  
- **Configuration:** `x_pos_genesis`  

**Summary:**  
Primary 2D balancing task focused on stabilization along the longitudinal (X) axis.
---

### **EXPERIMENT 003: 2D Robustness (Domain Randomization)**

- **Log Name:** Experimento003  
- **Environment Name:** `XPos2DEnv`  
- **Environment File:** `x_pos_env_genesis_mass_friction.py`  
- **Configuration:** `x_pos_genesis`  

**Summary:**
An extension of Experiment 002 aimed at improving real-world transferability through physical domain randomization.
---

### **EXPERIMENT 006: 3D Control (XY Plane)**

- **Log Name:** Experimento006  
- **Environment Name:** `XYPos3DEnv`  
- **Environment File:** `xy_pos_env_mass_friction.py`  
- **Configuration:** `xy_pos_config_mass_friction`  

**Summary:**  
Full XY-plane control of a **3-RRS parallel manipulator** under high-fidelity 3D physics.
---

### **EXPERIMENT 007: 3D Robustness (Domain Randomization)**

- **Log Name:** Experimento007  
- **Environment Name:** `XPos2DEnv` (Student Architecture)  
- **Environment File:** `x_pos_env_teacher_student.py`  
- **Configuration:** `xy_pos_config_mass_friction`  

**Summary:**  
An extension of Experiment 006 aimed at improving real-world transferability through physical domain randomization.