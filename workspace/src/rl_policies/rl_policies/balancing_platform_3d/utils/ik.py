import math
import numpy as np
import torch

def inverse_kinematics(psi_x: float, psi_y: float, o7_z: float) -> torch.Tensor:
    """
    Inverse kinematics strictly following the paper "Position Kinematics of a 3-RRS PM".
    
    Args:
        psi_x (float): Roll angle.
        psi_y (float): Pitch angle.
        o7_z (float): Height O7z of the platform center.
        
    Returns:
        torch.Tensor: Motor angles [theta1, theta2, theta3].
    """
    RB = 0.078268  # Base radius (m): distance from O0 to each base joint
    RP = 0.08  # Platform radius (m): distance from O7 to the attachment point
    L1 = 0.08  # First link length (m) (Proximal link)
    L2 = 0.078  # Second link length (m) (Distal link)
    
    sx = math.sin(psi_x)
    cx = math.cos(psi_x)
    sy = math.sin(psi_y)
    cy = math.cos(psi_y)
    
    psi_z = math.atan2(-sx * sy, cx + cy)
    sz = math.sin(psi_z)
    cz = math.cos(psi_z)

    ux = cy * cz
    uy = sx * sy * cz + cx * sz
    vy = cx * cz - sx * sy * sz

    R = np.array([
        [            cy*cz,             -cy*sz,        sy ],
        [ sx*sy*cz + cx*sz,   cx*cz - sx*sy*sz,    -sx*cy ],
        [ sx*sz - cx*sy*cz,   sx*cz + cx*sy*sz,     cx*cy ]
    ])

    o7_x = RP * (ux - vy) / 2.0
    o7_y = -uy * RP
    vec_O0_O7 = np.array([o7_x, o7_y, o7_z])

    thetas = []
    alphas = [0.0, 2.0 * math.pi/3.0, 4.0 * math.pi/3.0]

    for i in range(3):
        alpha = alphas[i] 
        vec_p = np.array([RP, 0.0, 0.0])

        # This matrix rotates vector p towards the corresponding vertex of the triangle.
        c_alpha, s_alpha = math.cos(alpha), math.sin(alpha)
        Rz_alpha = np.array([
            [c_alpha, -s_alpha, 0.0],
            [s_alpha,  c_alpha, 0.0],
            [ 0.0,         0.0, 1.0]
        ])

        vec_O0_O7j = vec_O0_O7 + R @ (Rz_alpha @ vec_p)
        O7j_x = vec_O0_O7j[0]
        O7j_z = vec_O0_O7j[2]
        
        Ai = 2 * L1 * c_alpha * (-O7j_x + RB * c_alpha)
        Bi = 2 * L1 * O7j_z * (c_alpha**2)
        Ci = O7j_x**2 - 2 * RB * O7j_x * c_alpha + c_alpha**2 * (RB**2 + L1**2 - L2**2 + O7j_z**2)

        D = Ai**2 + Bi**2 - Ci**2
        if D > 0:
            theta = 2 * math.atan2(-Bi + math.sqrt(D), Ci - Ai)
        else:
            theta = 0.0

        if i == 0:
            theta = - math.pi / 6 - theta
        else:
            theta = math.pi / 6 + theta
        
        thetas.append(theta)

    return torch.tensor([thetas], dtype=torch.float32)
