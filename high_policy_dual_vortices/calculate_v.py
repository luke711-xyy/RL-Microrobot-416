import math
import os
from os import path

import numpy as np
import torch
from math import cos, sin


directory_path = os.getcwd()
folder_name = path.basename(directory_path)


# 保持与单机器人分支相同的离散化依赖和历史平面映射约定。
Xf_match_q_fila_template = torch.load("Xf_match_q_fila.pt").to(torch.double)
Yf_match_q_fila_template = torch.load("Yf_match_q_fila.pt").to(torch.double)
Zf_match_q_fila_template = torch.load("Zf_match_q_fila.pt").to(torch.double)
Xf_all_fila_template = torch.load("Xf_all_fila.pt").to(torch.double)
Yf_all_fila_template = torch.load("Yf_all_fila.pt").to(torch.double)
Zf_all_fila_template = torch.load("Zf_all_fila.pt").to(torch.double)
Label_Matrix_fila = torch.load("Min_Distance_Label_Fila.pt").to(torch.int64)
Min_Distance_num_fila = torch.load("Min_Distance_num_fila.pt").view(-1).to(torch.int64)
Min_Distance_Label_fila = torch.load("Correponding_label_fila.pt").to(torch.double)


device = torch.device("cpu")

NL = 10
N_dense = int(NL * 8)
N = int(NL * 4)
FORCE_POINT_NUM = int(Xf_all_fila_template.shape[0])
MATCH_POINT_NUM = int(Xf_match_q_fila_template.shape[1])
MU = 1.0

torch.set_num_threads(int(os.environ.get("STOKES_NUM_THREADS", "5")))


# ================= 背景涡流（Taylor-Green 细胞涡流） =================

VORTEX_STRENGTH = float(os.environ.get("VORTEX_STRENGTH", "0.1"))
VORTEX_CELL_L = float(os.environ.get("VORTEX_CELL_L", "1.0"))


def vortex_velocity(x, y, Uv=None, Lv=None):
    """Taylor-Green 细胞涡流速度场（不可压：∇·u=0）。
    支持标量或 numpy 数组输入，返回 (ux, uy)。"""
    Uv = VORTEX_STRENGTH if Uv is None else Uv
    Lv = VORTEX_CELL_L if Lv is None else Lv
    k = math.pi / Lv
    ux = Uv * np.sin(k * x) * np.cos(k * y)
    uy = -Uv * np.cos(k * x) * np.sin(k * y)
    return ux, uy


def vortex_vorticity(x, y, Uv=None, Lv=None):
    """涡度场 ω = ∂v/∂x − ∂u/∂y（用于可视化背景）。"""
    Uv = VORTEX_STRENGTH if Uv is None else Uv
    Lv = VORTEX_CELL_L if Lv is None else Lv
    k = math.pi / Lv
    return 2.0 * Uv * k * np.sin(k * x) * np.sin(k * y)


def MatrixQ(L, theta, Qu, Q1, Ql, Q2):
    q_up = torch.cat((Qu, -Q2), dim=1)
    q_down = torch.cat((Ql, Q1), dim=1)
    return torch.cat((q_up, q_down), dim=0)


def MatrixQp(L, theta):
    Qu = torch.cat(
        (
            torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    Ql = torch.cat(
        (
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    q1 = L * torch.cos(theta[2:])
    q2 = L * torch.sin(theta[2:])

    Q1 = q1.reshape(1, -1).repeat(N + 1, 1)
    Q1 = torch.tril(Q1, -1)

    Q2 = q2.reshape(1, -1).repeat(N + 1, 1)
    Q2 = torch.tril(Q2, -1)

    Q = torch.cat((Qu, Q1, Ql, Q2), dim=1)
    Q = Q.reshape(2 * (N + 1), -1)
    return Q, Qu, Q1, Ql, Q2


def MatrixQp_dense(L, theta):
    Qu = torch.cat(
        (
            torch.ones((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.zeros((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    Ql = torch.cat(
        (
            torch.zeros((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.ones((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    q1 = L * torch.cos(theta[2:])
    q2 = L * torch.sin(theta[2:])

    Q1 = q1.reshape(1, -1).repeat(N_dense + 1, 1)
    Q1 = torch.tril(Q1, -1)

    Q2 = q2.reshape(1, -1).repeat(N_dense + 1, 1)
    Q2 = torch.tril(Q2, -1)

    Q = torch.cat((Qu, Q1, Ql, Q2), dim=1)
    Q = Q.reshape(2 * (N_dense + 1), -1)
    return Q, Qu, Q1, Ql, Q2


def MatrixB(L, theta, Y):
    B1 = 0.5 * torch.cat(
        (
            2 * torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    B1[0, 0] = 0.5
    B1[-1, 0] = 0.5
    B1 = B1.reshape(1, -1)

    B2 = 0.5 * torch.cat(
        (
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            2 * torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    B2[0, 1] = 0.5
    B2[-1, 1] = 0.5
    B2 = B2.reshape(1, -1)

    B3 = torch.cat((-(Y[:, 1] - Y[0, 1]).view(1, -1), (Y[:, 0] - Y[0, 0]).view(1, -1)), dim=1)
    B3 = B3.reshape(1, -1)

    B = torch.cat((B1, B2), dim=0)

    Bx = B[:, 0::2]
    Bx = torch.cat((Bx, B3[:, : Bx.shape[1]]), dim=0)
    By = B[:, 1::2]
    By = torch.cat((By, B3[:, Bx.shape[1] :]), dim=0)

    Min_Distance_num_fila_copy = Min_Distance_num_fila.clone().reshape(1, -1).repeat(3, 1).to(torch.double)
    Bx = Bx * Min_Distance_num_fila_copy
    By = By * Min_Distance_num_fila_copy
    return torch.cat((Bx, By), dim=1)


def MatrixC(action_absolute):
    C1 = torch.zeros((N + 2, 3), dtype=torch.double, device=device)
    C1[0, 0] = 1
    C1[1, 1] = 1
    C1[2:, 2] = 1

    C2 = torch.zeros((N + 2, 1), dtype=torch.double, device=device)
    C2[3:, :] = action_absolute.view(-1, 1)
    return C1, C2


def MatrixD_sum(beta_ini, absU):
    D1 = torch.zeros((NL * 2 + 2, 3), dtype=torch.double, device=device)
    D2 = torch.zeros((NL * 2 + 2, 1), dtype=torch.double, device=device)
    for i in range(NL + 1):
        if i == 0:
            D1[i * 2, 0] = 1
            D1[i * 2 + 1, 1] = 1
        elif i == 1:
            D1[i * 2, 0] = 1
            D1[i * 2, 2] = D1[(i - 1) * 2, 2] - sin(beta_ini[i - 1]) * 1.0 / NL
            D1[i * 2 + 1, 1] = 1
            D1[i * 2 + 1, 2] = D1[(i - 1) * 2 + 1, 2] + cos(beta_ini[i - 1]) * 1.0 / NL
            D2[i * 2, :] = D2[(i - 1) * 2, :]
            D2[i * 2 + 1, :] = D2[(i - 1) * 2 + 1, :]
        else:
            D1[i * 2, 0] = 1
            D1[i * 2, 2] = D1[(i - 1) * 2, 2] - sin(beta_ini[i - 1]) * 1.0 / NL
            D1[i * 2 + 1, 1] = 1
            D1[i * 2 + 1, 2] = D1[(i - 1) * 2 + 1, 2] + cos(beta_ini[i - 1]) * 1.0 / NL
            D2[i * 2, :] = D2[(i - 1) * 2, :] - absU[i - 2] * sin(beta_ini[i - 1]) * 1.0 / NL
            D2[i * 2 + 1, :] = D2[(i - 1) * 2 + 1, :] + absU[i - 2] * cos(beta_ini[i - 1]) * 1.0 / NL
    return D1, D2


def MatrixD_position(beta_ini, Xini, Yini, L):
    D1 = torch.zeros((NL + 1, NL), dtype=torch.double, device=device)
    D2 = torch.ones((NL + 1, 1), dtype=torch.double, device=device) * Yini
    D1x = torch.zeros((NL + 1, NL), dtype=torch.double, device=device)
    D2x = torch.ones((NL + 1, 1), dtype=torch.double, device=device) * Xini
    for i in range(NL + 1):
        if i > 0:
            D1[i, :i] = torch.sin(beta_ini[:i]) * L
            D1x[i, :i] = torch.cos(beta_ini[:i]) * L
    return D1, D2, D1x, D2x


def _block_diag_two(left, right):
    output = torch.zeros(
        (left.shape[0] + right.shape[0], left.shape[1] + right.shape[1]),
        dtype=torch.double,
        device=device,
    )
    output[: left.shape[0], : left.shape[1]] = left
    output[left.shape[0] :, left.shape[1] :] = right
    return output


def build_dual_Q_total(Q1, Q2):
    point_num = int(Q1.shape[0] // 2)
    state_dim_1 = int(Q1.shape[1])
    state_dim_2 = int(Q2.shape[1])

    # 单体 Q 的行顺序是 [body_x, body_z]。
    # 双体耦合时，B_all 的平面列顺序应为 [body1_x, body2_x, body1_z, body2_z]，
    # 因此这里不能直接 block-diag(Q1, Q2)，而要显式重排成同一套基底。
    output = torch.zeros((point_num * 4, state_dim_1 + state_dim_2), dtype=torch.double, device=device)
    output[0:point_num, 0:state_dim_1] = Q1[0:point_num, :]
    output[point_num : point_num * 2, state_dim_1 : state_dim_1 + state_dim_2] = Q2[0:point_num, :]
    output[point_num * 2 : point_num * 3, 0:state_dim_1] = Q1[point_num:, :]
    output[point_num * 3 : point_num * 4, state_dim_1 : state_dim_1 + state_dim_2] = Q2[point_num:, :]
    return output


def _initial_single(x, w, x_first, n_segments, n_links):
    segment_length = 1.0 / n_segments
    e = segment_length * 0.1

    Xini = float(x_first[0])
    Yini = float(x_first[1])

    beta_ini = torch.tensor(x[2:].copy(), dtype=torch.double, device=device)
    NI = np.zeros(n_links, dtype=np.double)
    for i in range(n_links):
        NI[i] = int(int(n_segments / n_links) * (i + 1) - 1)
        if i > 0:
            beta_ini[i] += beta_ini[i - 1]

    theta = torch.zeros((n_segments + 2), dtype=torch.double, device=device)
    forQp = torch.ones((n_segments + 2), dtype=torch.double, device=device)
    forQp[0] = Xini
    forQp[1] = Yini
    theta[0] = Xini
    theta[1] = Yini

    ratio = int(n_segments / n_links)
    for i in range(n_segments):
        theta[i + 2] = beta_ini[int(i / ratio)]

    if n_segments == N:
        Q, Qu, Q1, Ql, Q2 = MatrixQp(segment_length, theta)
    else:
        Q, Qu, Q1, Ql, Q2 = MatrixQp_dense(segment_length, theta)

    positions = torch.matmul(Q, forQp).reshape(-1, 2)

    absU = w.copy()
    action = absU.copy()
    for i in range(n_links):
        if 0 < i < n_links - 1:
            absU[i] += absU[i - 1]

    action_absolute = torch.zeros((n_segments - 1), dtype=torch.double, device=device)
    for i in range(n_links - 1):
        a = int(NI[i])
        if i == n_links - 2:
            action_absolute[a:] = absU[i]
        else:
            b = int(NI[i + 1])
            action_absolute[a:b] = absU[i]

    return {
        "L": segment_length,
        "e": e,
        "positions": positions,
        "theta": theta,
        "action_absolute": action_absolute,
        "Q": Q,
        "Qu": Qu,
        "Q1": Q1,
        "Ql": Ql,
        "Q2": Q2,
        "action": action,
        "beta_ini": beta_ini,
        "absU": absU,
        "Xini": Xini,
        "Yini": Yini,
    }


def build_match_points(dense_positions):
    # 沿用单机器人分支的“稀疏力点 <-> 邻近 dense 采样点”对应关系，
    # 只是把 dense 点坐标换成当前机器人几何下的真实位置。
    x_match = torch.zeros((FORCE_POINT_NUM, MATCH_POINT_NUM), dtype=torch.double, device=device)
    z_match = torch.zeros((FORCE_POINT_NUM, MATCH_POINT_NUM), dtype=torch.double, device=device)
    valid_mask = torch.zeros((FORCE_POINT_NUM, MATCH_POINT_NUM), dtype=torch.double, device=device)

    dense_x = dense_positions[:, 0]
    dense_z = dense_positions[:, 1]

    for sparse_idx in range(FORCE_POINT_NUM):
        selected = Label_Matrix_fila[:, sparse_idx].to(torch.bool)
        count = int(Min_Distance_num_fila[sparse_idx].item())
        if count <= 0:
            continue
        x_match[sparse_idx, :count] = dense_x[selected][:count]
        z_match[sparse_idx, :count] = dense_z[selected][:count]
        valid_mask[sparse_idx, :count] = Min_Distance_Label_fila[sparse_idx, :count]

    return x_match, z_match, valid_mask


def build_joint_stokeslet_matrix(force_points, x_match, z_match, valid_mask, e):
    # 两个机器人共享同一套流体线性系统。
    # 这里不再拆成两个独立求解，而是直接把两条链条的离散点并在一起做一次联合 Stokeslet 求解。
    point_num = force_points.shape[0]

    force_x = force_points[:, 0].view(-1, 1, 1)
    force_z = force_points[:, 1].view(-1, 1, 1)
    target_x = x_match.view(1, point_num, MATCH_POINT_NUM)
    target_z = z_match.view(1, point_num, MATCH_POINT_NUM)
    valid = valid_mask.view(1, point_num, MATCH_POINT_NUM)

    delta_x = force_x - target_x
    delta_z = force_z - target_z
    delta_y = torch.zeros_like(delta_x)

    R = torch.sqrt(delta_x**2 + delta_y**2 + delta_z**2 + e**2)
    RD = 1.0 / R
    RD3 = RD**3
    e2 = e**2

    s00 = (RD + e2 * RD3 + delta_x * delta_x * RD3) * valid
    sxz = (delta_x * delta_z * RD3) * valid
    syy = (RD + e2 * RD3) * valid
    szz = (RD + e2 * RD3 + delta_z * delta_z * RD3) * valid

    s00 = torch.sum(s00, dim=2)
    sxz = torch.sum(sxz, dim=2)
    syy = torch.sum(syy, dim=2)
    szz = torch.sum(szz, dim=2)

    A = torch.zeros((point_num * 3, point_num * 3), dtype=torch.double, device=device)
    # 必须保持与单体原始求解器一致的块顺序：[x, z, y_unused]。
    # 虽然当前任务只在 x-z 平面运动，但第三个分量的位置仍要保留原顺序，
    # 否则 B_all / Q_total 的平面块就会与 A 的列基底错位。
    A[0:point_num, 0:point_num] = s00
    A[0:point_num, point_num : point_num * 2] = sxz
    A[point_num : point_num * 2, 0:point_num] = sxz
    A[point_num : point_num * 2, point_num : point_num * 2] = szz
    A[point_num * 2 : point_num * 3, point_num * 2 : point_num * 3] = syy
    return A / (8 * math.pi * MU)


def build_dual_B_all(B1, B2, total_points):
    force_points_per_body = int(B1.shape[1] // 2)
    B_planar = torch.zeros((6, total_points * 2), dtype=torch.double, device=device)

    # ?? B ????? 3 ??????[Fx, Fz, torque]?
    # ?? B ????? [body_x, body_z]?
    #
    # ?????????????????????
    #   rows 0:3 -> ??? 1 ? 3 ?????
    #   rows 3:6 -> ??? 2 ? 3 ?????
    #
    # ????????????????
    #   [body1_x, body2_x, body1_z, body2_z, body_y_unused]
    #
    # ?????????? 3 ???????????? x/z ????
    # ??? body1_x/body2_x ??????body1_z/body2_z ???????????????????
    # ??????? M ???? 3 ????
    B1_x = B1[:, :force_points_per_body]
    B1_z = B1[:, force_points_per_body:]
    B2_x = B2[:, :force_points_per_body]
    B2_z = B2[:, force_points_per_body:]

    B_planar[0:3, 0:force_points_per_body] = B1_x
    B_planar[0:3, total_points : total_points + force_points_per_body] = B1_z

    B_planar[3:6, force_points_per_body:total_points] = B2_x
    B_planar[3:6, total_points + force_points_per_body : total_points * 2] = B2_z

    B_supply = torch.zeros((6, total_points), dtype=torch.double, device=device)
    return torch.cat((B_planar, B_supply), dim=1)


def Calculate_velocity_dual(x1, w1, x_first1, x2, w2, x_first2):
    # 先分别构造两条链条的几何与运动学矩阵，再在流体层面做联合求解。
    body1 = _initial_single(x1, w1, x_first1, N, NL)
    body2 = _initial_single(x2, w2, x_first2, N, NL)
    dense1 = _initial_single(x1, w1, x_first1, N_dense, NL)
    dense2 = _initial_single(x2, w2, x_first2, N_dense, NL)

    force_points_all = torch.cat((body1["positions"], body2["positions"]), dim=0)
    x_match_1, z_match_1, valid_1 = build_match_points(dense1["positions"])
    x_match_2, z_match_2, valid_2 = build_match_points(dense2["positions"])

    x_match_all = torch.cat((x_match_1, x_match_2), dim=0)
    z_match_all = torch.cat((z_match_1, z_match_2), dim=0)
    valid_all = torch.cat((valid_1, valid_2), dim=0)

    A = build_joint_stokeslet_matrix(force_points_all, x_match_all, z_match_all, valid_all, body1["e"])

    B1 = MatrixB(body1["L"], body1["theta"], body1["positions"])
    B2 = MatrixB(body2["L"], body2["theta"], body2["positions"])
    B_all = build_dual_B_all(B1, B2, force_points_all.shape[0])

    # 单体正式求解时，进入约化刚体系统的是 MatrixQ(...) 重排后的 Q，
    # 不是 MatrixQp/MatrixQp_dense 直接返回的原始堆叠结果。
    # 双体这里也必须沿用同一语义，否则 MT = AB @ Q_total 的列意义会再次错位。
    Q_single_1 = MatrixQ(body1["L"], body1["theta"], body1["Qu"], body1["Q1"], body1["Ql"], body1["Q2"])
    Q_single_2 = MatrixQ(body2["L"], body2["theta"], body2["Qu"], body2["Q1"], body2["Ql"], body2["Q2"])
    Q_total = build_dual_Q_total(Q_single_1, Q_single_2)

    C1_1, C2_1 = MatrixC(body1["action_absolute"])
    C1_2, C2_2 = MatrixC(body2["action_absolute"])
    C1_total = _block_diag_two(C1_1, C1_2)
    C2_total = torch.cat((C2_1, C2_2), dim=0)

    AB = torch.linalg.solve(A.T, B_all.T).T
    AB = AB[:, : Q_total.shape[0]]

    MT = torch.matmul(AB, Q_total)
    M = torch.matmul(MT, C1_total)
    R = -torch.matmul(MT, C2_total)
    rigid_vel = torch.linalg.solve(M, R).view(-1)

    rigid_vel_1 = rigid_vel[:3].view(-1, 1)
    rigid_vel_2 = rigid_vel[3:].view(-1, 1)

    D1_1, D2_1 = MatrixD_sum(body1["beta_ini"], body1["absU"])
    D1_2, D2_2 = MatrixD_sum(body2["beta_ini"], body2["absU"])
    veloall1 = torch.matmul(D1_1, rigid_vel_1) + D2_1
    veloall2 = torch.matmul(D1_2, rigid_vel_2) + D2_2

    # 这里返回给高层环境的是“低层 primitive 状态”的更新量，
    # 维度必须和输入的 x1/x2 保持一致（12 = 3 + 9），
    # 不能误用求解器离散数 N=40 去生成 42 维状态。
    velon1 = np.zeros_like(x1, dtype=np.float64)
    velon2 = np.zeros_like(x2, dtype=np.float64)

    veloall1_np = np.squeeze(veloall1.detach().cpu().numpy())
    veloall2_np = np.squeeze(veloall2.detach().cpu().numpy())
    rigid_vel_np = rigid_vel.detach().cpu().numpy()

    velon1[0] = np.mean(veloall1_np[::2]) * 1.1
    velon1[1] = np.mean(veloall1_np[1::2]) * 1.1
    velon1[2] = rigid_vel_np[2]
    velon1[3:] = body1["action"]

    velon2[0] = np.mean(veloall2_np[::2]) * 1.1
    velon2[1] = np.mean(veloall2_np[1::2]) * 1.1
    velon2[2] = rigid_vel_np[5]
    velon2[3:] = body2["action"]

    Xp1 = np.squeeze(body1["positions"][:, 0].detach().cpu().numpy())
    Yp1 = np.squeeze(body1["positions"][:, 1].detach().cpu().numpy())
    Xp2 = np.squeeze(body2["positions"][:, 0].detach().cpu().numpy())
    Yp2 = np.squeeze(body2["positions"][:, 1].detach().cpu().numpy())

    pressure_all = np.zeros((force_points_all.shape[0],), dtype=np.float64)
    pressure_diff = 0.0
    pressure_end = 0.0

    return (
        velon1,
        rigid_vel_np[:3],
        Xp1,
        Yp1,
        velon2,
        rigid_vel_np[3:],
        Xp2,
        Yp2,
        pressure_diff,
        pressure_end,
        pressure_all,
    )


def RK_dual(x1, w1, x_first1, x2, w2, x_first2):
    # 与单机器人保持一致：每个底层 step 内部再做 10 次数值子步。
    Xn1 = 0.0
    Xn2 = 0.0
    Yn1 = 0.0
    Yn2 = 0.0
    r1 = 0.0
    r2 = 0.0

    xc1 = x1.copy()
    xc2 = x2.copy()
    x_first_delta1 = np.zeros((2,), dtype=np.float64)
    x_first_delta2 = np.zeros((2,), dtype=np.float64)
    x_fc1 = x_first1.copy()
    x_fc2 = x_first2.copy()

    Ntime = 10
    whole_time = 0.2
    part_time = whole_time / Ntime

    Xp1 = np.zeros((NL + 1,), dtype=np.float64)
    Yp1 = np.zeros((NL + 1,), dtype=np.float64)
    Xp2 = np.zeros((NL + 1,), dtype=np.float64)
    Yp2 = np.zeros((NL + 1,), dtype=np.float64)

    for _ in range(Ntime):
        (
            V1,
            Vo1,
            Xp1,
            Yp1,
            V2,
            Vo2,
            Xp2,
            Yp2,
            pressure_diff,
            pressure_end,
            pressure_all,
        ) = Calculate_velocity_dual(xc1, w1, x_fc1, xc2, w2, x_fc2)

        # 叠加背景涡流（在质心与 x_first 上分别评估）
        bgx1, bgy1 = vortex_velocity(xc1[0], xc1[1])
        bgx2, bgy2 = vortex_velocity(xc2[0], xc2[1])
        V1[0] += bgx1; V1[1] += bgy1
        V2[0] += bgx2; V2[1] += bgy2
        bfx1, bfy1 = vortex_velocity(x_fc1[0], x_fc1[1])
        bfx2, bfy2 = vortex_velocity(x_fc2[0], x_fc2[1])
        Vo1[0] += bfx1; Vo1[1] += bfy1
        Vo2[0] += bfx2; Vo2[1] += bfy2

        k1_1 = part_time * V1
        k1_2 = part_time * V2

        # RK2 中点：在半步推进后的位置重新评估 Stokeslet 与背景涡流
        x_fc1_mid = x_fc1 + 0.5 * part_time * Vo1[:2]
        x_fc2_mid = x_fc2 + 0.5 * part_time * Vo2[:2]
        xc1_mid = xc1 + 0.5 * k1_1
        xc2_mid = xc2 + 0.5 * k1_2

        (
            V1,
            Vo1,
            Xp1,
            Yp1,
            V2,
            Vo2,
            Xp2,
            Yp2,
            pressure_diff,
            pressure_end,
            pressure_all,
        ) = Calculate_velocity_dual(
            xc1_mid,
            w1,
            x_fc1_mid,
            xc2_mid,
            w2,
            x_fc2_mid,
        )

        bgx1m, bgy1m = vortex_velocity(xc1_mid[0], xc1_mid[1])
        bgx2m, bgy2m = vortex_velocity(xc2_mid[0], xc2_mid[1])
        V1[0] += bgx1m; V1[1] += bgy1m
        V2[0] += bgx2m; V2[1] += bgy2m
        bfx1m, bfy1m = vortex_velocity(x_fc1_mid[0], x_fc1_mid[1])
        bfx2m, bfy2m = vortex_velocity(x_fc2_mid[0], x_fc2_mid[1])
        Vo1[0] += bfx1m; Vo1[1] += bfy1m
        Vo2[0] += bfx2m; Vo2[1] += bfy2m

        k2_1 = part_time * V1
        k2_2 = part_time * V2

        xc1 += k2_1
        xc2 += k2_2
        xc1[2] = (xc1[2] + math.pi) % (2 * math.pi) - math.pi
        xc2[2] = (xc2[2] + math.pi) % (2 * math.pi) - math.pi

        Xn1 += k2_1[0]
        Yn1 += k2_1[1]
        Xn2 += k2_2[0]
        Yn2 += k2_2[1]
        r1 += k2_1[0] / whole_time
        r2 += k2_2[0] / whole_time

        x_first_delta1 += part_time * Vo1[:2]
        x_first_delta2 += part_time * Vo2[:2]
        x_fc1 += part_time * Vo1[:2]
        x_fc2 += part_time * Vo2[:2]

    return (
        xc1,
        Xn1,
        Yn1,
        r1,
        x_first_delta1,
        Xp1,
        Yp1,
        xc2,
        Xn2,
        Yn2,
        r2,
        x_first_delta2,
        Xp2,
        Yp2,
        0.0,
        0.0,
        np.zeros((FORCE_POINT_NUM * 2,), dtype=np.float64),
    )


def compute_stokeslet_forces(x1, w1, x_first1, x2, w2, x_first2):
    """Compute Stokeslet force distribution for current robot configuration and motion."""
    body1 = _initial_single(x1, w1, x_first1, N, NL)
    body2 = _initial_single(x2, w2, x_first2, N, NL)
    dense1 = _initial_single(x1, w1, x_first1, N_dense, NL)
    dense2 = _initial_single(x2, w2, x_first2, N_dense, NL)

    force_points_all = torch.cat((body1["positions"], body2["positions"]), dim=0)
    total_points = force_points_all.shape[0]

    x_match_1, z_match_1, valid_1 = build_match_points(dense1["positions"])
    x_match_2, z_match_2, valid_2 = build_match_points(dense2["positions"])
    x_match_all = torch.cat((x_match_1, x_match_2), dim=0)
    z_match_all = torch.cat((z_match_1, z_match_2), dim=0)
    valid_all = torch.cat((valid_1, valid_2), dim=0)

    A = build_joint_stokeslet_matrix(
        force_points_all, x_match_all, z_match_all, valid_all, body1["e"]
    )

    B1_mat = MatrixB(body1["L"], body1["theta"], body1["positions"])
    B2_mat = MatrixB(body2["L"], body2["theta"], body2["positions"])
    B_all = build_dual_B_all(B1_mat, B2_mat, total_points)

    Q_single_1 = MatrixQ(
        body1["L"], body1["theta"], body1["Qu"], body1["Q1"], body1["Ql"], body1["Q2"]
    )
    Q_single_2 = MatrixQ(
        body2["L"], body2["theta"], body2["Qu"], body2["Q1"], body2["Ql"], body2["Q2"]
    )
    Q_total = build_dual_Q_total(Q_single_1, Q_single_2)

    C1_1, C2_1 = MatrixC(body1["action_absolute"])
    C1_2, C2_2 = MatrixC(body2["action_absolute"])
    C1_total = _block_diag_two(C1_1, C1_2)
    C2_total = torch.cat((C2_1, C2_2), dim=0)

    AB = torch.linalg.solve(A.T, B_all.T).T
    AB = AB[:, : Q_total.shape[0]]

    MT = torch.matmul(AB, Q_total)
    M_mat = torch.matmul(MT, C1_total)
    R_vec = -torch.matmul(MT, C2_total)
    rigid_vel = torch.linalg.solve(M_mat, R_vec).view(-1)

    v_gen = C1_total @ rigid_vel.view(-1, 1) + C2_total
    u_surf = Q_total @ v_gen

    u_full = torch.zeros(3 * total_points, 1, dtype=torch.double, device=device)
    u_full[: u_surf.shape[0]] = u_surf
    forces = torch.linalg.solve(A, u_full).view(-1)

    fp_x = force_points_all[:, 0].detach().cpu().numpy().copy()
    fp_y = force_points_all[:, 1].detach().cpu().numpy().copy()
    f_x = forces[:total_points].detach().cpu().numpy().copy()
    f_y = forces[total_points : 2 * total_points].detach().cpu().numpy().copy()
    e_val = float(body1["e"])

    return fp_x, fp_y, f_x, f_y, e_val


def evaluate_stokeslet_velocity(grid_x, grid_y, fp_x, fp_y, f_x, f_y, e_val):
    """Evaluate Stokeslet-induced fluid velocity at grid points given force distribution."""
    fp_x_3d = fp_x.reshape(-1, 1, 1)
    fp_y_3d = fp_y.reshape(-1, 1, 1)
    f_x_3d = f_x.reshape(-1, 1, 1)
    f_y_3d = f_y.reshape(-1, 1, 1)

    dx = grid_x[np.newaxis, :, :] - fp_x_3d
    dy = grid_y[np.newaxis, :, :] - fp_y_3d

    R = np.sqrt(dx ** 2 + dy ** 2 + e_val ** 2)
    R_inv = 1.0 / R
    R3_inv = R_inv ** 3
    e2 = e_val ** 2

    ux = np.sum(
        (R_inv + e2 * R3_inv + dx * dx * R3_inv) * f_x_3d
        + (dx * dy * R3_inv) * f_y_3d,
        axis=0,
    )
    uy = np.sum(
        (dx * dy * R3_inv) * f_x_3d
        + (R_inv + e2 * R3_inv + dy * dy * R3_inv) * f_y_3d,
        axis=0,
    )

    ux /= 8 * math.pi * MU
    uy /= 8 * math.pi * MU

    return ux, uy
