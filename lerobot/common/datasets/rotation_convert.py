import torch
import torch.nn.functional as F
import math

# -----------------------
# Base Utils
# -----------------------
def _as_tensor(x, dtype=None, device=None):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype, device=device)
    return x

# -----------------------
# 6D <-> matrix
# -----------------------
def ortho6d_to_matrix(ortho6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    ortho6d: (..., 6) where first 3 are column0, next 3 are column1
    returns: (..., 3, 3) with columns [x, y, z]
    """
    ortho6d = _as_tensor(ortho6d)
    if ortho6d.shape[-1] != 6:
        raise ValueError("last dim must be 6")
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = F.normalize(x_raw, p=2, dim=-1, eps=eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, p=2, dim=-1, eps=eps)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)  # (...,3,3) columns are x,y,z
    return R

def matrix_to_ortho6d(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3)
    returns: (..., 6) with concat of first two columns [col0, col1]
    """
    R = _as_tensor(R)
    if R.shape[-2:] != (3,3):
        raise ValueError("R must have shape (...,3,3)")
    col0 = R[..., :, 0]  # (...,3)
    col1 = R[..., :, 1]
    return torch.cat([col0, col1], dim=-1)  # (...,6)

# -----------------------
# Quaternion <-> matrix
# -----------------------
def quaternion_to_matrix(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q: (...,4) in (x, y, z, w) format
    returns: (...,3,3)
    """
    q = _as_tensor(q)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must have last dim = 4 (x, y, z, w)")

    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))  # normalize
    x, y, z, w = q.unbind(dim=-1)  # each (...,)

    # compute matrix elements
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R00 = ww + xx - yy - zz
    R01 = 2 * (xy - wz)
    R02 = 2 * (xz + wy)

    R10 = 2 * (xy + wz)
    R11 = ww - xx + yy - zz
    R12 = 2 * (yz - wx)

    R20 = 2 * (xz - wy)
    R21 = 2 * (yz + wx)
    R22 = ww - xx - yy + zz

    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)  # (...,3,3)
    return R

def matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: (...,3,3)
    returns: (...,4) as (x, y, z, w)
    Robust numerically using standard branch on trace.
    """
    R = _as_tensor(R)
    if R.shape[-2:] != (3,3):
        raise ValueError("R must have shape (...,3,3)")

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    # prepare containers
    w = torch.empty_like(trace)
    x = torch.empty_like(trace)
    y = torch.empty_like(trace)
    z = torch.empty_like(trace)

    # case trace > 0
    pos = trace > 0
    if pos.any():
        s = torch.sqrt(trace[pos] + 1.0) * 2.0  # s = 4*w
        w[pos] = 0.25 * s
        x[pos] = (R[pos][...,2,1] - R[pos][...,1,2]) / s
        y[pos] = (R[pos][...,0,2] - R[pos][...,2,0]) / s
        z[pos] = (R[pos][...,1,0] - R[pos][...,0,1]) / s

    # other cases: determine which diagonal is biggest
    not_pos = ~pos
    if not_pos.any():
        # create view of sub-block
        m00_n = m00[not_pos]
        m11_n = m11[not_pos]
        m22_n = m22[not_pos]
        Rn = R[not_pos]

        cond_x = (m00_n > m11_n) & (m00_n > m22_n)
        cond_y = (m11_n > m22_n) & (~cond_x)
        cond_z = ~(cond_x | cond_y)

        # X biggest
        if cond_x.any():
            s = torch.sqrt(1.0 + m00_n[cond_x] - m11_n[cond_x] - m22_n[cond_x]) * 2.0
            x[not_pos][cond_x] = 0.25 * s
            w[not_pos][cond_x] = (Rn[cond_x][...,2,1] - Rn[cond_x][...,1,2]) / s
            y[not_pos][cond_x] = (Rn[cond_x][...,0,1] + Rn[cond_x][...,1,0]) / s
            z[not_pos][cond_x] = (Rn[cond_x][...,0,2] + Rn[cond_x][...,2,0]) / s

        # Y biggest
        if cond_y.any():
            s = torch.sqrt(1.0 + m11_n[cond_y] - m00_n[cond_y] - m22_n[cond_y]) * 2.0
            y[not_pos][cond_y] = 0.25 * s
            w[not_pos][cond_y] = (Rn[cond_y][...,0,2] - Rn[cond_y][...,2,0]) / s
            x[not_pos][cond_y] = (Rn[cond_y][...,0,1] + Rn[cond_y][...,1,0]) / s
            z[not_pos][cond_y] = (Rn[cond_y][...,1,2] + Rn[cond_y][...,2,1]) / s

        # Z biggest
        if cond_z.any():
            s = torch.sqrt(1.0 + m22_n[cond_z] - m00_n[cond_z] - m11_n[cond_z]) * 2.0
            z[not_pos][cond_z] = 0.25 * s
            w[not_pos][cond_z] = (Rn[cond_z][...,1,0] - Rn[cond_z][...,0,1]) / s
            x[not_pos][cond_z] = (Rn[cond_z][...,0,2] + Rn[cond_z][...,2,0]) / s
            y[not_pos][cond_z] = (Rn[cond_z][...,1,2] + Rn[cond_z][...,2,1]) / s

    q = torch.stack([x, y, z, w], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q

# -----------------------
# Euler(zyx) <-> matrix
# -----------------------
def euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    euler: (..., 3) angles (roll_x, pitch_y, yaw_z) ????
    NOTE: We assume input order is (roll, pitch, yaw) but the ZYX composition
    corresponds to R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    For clarity, the input vector is [roll, pitch, yaw] in radians.
    returns: (...,3,3)
    """
    euler = _as_tensor(euler)
    if euler.shape[-1] != 3:
        raise ValueError(f"euler must have last dim 3: (roll, pitch, yaw), get {euler.shape[-1]}")

    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = torch.cos(roll); sr = torch.sin(roll)
    cp = torch.cos(pitch); sp = torch.sin(pitch)
    cz = torch.cos(yaw); sz = torch.sin(yaw)

    # Compose R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R00 = cz * cp
    R01 = cz * sp * sr - sz * cr
    R02 = cz * sp * cr + sz * sr

    R10 = sz * cp
    R11 = sz * sp * sr + cz * cr
    R12 = sz * sp * cr - cz * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)
    return R

def matrix_to_euler(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    R: (...,3,3)
    returns: (...,3) = (roll, pitch, yaw) such that R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Handles gimbal lock approximately: when cos(pitch) ~ 0, yaw set to 0 and roll derived.
    """
    R = _as_tensor(R)
    if R.shape[-2:] != (3,3):
        raise ValueError("R must have shape (...,3,3)")

    R20 = R[..., 2, 0]
    # clamp for numerical safety
    pitch = torch.asin((-R20).clamp(-1.0, 1.0))

    cos_pitch = torch.cos(pitch)
    singular = cos_pitch.abs() < eps

    # default (non-singular) formulas
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw  = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    # handle singularities: when |cos(pitch)| ~ 0
    if singular.any():
        # for those entries, set yaw = 0 and compute roll from first row instead
        idx = singular
        # When pitch ~ +pi/2 or -pi/2, R20 = -sin(pitch) -> check sign
        # Use alternative computations to avoid division by zero.
        # Set yaw = 0, roll = atan2(-R01, R11)  (common fallback)
        roll_sing = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
        roll = torch.where(idx, roll_sing, roll)
        yaw = torch.where(idx, torch.zeros_like(yaw), yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)

# -----------------------
# Convenience wrappers
# -----------------------
def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(euler_to_matrix(euler))

def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler(quaternion_to_matrix(q))

def ortho6d_to_quaternion(ortho6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(ortho6d_to_matrix(ortho6d))

def quaternion_to_ortho6d(q: torch.Tensor) -> torch.Tensor:
    return matrix_to_ortho6d(quaternion_to_matrix(q))

def euler_to_ortho6d(euler: torch.Tensor) -> torch.Tensor:
    return matrix_to_ortho6d(euler_to_matrix(euler))

def ortho6d_to_euler(ortho6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler(ortho6d_to_matrix(ortho6d))