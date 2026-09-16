"""
============================================================
MPC PLANNER -- shared by drive_brain_mpc.py and
                simulation_trajectory_obstacles.py
============================================================
A sampling-based ("random shooting") receding-horizon MPC:

  1. Sample a batch of candidate control sequences (steering, accel)
     around a warm-started previous solution.
  2. Roll every sequence forward through a kinematic bicycle model.
  3. Score every rollout with a cost function you supply (collision
     cost + reference-tracking cost + control-effort cost).
  4. Keep the best sequence, apply its FIRST control, re-plan next
     step (that's what makes it "receding horizon").

Why this style instead of a gradient/QP solver: zero external solver
dependency (just numpy), trivially real-time at small horizons, and
it fails gracefully -- a rough frame just returns a slightly
worse-but-still-collision-checked rollout instead of a solver
divergence. This is intentionally the same conceptual family as your
iMPC research, simplified so it runs on a Jetson Nano every frame.

Both consumers of this file share the same convention: state is
(x, y, theta, v) and obstacles are only ever things you tell the cost
function about -- if you don't pass it, the planner literally cannot
see it. That's what enforces "only the perception window / sensor
range counts" at the architecture level, not just as a comment.
============================================================
"""

import numpy as np


class KinematicBicycle:
    """states: x, y, theta, v.  controls: steering angle (rad), accel."""

    def __init__(self, wheelbase, max_steer, max_speed, min_speed=0.0):
        self.L = wheelbase
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.min_speed = min_speed


class SamplingMPC:
    """Vectorized (all samples rolled out together) receding-horizon planner."""

    def __init__(self, model, horizon=8, dt=0.1, num_samples=150,
                 steer_std=0.20, accel_std=0.35, seed=None):
        self.model = model
        self.horizon = horizon
        self.dt = dt
        self.num_samples = max(num_samples, 4)
        self.steer_std = steer_std
        self.accel_std = accel_std
        self.rng = np.random.default_rng(seed)
        self.prev_best_seq = None  # warm start

    def _sample_sequences(self):
        H, N = self.horizon, self.num_samples
        if self.prev_best_seq is not None:
            base = np.vstack([self.prev_best_seq[1:], np.zeros((1, 2))])
        else:
            base = np.zeros((H, 2))

        steer_noise = self.rng.normal(0, self.steer_std, size=(N, H))
        accel_noise = self.rng.normal(0, self.accel_std, size=(N, H))
        seqs = np.stack([base[:, 0] + steer_noise, base[:, 1] + accel_noise], axis=-1)
        seqs[0] = base                # pure warm-started guess
        seqs[1] = np.zeros((H, 2))    # coast / do-nothing, always an option
        return seqs

    def rollout(self, state, seqs):
        """seqs: (N, H, 2) -> traj: (N, H+1, 4)"""
        N, H, _ = seqs.shape
        states = np.tile(np.array(state, dtype=float), (N, 1))  # (N, 4)
        traj = np.zeros((N, H + 1, 4))
        traj[:, 0, :] = states

        m = self.model
        for t in range(H):
            x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
            steer = np.clip(seqs[:, t, 0], -m.max_steer, m.max_steer)
            accel = seqs[:, t, 1]
            v = np.clip(v + accel * self.dt, m.min_speed, m.max_speed)
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            theta = theta + (v / m.L) * np.tan(steer) * self.dt
            states = np.stack([x, y, theta, v], axis=1)
            traj[:, t + 1, :] = states
        return traj

    def plan(self, state, cost_fn):
        """cost_fn(traj (N,H+1,4), seqs (N,H,2)) -> costs (N,)."""
        seqs = self._sample_sequences()
        traj = self.rollout(state, seqs)
        costs = cost_fn(traj, seqs)
        best_idx = int(np.argmin(costs))
        best_seq = seqs[best_idx]
        self.prev_best_seq = best_seq
        return best_seq[0], traj[best_idx], float(costs[best_idx])

    def reset_warm_start(self):
        self.prev_best_seq = None


def nearest_ref_point_cost(points_xy):
    """points_xy: (K, 2) reference path/corridor points.
    Returns fn(xy_batch (..., 2)) -> squared distance to nearest ref point,
    broadcastable over any leading shape. K==0 -> always 0 (no reference)."""
    ref = np.asarray(points_xy, dtype=float)
    if ref.shape[0] == 0:
        return lambda xy: np.zeros(xy.shape[:-1])

    def fn(xy):
        diff = xy[..., None, :] - ref[None, :, :]   # (..., K, 2)
        d2 = np.sum(diff ** 2, axis=-1)              # (..., K)
        return np.min(d2, axis=-1)
    return fn
