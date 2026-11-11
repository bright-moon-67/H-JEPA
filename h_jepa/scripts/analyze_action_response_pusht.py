"""Quick physical audit of paired outcomes (NOT a training label)."""
import argparse
import numpy as np


def wrapped_angle_delta(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()

    with np.load(args.data, allow_pickle=False) as d:
        s_a = d["state_a"].copy()
        s_b = d["state_b"].copy()
        modes = d["mode"].copy()
        contact_a = d["contacts_a"].copy()
        contact_b = d["contacts_b"].copy()
        a = d["acts_a"].copy()
        b = d["acts_b"].copy()

    # The physical state is [agent_xy(2), block_xy(2), block_angle(1), agent_vel(2)].
    delta_agent = np.linalg.norm(s_a[:, :2] - s_b[:, :2], axis=1)
    delta_block = np.linalg.norm(s_a[:, 2:4] - s_b[:, 2:4], axis=1)
    delta_theta = np.degrees(
        np.abs(wrapped_angle_delta(s_a[:, 4], s_b[:, 4]))
    )
    n = len(modes)
    print(f"pairs={n}, all actions different={np.all(np.any(a != b, axis=(1,2)))}")
    print(f"same actions in every anchor={bool(np.all(a == a[0]) and np.all(b == b[0]))}")
    for mode in np.unique(modes):
        m = modes == mode
        contact = (contact_a[m] + contact_b[m]) > 0
        print(
            f"{str(mode):>9s}: n={m.sum():3d}, contact={contact.mean():.1%}, "
            f"agent-effect={delta_agent[m].mean():7.2f}px, "
            f"block-effect={delta_block[m].mean():7.2f}px, "
            f"rotation-effect={delta_theta[m].mean():6.2f}deg"
        )


if __name__ == "__main__":
    main()
