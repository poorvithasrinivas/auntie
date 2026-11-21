import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def render_specular_fallback(
    ns: float,
    output_path,
    size: int = 900,
) -> None:
    """Render a compact Phong-like highlight as a local, Blender-independent preview."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=size / 6)
    ax.set_facecolor("#111827")
    ax.set_aspect("equal")
    ax.axis("off")

    y, x = np.mgrid[-1:1:900j, -1:1:900j]
    radius = np.sqrt(x * x + y * y)
    normal_z = np.sqrt(np.clip(1 - radius * radius, 0, 1))
    view_x, view_y, view_z = 0.25, 0.35, 0.906
    light_x, light_y, light_z = 0.0, 0.0, 1.0
    reflected_z = 2 * normal_z * normal_z - 1
    dot = np.clip(
        x * 0.25 + y * 0.35 + normal_z * reflected_z,
        0,
        1,
    )
    specular = dot ** ns
    sphere = np.where(radius <= 1, specular * 0.82 + normal_z * 0.12, 0)

    ax.imshow(sphere, extent=(-1, 1, -1, 1), origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#9ca3af", linewidth=2))
    ax.text(
        0,
        -1.12,
        f"Phong n_s = {ns:.1f}",
        ha="center",
        va="top",
        color="white",
        fontsize=15,
    )
    fig.patch.set_facecolor("#111827")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
