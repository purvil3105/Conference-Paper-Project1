"""Draw the paper's two flowcharts as vector PDFs (IEEE single-column width).

Fig. 1 follows the request statuses in nbiot_rach/ledger.py and the transitions
in nbiot_rach/environment.py. Fig. 2 follows the ten steps of Algorithm 1 in
main.tex. If either source changes, update this file and rerun it:

    python paper/figures/make_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

HERE = Path(__file__).parent
LINE = 0.7
GRAY = "#e4e4e4"


def box(ax, cx, cy, w, h, text, fill="white", dashed=False):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fill,
                           edgecolor="black", linewidth=LINE,
                           linestyle="--" if dashed else "-"))
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.05)


def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy),
                          (cx, cy - h / 2)], closed=True, facecolor="white",
                         edgecolor="black", linewidth=LINE))
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.05)


def arrow(ax, *points):
    """Polyline through ``points`` with an arrowhead on the last segment."""
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="black", linewidth=LINE,
                solid_capstyle="projecting")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=7, linewidth=LINE, color="black",
                                 shrinkA=0, shrinkB=0))


def label(ax, x, y, text, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=8, fontstyle="italic")


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png")
    plt.close(fig)


def request_life_cycle():
    fig, ax = canvas(3.5, 2.45)
    left, mid, right = 0.66, 1.78, 3.02
    top, centre, low = 2.17, 1.3, 0.34

    box(ax, left, top, 0.62, 0.3, "Arrival")
    box(ax, left, centre, 0.92, 0.42, "WAITING/\nBARRED")
    diamond(ax, mid, centre, 0.96, 0.82, "Singleton,\nRAR and\nMsg3?")
    box(ax, right, centre, 0.9, 0.42, "ACCESS IN\nPROGRESS")
    box(ax, right, top, 0.9, 0.3, "COMPLETED", fill=GRAY)
    diamond(ax, mid, low, 0.96, 0.5, "Tries < 10?")
    box(ax, left, low, 0.86, 0.3, "BACKOFF")
    box(ax, right, low, 0.8, 0.3, "DROPPED", fill=GRAY)

    arrow(ax, (left, top - 0.15), (left, centre + 0.21))
    arrow(ax, (left + 0.46, centre), (mid - 0.48, centre))
    label(ax, 1.21, centre + 0.12, r"$p_A$")
    arrow(ax, (mid + 0.48, centre), (right - 0.45, centre))
    label(ax, 2.41, centre + 0.11, "yes")
    arrow(ax, (right, centre + 0.21), (right, top - 0.15))
    label(ax, right + 0.06, 1.83, "112 ms", ha="left")
    arrow(ax, (mid, centre - 0.41), (mid, low + 0.25))
    label(ax, mid + 0.05, 0.75, "no", ha="left")
    arrow(ax, (mid - 0.48, low), (left + 0.43, low))
    label(ax, 1.2, low + 0.11, "yes")
    arrow(ax, (mid + 0.48, low), (right - 0.4, low))
    label(ax, 2.44, low + 0.11, "no")
    arrow(ax, (left, low + 0.15), (left, centre - 0.21))
    label(ax, left - 0.06, 0.72, "timer expires", ha="right")
    # barred requests stay WAITING and are redrawn at the next opportunity
    arrow(ax, (left - 0.46, centre + 0.1), (0.08, centre + 0.1), (0.08, centre + 0.38),
          (left - 0.25, centre + 0.38), (left - 0.25, centre + 0.21))
    label(ax, 0.2, centre + 0.5, r"$1-p_A$", ha="left")
    save(fig, "fig_lifecycle")


def control_loop():
    fig, ax = canvas(3.5, 3.45)
    left, right = 1.0, 2.62
    w, h = 1.32, 0.34
    rows = [2.98, 2.56, 2.14, 1.72, 1.3, 0.88, 0.46]

    ax.text(left, 3.32, "NB-IoT cell (simulator)", ha="center", fontweight="bold")
    ax.text(right, 3.32, "Controller at the eNB", ha="center", fontweight="bold")

    box(ax, left, rows[0], w, h, "(1) Create arrivals")
    box(ax, left, rows[1], w, h, "(2) Process deadlines")
    box(ax, left, rows[2], w, h, "(3) Release requests\nin backoff")
    box(ax, right, rows[2], w, h, "(4) Observe and\nencode state $s$")
    box(ax, right, rows[3], w, h, "(5) Select feasible\naction $a$")
    box(ax, right, rows[4], w, h, "(6) Apply at next\nNPRACH boundary")
    box(ax, left, rows[4], w, h, "(7) Contention, RAR,\nMsg3, resolution")
    box(ax, left, rows[5], w, h, "(8) Record outcomes\nand resource use")
    box(ax, right, rows[5], w, h, "(9) Compute reward $r$")
    box(ax, right, rows[6], w, h, "(10) Observe next\nstate; update $Q$", dashed=True)

    half_w, half_h = w / 2, h / 2
    arrow(ax, (left, rows[0] - half_h), (left, rows[1] + half_h))
    arrow(ax, (left, rows[1] - half_h), (left, rows[2] + half_h))
    arrow(ax, (left + half_w, rows[2]), (right - half_w, rows[2]))
    arrow(ax, (right, rows[2] - half_h), (right, rows[3] + half_h))
    arrow(ax, (right, rows[3] - half_h), (right, rows[4] + half_h))
    arrow(ax, (right - half_w, rows[4]), (left + half_w, rows[4]))
    arrow(ax, (left, rows[4] - half_h), (left, rows[5] + half_h))
    arrow(ax, (left + half_w, rows[5]), (right - half_w, rows[5]))
    arrow(ax, (right, rows[5] - half_h), (right, rows[6] + half_h))
    # next decision window: back to step (1)
    arrow(ax, (right - half_w, rows[6]), (0.12, rows[6]), (0.12, rows[0]),
          (left - half_w, rows[0]))
    label(ax, 0.2, 0.34, "Next 640 ms decision window", ha="left")
    save(fig, "fig_control_loop")


if __name__ == "__main__":
    request_life_cycle()
    control_loop()
