# Kisladag_Gold_Dr.Mutlu_Zeybek

import re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Polygon
from matplotlib.offsetbox import AnnotationBbox, DrawingArea

raw = """GC-271 687755 4261727 966 0.50 23
GC-271 687759 4261645 789 0.78 303
GC-272 687694 4261462 833 0.46 415
GC-273 687823 4261612 836 0.36 183
GC-274 687807 4261524 909 0.79 300
GC-275 687701 4261729 805 0.91 388
GC-276 687174 4261444 935 0.62 275
GC-277 686970 4261636 891 0.33 206
GC-278 687400 4261308 1013 0.43 58
GC-278 687397 4261359 821 1.08 300
GC-279 687154 4261760 919 0.56 240
GC-279 687147 4261846 749 0.44 70
GC-280 687571 4261364 868 1.11 301
GC-281 687502 4261377 836 1.70 340
GC-282 687108 4261714 866 0.80 399
GC-283 687762 4261536 853 0.84 325
GC-284 687596 4261464 850 0.40 324
GC-285 687056 4261595 838 0.70 391
GC-286 687569 4261783 989 0.22 28
GC-286 687608 4261842 873 0.57 218
GC-286 687640 4261901 738 0.23 20
GC-287 687263 4261405 936 1.01 194
GC-288 687123 4261481 1010 0.45 25
GC-288 687125 4261435 931 0.45 138
GC-289 687381 4261462 802 1.07 198
GC-290 687443 4261818 929 1.28 165
GC-291 687239 4261431 893 1.01 275
GC-292 687200 4261637 958 0.66 185
GC-292 687203 4261715 766 0.53 174
GC-293 687221 4261586 986 0.53 133
GC-293 687272 4261586 902 0.37 30
GC-294 687031 4261553 831 0.43 311
GC-295 687101 4261654 861 0.75 316
GC-296 687239 4261527 809 0.69 125
GC-297 687051 4261780 886 0.24 20
GC-297 687093 4261697 751 0.48 213
GC-298 686948 4261430 996 0.28 20
GC-298 686948 4261449 964 0.22 30
GC-298 686947 4261492 890 0.30 15
GC-298 686945 4261560 770 0.98 67
GC-299 687092 4261517 985 0.29 25
GC-299 687073 4261498 931 0.33 25
GC-299 687027 4261454 805 0.44 178
GC-303 686904 4261425 894 0.34 148
GC-303 686911 4261515 733 0.20 30
GC-303 686913 4261547 676 0.26 38
GC-303 686915 4261570 634 0.21 19
GC-304 686831 4261589 740 0.74 265
GC-313 687500 4261782 969 0.60 48
GC-313 687501 4261845 857 0.80 175
GC-317 687793 4261650 912 0.38 203
GC-317 687873 4261650 811 0.26 15
GC-320 687467 4261359 913 0.66 188
GC-320 687492 4261291 807 0.24 15
GC-321 687821 4261478 793 0.69 519
GC-322 687679 4261776 802 0.68 293
GC-323 687124 4261777 952 0.63 123
GC-324 687575 4261553 974 0.24 15
GC-324 687663 4261446 747 0.60 456
GC-325 686958 4261375 938 0.34 33
GC-325 686967 4261447 824 0.41 168
GC-325 686978 4261553 662 0.35 120
GC-327 687174 4261332 671 0.27 33
GC-327 687173 4261350 611 0.34 68
GC-327 687171 4261367 556 0.22 38
GC-328 687319 4261397 879 0.74 258
GC-328 687299 4261284 634 0.77 43
GC-329 686716 4261601 870 0.48 108
GC-329 686666 4261532 693 0.36 15
GC-329 686650 4261510 633 0.38 27
GC-330 686971 4261722 724 0.39 43
GC-330 686968 4261691 638 0.32 90
GC-331 687421 4261250 863 0.74 18
GC-333 687852 4261411 991 0.33 50
GC-334 687784 4261349 918 0.43 155
GC-334 687791 4261284 779 0.23 23
GC-335 687827 4261448 973 0.49 155
GC-335 687948 4261449 799 0.20 17
GC-336 687297 4261392 716 0.52 156
GC-337 687102 4261436 829 0.42 451
GC-338 687358 4261458 771 0.63 345
GC-339 686901 4261649 707 0.40 118
GC-340 686787 4261654 765 0.37 231
GC-341 686900 4261647 881 0.27 25
GC-341 687017 4261644 665 0.36 276
GC-342 686829 4261499 752 0.44 363
GC-344 686576 4261473 975 0.34 25
GC-344 686576 4261438 855 0.34 45
GC-344 686578 4261400 714 0.23 58
GC-345 686592 4261581 739 0.46 157
GC-346 687242 4261672 992 0.68 93
GC-346 687310 4261709 888 0.47 38
GC-348 687189 4261390 992 0.43 135
GC-349 687628 4261894 989 0.19 20
GC-349 687601 4261819 879 0.93 117
GC-350 687839 4261670 846 0.21 18
GC-351 687865 4261631 844 0.40 63
GC-351 687896 4261485 432 0.44 783
GC-352 687398 4261752 697 0.58 532
GC-353 687544 4261725 669 0.84 560
GC-354 687682 4261871 902 0.28 48
GC-354 687636 4261810 616 0.39 500
GC-355 687453 4261337 556 0.84 206
GC-357 687569 4261324 529 0.44 473
GC-358 687786 4261317 867 0.23 60
GC-358 687799 4261395 544 0.53 422
GC-359 687840 4261385 993 0.26 80
GC-360 687426 4261390 664 0.83 383
GC-361 687543 4261281 816 0.20 15
GC-361 687522 4261404 546 0.76 530
GC-362 687340 4261789 676 0.52 302
GC-363 687436 4261742 618 0.67 355
GC-364 687806 4261408 996 0.45 63
GC-364 687630 4261390 502 0.49 955
GR-307 686924 4261623 955 0.19 90
GR-307 686926 4261747 721 0.37 40
GR-309 687019 4261626 798 0.59 421
GR-310 687829 4261715 850 0.32 20
GR-310 687805 4261664 744 0.57 170
GR-311 687827 4261413 980 0.44 140
GR-311 687829 4261394 870 0.26 38
GR-312 687348 4261822 854 0.52 238
GR-315 687449 4261346 981 0.31 95
GR-315 687449 4261440 804 1.01 285
PZ-01 687773 4261269 898 0.33 60
PZ-01 687773 4261269 804 0.22 68
PZ-06 686930 4261600 852 0.63 288
PZ-07 687715 4261196 740 0.35 39"""

rows = []
for line in raw.splitlines():
    p = line.split()
    if len(p) == 6:
        rows.append(
            [p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])]
        )

df = pd.DataFrame(rows, columns=["DHID", "East", "North", "Elevation", "Au", "Length"])
df["metal"] = df["Au"] * df["Length"]
print("Rows:", len(df), "| Holes:", df.DHID.nunique())

# --- interpolation grids ---
xi = np.linspace(df.East.min(), df.East.max(), 200)
yi = np.linspace(df.North.min(), df.North.max(), 200)
XI, YI = np.meshgrid(xi, yi)


# length-weighted IDW
def idw(x, y, z, w, XI, YI, power=2):
    Zi = np.zeros_like(XI)
    Ws = np.zeros_like(XI)
    for xp, yp, zp, wp in zip(x, y, z, w):
        d = np.hypot(XI - xp, YI - yp) + 1e-9
        wt = wp / d**power
        Zi += wt * zp
        Ws += wt
    return Zi / Ws


Zi = idw(df.East.values, df.North.values, df.Au.values, df.Length.values, XI, YI)


# --- helpers: north arrow + scale bar ---
def add_north_arrow(ax, x=0.94, y=0.94, size=0.08):
    """Add a north arrow in axes-fraction coordinates."""
    da = DrawingArea(size * 100, size * 100, 0, 0)
    triangle = Polygon(
        [[size * 50, size * 100], [size * 20, 0], [size * 80, 0]],
        closed=True,
        facecolor="black",
        edgecolor="black",
    )
    da.add_artist(triangle)
    ab = AnnotationBbox(
        da,
        (x, y),
        xycoords="axes fraction",
        frameon=True,
        box_alignment=(0.5, 0.5),
        bboxprops=dict(edgecolor="black", facecolor="white", alpha=0.85),
    )
    ax.add_artist(ab)
    ax.text(
        x,
        y + size * 0.72,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def add_scale_bar(ax, length_m=200, x=0.08, y=0.06, color="black"):
    """Add a horizontal scale bar of given real-world length (map units = metres)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width_frac = length_m / (x1 - x0)
    # bar in axes-fraction coordinates
    ax.plot(
        [x, x + width_frac],
        [y, y],
        transform=ax.transAxes,
        color=color,
        lw=3,
        solid_capstyle="butt",
        clip_on=False,
        zorder=10,
    )
    # end ticks
    for xf in (x, x + width_frac):
        ax.plot(
            [xf, xf],
            [y - 0.012, y + 0.012],
            transform=ax.transAxes,
            color=color,
            lw=2,
            clip_on=False,
            zorder=10,
        )
    # label
    ax.text(
        x + width_frac / 2,
        y + 0.02,
        f"{length_m} m",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
        zorder=10,
    )


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
sc = ax.scatter(
    df.East,
    df.North,
    s=df.Length * 2,
    c=df.Au,
    cmap="YlOrRd",
    edgecolor="k",
    linewidth=0.4,
    alpha=0.85,
)
plt.colorbar(sc, ax=ax, shrink=0.45, label="Au (g/t)")
ax.set_title("Bubble map — size=length, color=Au")

ax = axes[0, 1]
cf = ax.contourf(XI, YI, Zi, levels=20, cmap="YlOrRd")
plt.colorbar(cf, ax=ax, shrink=0.45, label="Au (g/t)")
ax.scatter(df.East, df.North, s=8, c="k", alpha=0.4)
ax.set_title("Length-weighted IDW Au surface")

ax = axes[1, 0]
sc = ax.scatter(
    df.East,
    df.North,
    s=df.metal / 3,
    c=df.metal,
    cmap="viridis",
    edgecolor="k",
    linewidth=0.3,
    alpha=0.85,
)
plt.colorbar(sc, ax=ax, shrink=0.45, label="Au × Length (g/t·m)")
ax.set_title("Metal content proxy")

ax = axes[1, 1]
sc = ax.scatter(
    df.East,
    df.North,
    s=40,
    c=df.Elevation,
    cmap="terrain_r",
    edgecolor="k",
    linewidth=0.4,
)
plt.colorbar(sc, ax=ax, shrink=0.45, label="Elevation (m)")
ax.set_title("Intercept mid-point elevation")

for ax in axes.flat:
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")

# add north arrow + scale bar to every panel
for ax in axes.flat:
    add_north_arrow(ax)
    add_scale_bar(ax, length_m=200)

plt.tight_layout()
plt.savefig("kisladag_geochemical_maps.png", dpi=300)
print("Saved: kisladag_geochemical_maps.png")

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

