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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---- Data (from Appendix 1) ----
data = [
    ("GC-271", 687755, 4261727, 966, 0.50, 23),
    ("GC-271", 687759, 4261645, 789, 0.78, 303),
    ("GC-272", 687694, 4261462, 833, 0.46, 415),
    ("GC-273", 687823, 4261612, 836, 0.36, 183),
    ("GC-274", 687807, 4261524, 909, 0.79, 300),
    ("GC-275", 687701, 4261729, 805, 0.91, 388),
    ("GC-276", 687174, 4261444, 935, 0.62, 275),
    ("GC-277", 686970, 4261636, 891, 0.33, 206),
    ("GC-278", 687400, 4261308, 1013, 0.43, 58),
    ("GC-278", 687397, 4261359, 821, 1.08, 300),
    ("GC-279", 687154, 4261760, 919, 0.56, 240),
    ("GC-279", 687147, 4261846, 749, 0.44, 70),
    ("GC-280", 687571, 4261364, 868, 1.11, 301),
    ("GC-281", 687502, 4261377, 836, 1.70, 340),
    ("GC-282", 687108, 4261714, 866, 0.80, 399),
    ("GC-283", 687762, 4261536, 853, 0.84, 325),
    ("GC-284", 687596, 4261464, 850, 0.40, 324),
    ("GC-285", 687056, 4261595, 838, 0.70, 391),
    ("GC-286", 687569, 4261783, 989, 0.22, 28),
    ("GC-286", 687608, 4261842, 873, 0.57, 218),
    ("GC-286", 687640, 4261901, 738, 0.23, 20),
    ("GC-287", 687263, 4261405, 936, 1.01, 194),
    ("GC-288", 687123, 4261481, 1010, 0.45, 25),
    ("GC-288", 687125, 4261435, 931, 0.45, 138),
    ("GC-289", 687381, 4261462, 802, 1.07, 198),
    ("GC-290", 687443, 4261818, 929, 1.28, 165),
    ("GC-291", 687239, 4261431, 893, 1.01, 275),
    ("GC-292", 687200, 4261637, 958, 0.66, 185),
    ("GC-292", 687203, 4261715, 766, 0.53, 174),
    ("GC-293", 687221, 4261586, 986, 0.53, 133),
    ("GC-293", 687272, 4261586, 902, 0.37, 30),
    ("GC-294", 687031, 4261553, 831, 0.43, 311),
    ("GC-295", 687101, 4261654, 861, 0.75, 316),
    ("GC-296", 687239, 4261527, 809, 0.69, 125),
    ("GC-297", 687051, 4261780, 886, 0.24, 20),
    ("GC-297", 687093, 4261697, 751, 0.48, 213),
    ("GC-298", 686948, 4261430, 996, 0.28, 20),
    ("GC-298", 686948, 4261449, 964, 0.22, 30),
    ("GC-298", 686947, 4261492, 890, 0.30, 15),
    ("GC-298", 686945, 4261560, 770, 0.98, 67),
    ("GC-299", 687092, 4261517, 985, 0.29, 25),
    ("GC-299", 687073, 4261498, 931, 0.33, 25),
    ("GC-299", 687027, 4261454, 805, 0.44, 178),
    ("GC-303", 686904, 4261425, 894, 0.34, 148),
    ("GC-303", 686911, 4261515, 733, 0.20, 30),
    ("GC-303", 686913, 4261547, 676, 0.26, 38),
    ("GC-303", 686915, 4261570, 634, 0.21, 19),
    ("GC-304", 686831, 4261589, 740, 0.74, 265),
    ("GC-313", 687500, 4261782, 969, 0.60, 48),
    ("GC-313", 687501, 4261845, 857, 0.80, 175),
    ("GC-317", 687793, 4261650, 912, 0.38, 203),
    ("GC-317", 687873, 4261650, 811, 0.26, 15),
    ("GC-320", 687467, 4261359, 913, 0.66, 188),
    ("GC-320", 687492, 4261291, 807, 0.24, 15),
    ("GC-321", 687821, 4261478, 793, 0.69, 519),
    ("GC-322", 687679, 4261776, 802, 0.68, 293),
    ("GC-323", 687124, 4261777, 952, 0.63, 123),
    ("GC-324", 687575, 4261553, 974, 0.24, 15),
    ("GC-324", 687663, 4261446, 747, 0.60, 456),
    ("GC-325", 686958, 4261375, 938, 0.34, 33),
    ("GC-325", 686967, 4261447, 824, 0.41, 168),
    ("GC-325", 686978, 4261553, 662, 0.35, 120),
    ("GC-327", 687174, 4261332, 671, 0.27, 33),
    ("GC-327", 687173, 4261350, 611, 0.34, 68),
    ("GC-327", 687171, 4261367, 556, 0.22, 38),
    ("GC-328", 687319, 4261397, 879, 0.74, 258),
    ("GC-328", 687299, 4261284, 634, 0.77, 43),
    ("GC-329", 686716, 4261601, 870, 0.48, 108),
    ("GC-329", 686666, 4261532, 693, 0.36, 15),
    ("GC-329", 686650, 4261510, 633, 0.38, 27),
    ("GC-330", 686971, 4261722, 724, 0.39, 43),
    ("GC-330", 686968, 4261691, 638, 0.32, 90),
    ("GC-331", 687421, 4261250, 863, 0.74, 18),
    ("GC-333", 687852, 4261411, 991, 0.33, 50),
    ("GC-334", 687784, 4261349, 918, 0.43, 155),
    ("GC-334", 687791, 4261284, 779, 0.23, 23),
    ("GC-335", 687827, 4261448, 973, 0.49, 155),
    ("GC-335", 687948, 4261449, 799, 0.20, 17),
    ("GC-336", 687297, 4261392, 716, 0.52, 156),
    ("GC-337", 687102, 4261436, 829, 0.42, 451),
    ("GC-338", 687358, 4261458, 771, 0.63, 345),
    ("GC-339", 686901, 4261649, 707, 0.40, 118),
    ("GC-340", 686787, 4261654, 765, 0.37, 231),
    ("GC-341", 686900, 4261647, 881, 0.27, 25),
    ("GC-341", 687017, 4261644, 665, 0.36, 276),
    ("GC-342", 686829, 4261499, 752, 0.44, 363),
    ("GC-344", 686576, 4261473, 975, 0.34, 25),
    ("GC-344", 686576, 4261438, 855, 0.34, 45),
    ("GC-344", 686578, 4261400, 714, 0.23, 58),
    ("GC-345", 686592, 4261581, 739, 0.46, 157),
    ("GC-346", 687242, 4261672, 992, 0.68, 93),
    ("GC-346", 687310, 4261709, 888, 0.47, 38),
    ("GC-348", 687189, 4261390, 992, 0.43, 135),
    ("GC-349", 687628, 4261894, 989, 0.19, 20),
    ("GC-349", 687601, 4261819, 879, 0.93, 117),
    ("GC-350", 687839, 4261670, 846, 0.21, 18),
    ("GC-351", 687865, 4261631, 844, 0.40, 63),
    ("GC-351", 687896, 4261485, 432, 0.44, 783),
    ("GC-352", 687398, 4261752, 697, 0.58, 532),
    ("GC-353", 687544, 4261725, 669, 0.84, 560),
    ("GC-354", 687682, 4261871, 902, 0.28, 48),
    ("GC-354", 687636, 4261810, 616, 0.39, 500),
    ("GC-355", 687453, 4261337, 556, 0.84, 206),
    ("GC-357", 687569, 4261324, 529, 0.44, 473),
    ("GC-358", 687786, 4261317, 867, 0.23, 60),
    ("GC-358", 687799, 4261395, 544, 0.53, 422),
    ("GC-359", 687840, 4261385, 993, 0.26, 80),
    ("GC-360", 687426, 4261390, 664, 0.83, 383),
    ("GC-361", 687543, 4261281, 816, 0.20, 15),
    ("GC-361", 687522, 4261404, 546, 0.76, 530),
    ("GC-362", 687340, 4261789, 676, 0.52, 302),
    ("GC-363", 687436, 4261742, 618, 0.67, 355),
    ("GC-364", 687806, 4261408, 996, 0.45, 63),
    ("GC-364", 687630, 4261390, 502, 0.49, 955),
    ("GR-307", 686924, 4261623, 955, 0.19, 90),
    ("GR-307", 686926, 4261747, 721, 0.37, 40),
    ("GR-309", 687019, 4261626, 798, 0.59, 421),
    ("GR-310", 687829, 4261715, 850, 0.32, 20),
    ("GR-310", 687805, 4261664, 744, 0.57, 170),
    ("GR-311", 687827, 4261413, 980, 0.44, 140),
    ("GR-311", 687829, 4261394, 870, 0.26, 38),
    ("GR-312", 687348, 4261822, 854, 0.52, 238),
    ("GR-315", 687449, 4261346, 981, 0.31, 95),
    ("GR-315", 687449, 4261440, 804, 1.01, 285),
    ("PZ-01", 687773, 4261269, 898, 0.33, 60),
    ("PZ-01", 687773, 4261269, 804, 0.22, 68),
    ("PZ-06", 686930, 4261600, 852, 0.63, 288),
    ("PZ-07", 687715, 4261196, 740, 0.35, 39),
]

df = pd.DataFrame(data, columns=["DHID", "East", "North", "Elevation", "Au", "Length"])

# ---- CHART 1: Histogram of Gold Grades (Au g/t) ----
plt.figure(figsize=(10, 6))

plt.hist(df["Au"], bins=15, color="goldenrod", edgecolor="black", alpha=0.85)

plt.axvline(
    df["Au"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {df['Au'].mean():.2f} g/t",
)

plt.axvline(
    df["Au"].median(),
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Median = {df['Au'].median():.2f} g/t",
)

plt.title("Distribution of Gold Grades (Au g/t) — 125 Intercepts", fontsize=14)
plt.xlabel("Gold Grade (g/t)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart1_au_histogram.png", dpi=300)


# ---- CHART 2: Histogram of Intercept Lengths ----
plt.figure(figsize=(10, 6))

plt.hist(df["Length"], bins=20, color="steelblue", edgecolor="black", alpha=0.85)

plt.axvline(
    df["Length"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {df['Length'].mean():.0f} m",
)

plt.axvline(
    df["Length"].median(),
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Median = {df['Length'].median():.0f} m",
)

plt.title("Distribution of Mineralized Intercept Lengths", fontsize=14)
plt.xlabel("Length (m)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart2_length_histogram.png", dpi=300)


# ---- CHART 3: Box Plot of Gold Grades ----
plt.figure(figsize=(8, 5))

sns.boxplot(x=df["Au"], color="goldenrod")

plt.title("Box Plot of Gold Grades (Au g/t)", fontsize=14)
plt.xlabel("Gold Grade (g/t)")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("chart3_au_boxplot.png", dpi=300)


# ---- CHART 4: Scatter — Grade vs Length ----
plt.figure(figsize=(10, 6))

plt.scatter(df["Length"], df["Au"], alpha=0.6, color="darkgreen", edgecolor="black")

plt.title("Gold Grade vs. Intercept Length", fontsize=14)
plt.xlabel("Intercept Length (m)")
plt.ylabel("Gold Grade (g/t)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chart4_grade_vs_length.png", dpi=300)


# ---- CHART 5: Spatial Map of Intercepts (colored by grade) ----
fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    df["East"],
    df["North"],
    c=df["Au"],
    cmap="YlOrRd",
    s=df["Length"] / 10 + 20,
    edgecolor="black",
    alpha=0.8,
)

# Set equal spatial scaling
ax.set_aspect("equal", adjustable="box")

# Title and axis labels
ax.set_title(
    "Spatial Distribution of Drill Intercepts\n"
    "(bubble size ∝ intercept length, color = Au grade)",
    fontsize=13,
)

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.grid(alpha=0.3)

# ---- NORTH ARROW ----
ax.annotate(
    "N",
    xy=(0.93, 0.95),
    xytext=(0.93, 0.86),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    arrowprops=dict(
        facecolor="black", edgecolor="black", width=3, headwidth=10, headlength=12
    ),
)

# ---- SCALE BAR (200 m) ----
scale_length = 200  # meters

# Get the final map limits
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Position the scale bar in the lower-left corner
x_start = x_min + 0.08 * (x_max - x_min)
y_start = y_min + 0.06 * (y_max - y_min)

# Main scale bar
ax.plot(
    [x_start, x_start + scale_length],
    [y_start, y_start],
    color="black",
    linewidth=3,
    solid_capstyle="butt",
)

# Scale bar end ticks
tick_height = 0.008 * (y_max - y_min)

ax.plot(
    [x_start, x_start],
    [y_start - tick_height, y_start + tick_height],
    color="black",
    linewidth=1.5,
)

ax.plot(
    [x_start + scale_length, x_start + scale_length],
    [y_start - tick_height, y_start + tick_height],
    color="black",
    linewidth=1.5,
)

# Scale bar label
ax.text(
    x_start + scale_length / 2,
    y_start + 0.025 * (y_max - y_min),
    "200 m",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)

# ---- COLORBAR: MATCH THE MAP HEIGHT ----
# Create the colorbar using the same vertical position and height as the map axes.
cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, shrink=0.55, aspect=30)

cbar.set_label("Au (g/t)", fontsize=10)

# Final layout
plt.tight_layout()

plt.savefig("chart5_spatial_map.png", dpi=300, bbox_inches="tight")


# ---- CHART 6: Grade vs Elevation ----
plt.figure(figsize=(10, 6))

plt.scatter(df["Elevation"], df["Au"], alpha=0.6, color="purple", edgecolor="black")

plt.title("Gold Grade vs. Elevation", fontsize=14)
plt.xlabel("Elevation (m)")
plt.ylabel("Gold Grade (g/t)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chart6_grade_vs_elevation.png", dpi=300)


# ---- SUMMARY STATISTICS ----
print("All charts saved successfully.")

print("\nSummary Statistics:")
print(f"  Records: {len(df)}")

print(
    f"  Au mean: {df['Au'].mean():.3f} g/t | "
    f"median: {df['Au'].median():.3f} | "
    f"min: {df['Au'].min()} | "
    f"max: {df['Au'].max()}"
)

print(
    f"  Length mean: {df['Length'].mean():.1f} m | "
    f"median: {df['Length'].median():.1f} | "
    f"min: {df['Length'].min()} | "
    f"max: {df['Length'].max()}"
)

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

"""
Alternative: Turkey provinces with gold grade data

Uses Natural Earth admin-1 boundaries.
Cartopy GeoAxes (PlateCarree) for map projection,
geographic coordinate labels, north arrow, and scale bar.
"""

import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.io import DownloadWarning
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# 1. Gold grade data
# ============================================================

province_data = {
    "İzmir": {"min": 1.10, "max": 36.60, "representative": 7.6},
    "Gümüşhane": {"min": 0.89, "max": 12.00, "representative": 6.5},
    "Artvin": {"min": 1.20, "max": 4.00, "representative": 2.6},
    "Erzincan": {"min": 1.70, "max": 1.70, "representative": 1.7},
    "Çanakkale": {"min": 0.55, "max": 5.76, "representative": 3.2},
    "Balıkesir": {"min": 0.70, "max": 6.43, "representative": 3.6},
    "Manisa": {"min": 1.38, "max": 1.38, "representative": 1.38},
    "Uşak": {"min": 1.12, "max": 1.12, "representative": 1.12},
    "Eskişehir": {"min": 0.162, "max": 6.04, "representative": 3.1},
    "Sivas": {"min": 1.30, "max": 1.63, "representative": 1.47},
    "Niğde": {"min": 3.12, "max": 8.00, "representative": 5.56},
    "Ordu": {"min": 1.04, "max": 1.20, "representative": 1.12},
    "Konya": {"min": 0.79, "max": 2.36, "representative": 2.36},
}


province_coords = {
    "İzmir": (27.14, 38.42),
    "Gümüşhane": (39.48, 40.46),
    "Artvin": (41.82, 41.18),
    "Erzincan": (39.49, 39.75),
    "Çanakkale": (26.41, 40.15),
    "Balıkesir": (27.89, 39.65),
    "Manisa": (27.43, 38.61),
    "Uşak": (29.41, 38.68),
    "Eskişehir": (30.52, 39.78),
    "Sivas": (37.02, 39.75),
    "Niğde": (34.68, 37.97),
    "Ordu": (37.88, 40.98),
    "Konya": (32.48, 37.87),
}


# ============================================================
# 2. Custom label positions
# ============================================================

# Offsets are in points: (horizontal, vertical)

label_offsets = {
    "İzmir": (-38, 30),
    "Manisa": (42, -28),
}


# ============================================================
# 3. Load Natural Earth admin-1 boundaries
# ============================================================

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DownloadWarning)

    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_1_states_provinces"
    )

world_admin1 = gpd.read_file(shp_path)

turkey_provinces = world_admin1[world_admin1["adm0_a3"] == "TUR"].copy()

if turkey_provinces.crs is None:
    turkey_provinces = turkey_provinces.set_crs(epsg=4326)
else:
    turkey_provinces = turkey_provinces.to_crs(epsg=4326)


# ============================================================
# 4. Create map
# ============================================================

plate = ccrs.PlateCarree()

fig, ax = plt.subplots(
    1,
    1,
    figsize=(14, 8),
    subplot_kw={"projection": plate},
)

ax.set_extent([25, 45, 35.5, 42.5], crs=plate)


# ============================================================
# 5. Province boundaries
# ============================================================

turkey_provinces.plot(
    ax=ax,
    transform=plate,
    facecolor="white",
    edgecolor="gray",
    linewidth=0.8,
    alpha=0.6,
    zorder=1,
)


# ============================================================
# 6. Gold grade points and labels
# ============================================================

provinces = list(province_data.keys())

grades = [province_data[p]["representative"] for p in provinces]

norm = Normalize(vmin=min(grades), vmax=max(grades))

cmap = plt.cm.YlOrRd

for prov in provinces:
    lon, lat = province_coords[prov]

    grade = province_data[prov]["representative"]

    color = cmap(norm(grade))

    # ----------------------------------------
    # Gold grade point
    # ----------------------------------------

    ax.scatter(
        lon,
        lat,
        s=400,
        c=[color],
        edgecolors="black",
        linewidth=1.5,
        transform=plate,
        zorder=5,
    )

    # ----------------------------------------
    # Gold grade range
    # ----------------------------------------

    min_g = province_data[prov]["min"]
    max_g = province_data[prov]["max"]

    grade_text = f"{min_g:.2f}" if min_g == max_g else f"{min_g:.2f}–{max_g:.2f}"

    label = f"{prov}\n({grade_text} g/t)"

    # ----------------------------------------
    # Custom label positions
    # ----------------------------------------

    offset = label_offsets.get(prov, (0, 14))

    # ----------------------------------------
    # Province labels
    # ----------------------------------------

    ax.annotate(
        label,
        xy=(lon, lat),
        xycoords="data",
        textcoords="offset points",
        xytext=offset,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=10,
        arrowprops=(
            dict(
                arrowstyle="-",
                color="black",
                linewidth=0.7,
                shrinkA=3,
                shrinkB=7,
            )
            if prov in ["İzmir", "Manisa"]
            else None
        ),
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=1.5,
        ),
    )


# ============================================================
# 7. Geographic coordinates and map styling
# ============================================================

# Longitude and latitude tick intervals

lon_ticks = np.arange(25, 46, 2)

lat_ticks = np.arange(36, 43, 1)

# Set coordinate ticks

ax.set_xticks(lon_ticks, crs=plate)

ax.set_yticks(lat_ticks, crs=plate)

# Format longitude and latitude labels

ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))

ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))

# Coordinate tick appearance

ax.tick_params(
    axis="both",
    which="major",
    labelsize=9,
    direction="out",
    length=4,
    width=0.8,
    pad=5,
    bottom=True,
    top=False,
    left=True,
    right=False,
    labelbottom=True,
    labelleft=True,
)

# Axis labels

ax.set_xlabel("Longitude (°E)", fontsize=11, labelpad=8)

ax.set_ylabel("Latitude (°N)", fontsize=11, labelpad=8)

# Map title

ax.set_title(
    "Schematic: Provinces in Türkiye with Reported Gold Grade (g/t) Data",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

# Geographic gridlines

ax.gridlines(
    crs=plate,
    draw_labels=False,
    linewidth=0.5,
    color="gray",
    alpha=0.4,
    linestyle="--",
    xlocs=lon_ticks,
    ylocs=lat_ticks,
    zorder=2,
)


# ============================================================
# 8. Colorbar
# ============================================================

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=ax,
    orientation="vertical",
    shrink=0.63,
    pad=0.02,
)

cbar.set_label("Representative Gold Grade (g/t Au)", fontsize=10)

cbar.ax.tick_params(labelsize=9)


# ============================================================
# 9. North arrow
# ============================================================


def draw_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.94, 0.94),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        zorder=20,
    )

    ax.annotate(
        "",
        xy=(0.94, 0.91),
        xytext=(0.94, 0.81),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
        zorder=20,
    )


# ============================================================
# 10. Manual scale bar
# ============================================================


def draw_manual_scale_bar(ax, lon0=38.5, lat0=36.3, km=100, height=0.12):
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat0))

    dlon = km / km_per_deg_lon

    # Main scale bar

    ax.plot(
        [lon0, lon0 + dlon],
        [lat0, lat0],
        color="black",
        lw=4,
        solid_capstyle="butt",
        transform=plate,
        zorder=20,
    )

    # End ticks

    for x in (lon0, lon0 + dlon):
        ax.plot(
            [x, x],
            [lat0 - height / 2, lat0 + height / 2],
            color="black",
            lw=1.5,
            transform=plate,
            zorder=20,
        )

    # Scale label

    ax.text(
        lon0 + dlon / 2,
        lat0 + height,
        f"{km} km",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        transform=plate,
        zorder=20,
    )


# ============================================================
# 11. Add north arrow and scale bar
# ============================================================

try:
    from matplotlib_map_utils.core.north_arrow import north_arrow
    from matplotlib_map_utils.core.scale_bar import scale_bar

    north_arrow(ax=ax, location="upper right", rotation={"degrees": 0}, scale=0.6)

    scale_bar(
        ax=ax,
        location="lower right",
        style="boxes",
        bar={"projection": plate},
        labels={"length": 100},
    )

    print("[Info] Using matplotlib_map_utils north arrow + scale bar.")

except Exception as e:
    print(f"[Info] matplotlib_map_utils unavailable ({e}); using manual fallback.")

    draw_north_arrow(ax)

    draw_manual_scale_bar(ax, lon0=38.5, lat0=36.3, km=100)


# ============================================================
# 12. Final layout and export
# ============================================================

plt.tight_layout()

plt.savefig(
    "turkey_gold_grade_schematic_map_ne.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

"""
Global Distribution of Major Gold Deposits and Mines
Gold Grade (Au, g/t)

Cartographic features:
- Robinson projection
- Geographic longitude and latitude coordinates
- Geographic graticule
- Horizontal geodesic scale bar (2,000 km)
- North arrow
- Logarithmic gold-grade color classification
- Selected deposit labels
- High-resolution PNG and PDF export

IMPORTANT:
Gold grades are illustrative values.
Verify all values using original technical reports
before scientific publication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Geod
from matplotlib.lines import Line2D

# ============================================================
# 1. GLOBAL GOLD DEPOSIT DATABASE
# ============================================================

data = [
    ("Kisladag", "Türkiye", 29.55, 38.48, 0.70),
    ("Copler", "Türkiye", 40.12, 39.75, 1.50),
    ("Ovacik", "Türkiye", 27.08, 39.08, 7.00),
    ("Muruntau", "Uzbekistan", 64.57, 41.50, 2.50),
    ("Olimpiada", "Russia", 93.43, 58.60, 3.50),
    ("Sukhoi Log", "Russia", 103.96, 57.55, 2.30),
    ("Grasberg", "Indonesia", 137.13, -4.05, 0.80),
    ("Lihir", "Papua New Guinea", 152.64, -3.13, 2.50),
    ("Cadia", "Australia", 149.03, -33.45, 0.40),
    ("Boddington", "Australia", 116.39, -32.80, 0.70),
    ("Fosterville", "Australia", 144.18, -36.77, 5.00),
    ("Carlin Trend", "USA", -116.10, 40.70, 2.50),
    ("Goldstrike", "USA", -116.28, 40.97, 3.00),
    ("Cortez", "USA", -116.68, 40.18, 2.00),
    ("Kirkland Lake", "Canada", -79.97, 48.15, 5.00),
    ("Detour Lake", "Canada", -79.72, 50.02, 1.00),
    ("Yanacocha", "Peru", -78.50, -6.98, 0.80),
    ("Pueblo Viejo", "Dominican Republic", -70.17, 18.93, 2.50),
    ("Paracatu", "Brazil", -46.85, -17.22, 0.40),
    ("Kibali", "DR Congo", 29.75, 3.10, 3.50),
    ("Loulo-Gounkoto", "Mali", -11.43, 13.05, 4.00),
    ("Ahafo", "Ghana", -2.32, 7.04, 1.50),
    ("Obuasi", "Ghana", -1.67, 6.20, 6.00),
    ("Geita", "Tanzania", 32.23, -2.87, 3.00),
    ("Mponeng", "South Africa", 27.42, -26.44, 8.00),
]

columns = ["Deposit", "Country", "Longitude", "Latitude", "Au_Grade"]

df = pd.DataFrame(data, columns=columns)

# ============================================================
# 2. FIGURE AND MAP PROJECTION
# ============================================================

fig = plt.figure(figsize=(17, 9), dpi=300)

map_crs = ccrs.Robinson()

ax = fig.add_subplot(1, 1, 1, projection=map_crs)

ax.set_global()
ax.set_facecolor("white")

# ============================================================
# 3. MAP FEATURES
# ============================================================

ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)

ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)

ax.add_feature(cfeature.COASTLINE, linewidth=0.55, edgecolor="black", zorder=1)

ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="gray", zorder=1)

ax.spines["geo"].set_linewidth(0.9)
ax.spines["geo"].set_edgecolor("black")

# ============================================================
# 4. GEOGRAPHIC COORDINATES AND GRATICULE
# ============================================================

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.45,
    color="gray",
    alpha=0.55,
    linestyle="--",
    x_inline=False,
    y_inline=False,
)

gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 30))
gl.ylocator = plt.FixedLocator(np.arange(-60, 91, 15))

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

gl.xlabel_style = {"size": 8, "color": "black"}
gl.ylabel_style = {"size": 8, "color": "black"}

gl.top_labels = True
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = True

# ============================================================
# 5. GOLD GRADE CLASSIFICATION
# ============================================================

grades = df["Au_Grade"].values

norm = mcolors.LogNorm(vmin=0.3, vmax=10)

cmap = plt.cm.YlOrRd

marker_sizes = 35 + grades * 15

# ============================================================
# 6. PLOT GLOBAL GOLD DEPOSITS
# ============================================================

scatter = ax.scatter(
    df["Longitude"],
    df["Latitude"],
    c=grades,
    s=marker_sizes,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    edgecolors="black",
    linewidths=0.65,
    alpha=0.90,
    zorder=5,
)

# ============================================================
# 7. DEPOSIT LABELS
# ============================================================

selected_labels = [
    "Kisladag",
    "Muruntau",
    "Grasberg",
    "Carlin Trend",
    "Fosterville",
    "Mponeng",
    "Lihir",
    "Olimpiada",
]

label_offsets = {
    "Kisladag": (3, 2),
    "Muruntau": (3, 2),
    "Grasberg": (-15, -5),
    "Carlin Trend": (-20, 3),
    "Fosterville": (3, -4),
    "Mponeng": (3, -3),
    "Lihir": (3, 3),
    "Olimpiada": (3, 2),
}

for _, row in df.iterrows():
    name = row["Deposit"]

    if name in selected_labels:
        dx, dy = label_offsets[name]

        ax.text(
            row["Longitude"] + dx,
            row["Latitude"] + dy,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=7,
            fontweight="bold",
            color="black",
            zorder=6,
        )

# ============================================================
# 8. COLORBAR
# ============================================================

cbar = plt.colorbar(
    scatter, ax=ax, orientation="vertical", pad=0.025, shrink=0.72, aspect=28
)

cbar.set_label("Gold Grade (Au, g/t)", fontsize=11, fontweight="bold", labelpad=12)

cbar.set_ticks([0.3, 0.5, 1, 2, 5, 10])

cbar.ax.tick_params(labelsize=9, width=0.7)

cbar.outline.set_linewidth(0.7)

# ============================================================
# 9. NORTH ARROW
# ============================================================

ax.annotate(
    "N",
    xy=(0.94, 1.05),
    xytext=(0.94, 0.95),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    arrowprops=dict(
        facecolor="black", edgecolor="black", width=2, headwidth=9, headlength=10
    ),
    zorder=10,
)

# ============================================================
# 10. HORIZONTAL GEODESIC SCALE BAR
# ============================================================

geod = Geod(ellps="WGS84")

# Scale bar length
scale_length_km = 2000

# Starting geographic coordinates
lon_start = -120
lat_start = -48

# Calculate geodesic endpoint
lon_end, lat_end, _ = geod.fwd(lon_start, lat_start, 90, scale_length_km * 1000)

# Transform coordinates into Robinson projection
x_start, y_start = map_crs.transform_point(lon_start, lat_start, ccrs.PlateCarree())

x_end, y_end = map_crs.transform_point(lon_end, lat_end, ccrs.PlateCarree())

# Horizontal scale bar
y_scale = y_start

ax.plot(
    [x_start, x_end],
    [y_scale, y_scale],
    transform=map_crs,
    color="black",
    linewidth=3,
    solid_capstyle="butt",
    zorder=10,
)

# Vertical end ticks
tick_length = 120000

for x in [x_start, x_end]:
    ax.plot(
        [x, x],
        [y_scale - tick_length / 2, y_scale + tick_length / 2],
        transform=map_crs,
        color="black",
        linewidth=1.5,
        zorder=10,
    )

# Scale bar label
ax.text(
    (x_start + x_end) / 2,
    y_scale + 220000,
    "2,000 km",
    transform=map_crs,
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="black",
    zorder=11,
)

# ============================================================
# 11. MAP TITLE
# ============================================================

ax.set_title(
    "Global Distribution of Major Gold Deposits and Mines "
    "According to Gold Grade (Au, g/t)",
    fontsize=15,
    fontweight="bold",
    pad=25,
)

# ============================================================
# 12. LEGEND
# ============================================================

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Gold deposit / mine",
        markerfacecolor="#E67E22",
        markeredgecolor="black",
        markersize=8,
    )
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=9,
    frameon=True,
    framealpha=0.95,
    edgecolor="black",
)

# ============================================================
# 13. DATA SOURCE NOTE
# ============================================================

fig.text(
    0.5,
    0.025,
    "Illustrative Au grades; verify against original "
    "deposit-specific technical reports before publication.",
    ha="center",
    fontsize=8,
    color="dimgray",
)

# ============================================================
# 14. EXPORT HIGH-RESOLUTION MAP
# ============================================================

plt.savefig(
    "Global_Major_Gold_Deposits_Au_Grade.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    "Global_Major_Gold_Deposits_Au_Grade.pdf", bbox_inches="tight", facecolor="white"
)

plt.show()

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



