import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Num_patient = [7047, 6838, 4879, 2117]

def get_label_rotation(angle, offset):
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    # This is the space between the end of the bar and the label
    padding = 0.001

    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle

        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=8
        )

def plot_circular_bar(df, GROUP_NAME, GROUPS_SIZE, COLORS, ALPHA, cluster_index):
    OFFSET = np.pi / 2
    VALUES = df["value"].values
    LABELS = df["name"].values
    GROUP = df["group"].values

    PAD = 1
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    offset = 0
    IDXS = []

    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-0.15, 0.35)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    #COLORS = [f"C{i+1}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

    for i in range(len(IDXS)):
        ax.bar(
            ANGLES[IDXS[i]], VALUES[i], width=WIDTH, color=COLORS[i],
            edgecolor="white", linewidth=2, alpha=ALPHA[i]
        )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    offset = 0
    for group, size in zip(GROUP_NAME, GROUPS_SIZE):
        # Add line below bars
        if size == 1:
            x1 = np.linspace(ANGLES[offset + PAD]-0.03, ANGLES[offset + PAD]+0.03, num=50)
        else:
            x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-0.01] * 50, color="#333333")

        #Add text to indicate group
        ax.text(
            np.mean(x1), -0.03, group, color="#333333", fontsize=10,
            fontweight="bold", ha="center", va="center"
        )

        # Add reference lines at 20, 40, 60, and 80
        if offset<47:
            x2 = np.linspace(ANGLES[offset + size]+0.08, ANGLES[offset + size + PAD]+0.05, num=50)
            ax.plot(x2, [0.05] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [0.1] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [0.15] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [0.2] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [0.25] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [0.3] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [0.35] * 50, color="#bebebe", lw=0.8)

        offset += size + PAD

    x2 = np.linspace(-0.04, 0.04,num=50)
    ax.plot(x2, [0.05] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.1] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.15] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.2] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.25] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.3] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.35] * 50, color="#bebebe", lw=0.8)

    plt.savefig('./result/PASC_c'+str(cluster_index)+'.jpg', dpi=600, format='jpg')

# You can generate the following data by your own dataset.
# Here we provided what we got from INSIGHT dataset.
Data = pd.read_excel('./result/INSIGHT_PASC.xlsx')

# You can change it according to which subphenotype you want to visualize.
cluster = 4

name = []
value = []
group = []
group_name = []
alpha = []

Matrix = np.zeros([Data.shape[0],4])
for i in range(Data.shape[0]):
    for j in range(4):
        Matrix[i,j] = int(Data.iloc[i,2+j][:Data.iloc[i,2+j].find('(')-1])/Num_patient[j]

for i in range(Data.shape[0]):

    if Data.iloc[i,6] not in group_name:
        group_name.append(Data.iloc[i,6])

    name.append(Data.iloc[i,0])
    value.append(Matrix[i,cluster-1])
    group.append(Data.iloc[i,1])
    if Matrix[i,cluster-1] == Matrix[i,:].max():
        alpha.append(1)
    else:
        alpha.append(0.1)

group_size = [1,4,6,2,5,6,6,2,1,1,2,2]

df = pd.DataFrame({
    'name': pd.Series(name),
    'value': pd.Series(value),
    'group': pd.Series(group)
                      })

color1 = ['#1f3696']*group_size[0]
color2 = ['#8d4bbb']*group_size[1]
color3 = ['#0c8918']*group_size[2]
color4 = ['#789262']*group_size[3]
color5 = ['#815463']*group_size[4]
color6 = ['#1b54f2']*group_size[5]
color7 = ['#ff7500']*group_size[6]
color8 = ['#ff2121']*group_size[7]
color9 = ['#70f3ff']*group_size[8]
color10 = ['#c4473d']*group_size[9]
color11 = ['#00e500']*group_size[10]
color12 = ['#f0c239']*group_size[11]

colors = color1+color2+color3+color4+color5+\
         color6+color7+color8+color9+color10+color11+color12


plot_circular_bar(df, group_name, group_size, colors, alpha, cluster)


