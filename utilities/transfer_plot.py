import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.configuration import x_coords, z_coords, Gate, WIDTH, H_GATE

def transferfunc(x_coord, z_coord):
    """Plots transfer function at specified gate coordinate"""
    for i in range(Gate.n_x):
        if (x_coords[Gate.ii_x[i+1]] > x_coord) & (x_coords[Gate.ii_x[i]] <= x_coord):
            for j in range(Gate.n_z):
                if (z_coords[Gate.ii_z[j+1]] > z_coord) & (z_coords[Gate.ii_z[j]] <= z_coord):
                    x_plot = i
                    z_plot = j

    fig, ax = plt.subplots(1,1,figsize=[15,5])
    for i in range(len(Gate.FRF[x_plot][z_plot])):
        ax.plot(Gate.FRF_f,abs(Gate.FRF[x_plot][z_plot][i]), label = 'mode '+str(i+1))
    ax.legend()
    ax.set_xlim(min(Gate.FRF_f),max(Gate.FRF_f))
    ax.set_yscale('log')
    ax.set_xlabel('f [Hz]')
    ax.set_ylabel('Modal coefficient')
    plt.close(fig)
    return fig

def plotmode_3D(mode, modetype):
    if modetype == "disp":
        label = 'Deflection [mm]'
        colmode = Gate.displist*1000
    elif modetype == "stress_pos":
        label = 'Stress [MPa]'
        colmode = Gate.stressposlist/10**6
    elif modetype == "stress_neg":
        label = 'Stress [MPa]'
        colmode = Gate.stressneglist/10**6
    elif modetype == "shear":
        label = 'Stress [MPa]'
        colmode = Gate.shearlist/10**6
        
    fig = plt.figure(figsize=[12,8])
    ax = Axes3D(fig)

    Zmin = min(colmode[:,mode-1])
    Zmax = max(colmode[:,mode-1])
    if modetype == "disp":
        if mode == 1:
            cmap = plt.cm.Reds
            norm = colors.Normalize()
        else:
            cmap = plt.cm.coolwarm
            norm = colors.TwoSlopeNorm(vmin=-Zmax, vcenter=0, vmax=Zmax)
    elif modetype in ['stress_pos', 'stress_neg', 'shear']:
        cmap = plt.cm.Reds
        norm = colors.PowerNorm(gamma=1)

    coords = []
    response = []
    for i in range(len(Gate.faces)):
        coords.append(Gate.verts[Gate.faces[i]-1])
        response.append(colmode[Gate.faces[i]-1,mode-1].mean())

    facets = Poly3DCollection(coords)
    facets.set_facecolor(cmap(norm(response)))
    ax.add_collection3d(facets)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),fraction=0.02, pad=0)
    cbar.set_label(label, rotation=270, labelpad=20)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim3d(0, WIDTH)
    ax.set_ylim3d(-4,4)
    ax.set_zlim3d(0,8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    x_scale=WIDTH
    y_scale=8
    z_scale=8

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)
    ax.get_proj=short_proj
    ax.view_init(30, 50)

    plt.close(fig)
    return ax, fig

def plotmodes_3D(modes=[1,2,3,4], modetype='stress_pos'):
    if modetype == "disp":
        label = 'Deflection [mm]'
        colmode = Gate.displist*1000
    elif modetype == "stress_pos":
        label = 'Stress [MPa]'
        colmode = Gate.stressposlist/10**6
    elif modetype == "stress_neg":
        label = 'Stress [MPa]'
        colmode = Gate.stressneglist/10**6
    elif modetype == "shear":
        label = 'Stress [MPa]'
        colmode = Gate.shearlist/10**6
        
    fig = plt.figure(figsize=[15,12])
    label = 'Normalized mode shape [-]'
    for m, mode in enumerate(modes):
        coords = []
        response = []
        for i in range(len(Gate.faces)):
            coords.append(Gate.verts[Gate.faces[i]-1])
            response.append(colmode[Gate.faces[i]-1, mode-1].mean())
            
        response /= np.max(response) # Normalize
        ax = fig.add_subplot(2, 2, m+1, projection='3d')        
        if modetype == "disp":
                cmap = plt.cm.coolwarm
                norm = colors.TwoSlopeNorm(vmin=0, vcenter=0, vmax=1)
        elif modetype in ['stress_pos', 'stress_neg', 'shear']:
            cmap = plt.cm.Reds
            norm = colors.Normalize(0,1)

        facets = Poly3DCollection(coords)
        facets.set_facecolor(cmap(norm(response)))
        ax.add_collection3d(facets)

        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), fraction=0.02, pad=.05)
        cbar.set_label(label, rotation=270, labelpad=20)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim3d(0, WIDTH)
        ax.set_ylim3d(-4,4)
        ax.set_zlim3d(0,8)
        ax.set_title('Mode %s'%mode)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.view_init(30, 50)

    plt.close(fig)
    return fig

