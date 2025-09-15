"""
Process optimization of slope stability analysis based on ensemble learning algorithm.  
The current version requires manual input of the low value of the stability coefficient (-0.1) 
for prediction using an integrated algorithm and calculation using the intensity subtraction method. 
 The optimized version can relatively reduce the number of iterations and improve the efficiency of computing power. 
 Please note that this is only the updated main program code segment.  
The rest of the library really see the project repository at : 
https://github.com/QiningDeng/The-programming-performance-of-ChatGPT-in-Slope-stability-analysis
"""

import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from A_mesh import mesh
from B_DCM import DCM

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define geometric parameters
x1 = 10   # Length of the front slope platform
x2 = 7    # Length of the slope body
x3 = 10   # Length of the rear slope platform
y1 = 10   # Height of the front slope platform
y2 = 8    # Height of the slope body
z  = 6    # Width of the slope

# Soil material parameters and mesh density
gamma = 17
c0 = 15
phi = 20
E = 30000000
v = 0.3
psi = 0.1
G = E / (2 * (1 + v))
K = E / (3 * (1 - 2 * v))
lamlda = K - 2 * G / 3
N_h = 1  # Mesh division density parameter

# Integration points and weights
Xi = np.array([
    [0.2500, 0.0714285714285714, 0.785714285714286, 0.0714285714285714, 0.0714285714285714, 0.399403576166799,
     0.100596423833201, 0.100596423833201, 0.399403576166799, 0.399403576166799, 0.100596423833201],
    [0.2500, 0.0714285714285714, 0.0714285714285714, 0.785714285714286, 0.0714285714285714, 0.100596423833201,
     0.399403576166799, 0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799],
    [0.2500, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.785714285714286, 0.100596423833201,
     0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799, 0.399403576166799]
])
WF = np.array([-0.013155555555555, 0.007622222222222, 0.007622222222222, 0.007622222222222, 0.007622222222222,
               0.024888888888888, 0.024888888888888, 0.024888888888888, 0.024888888888888, 0.024888888888888,
               0.024888888888888]).reshape(1, -1)

# Local shape functions and directional derivatives
xi_1 = Xi[0, :]
xi_2 = Xi[1, :]
xi_3 = Xi[2, :]
xi_0 = 1 - xi_1 - xi_2 - xi_3
n_q = len(xi_1)

HatP = np.array([
    xi_0 * (2 * xi_0 - 1), xi_1 * (2 * xi_1 - 1),
    xi_2 * (2 * xi_2 - 1), xi_3 * (2 * xi_3 - 1),
    4 * xi_0 * xi_1, 4 * xi_1 * xi_2,
    4 * xi_0 * xi_2, 4 * xi_1 * xi_3,
    4 * xi_2 * xi_3, 4 * xi_0 * xi_3
])

DHatP1 = np.array([
    -4 * xi_0 + 1, 4 * xi_1 - 1,
    np.zeros(n_q), np.zeros(n_q),
    4 * (xi_0 - xi_1), 4 * xi_2,
    -4 * xi_2, 4 * xi_3,
    np.zeros(n_q), -4 * xi_3
])
DHatP2 = np.array([
    -4 * xi_0 + 1, np.zeros(n_q),
    4 * xi_2 - 1, np.zeros(n_q),
    -4 * xi_1, 4 * xi_1,
    4 * (xi_0 - xi_2), np.zeros(n_q),
    4 * xi_3, -4 * xi_3
])
DHatP3 = np.array([
    -4 * xi_0 + 1, np.zeros(n_q),
    np.zeros(n_q), 4 * xi_3 - 1,
    -4 * xi_1, np.zeros(n_q),
    -4 * xi_2, 4 * xi_1,
    4 * xi_2, 4 * (xi_0 - xi_3)
])

coord, elem, surf, Q = mesh(N_h, x1, x2, x3, y1, y2, z)
Q = Q.astype(bool)
coord_Q = np.ravel(coord, order='F')[Q.ravel(order='F') == 1]

# Store mesh information
n_n = coord.shape[1]   # Number of nodes
n_unknown = len(np.array(coord_Q))  
n_p = elem.shape[0]    # Number of vertices per element
n_e = elem.shape[1]    # Number of elements
n_q = WF.shape[1]      # Number of integration points
n_int = n_e * n_q      # Total integration points

# Print mesh statistics
print(f'Mesh data: nodes = {n_n}, unknowns = {n_unknown}, elements = {n_e}, integration points = {n_int}')

# Define function for mesh visualization
def plot_surface_mesh(coord, surf, save_path="mesh_surface.png", dpi=1000):
    """
    Plot outer triangular surface (rotation + mirror + custom color)
    coord: ndarray, shape (3, N)
    surf: ndarray, shape (6, K)
    """
    coord = coord.T
    tris = surf[:3, :].astype(int).T
    faces = []

    for tri in tris:
        tri = tri - 1  # Adjust index
        faces.append(coord[tri])
    faces = np.array(faces)

    # Visualization
    R_x = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
    faces_rot = np.einsum('ij,klj->kli', R_x, faces)
    faces_rot[:, :, 0] = -faces_rot[:, :, 0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Set face color
    rgb_color = (232/255, 217/255, 197/255)
    pc = Poly3DCollection(faces_rot, facecolor=rgb_color, edgecolor="k", linewidths=0.2, alpha=0.7)
    ax.add_collection3d(pc)

    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# Example of plotting the mesh
plot_surface_mesh(coord, surf)

# Update material parameters at integration points
matrix_c0 = c0 * np.ones(n_int).reshape(1, -1)
matrix_phi = np.deg2rad(phi) * np.ones(n_int).reshape(1, -1)
matrix_psi = np.deg2rad(psi) * np.ones(n_int).reshape(1, -1)
matrix_G = G * np.ones(n_int).reshape(1, -1)
matrix_K = K * np.ones(n_int).reshape(1, -1)
matrix_lamlda = lamlda * np.ones(n_int).reshape(1, -1)
matrix_gamma = gamma * np.ones(n_int).reshape(1, -1)

# Replicate DHatP1, DHatP2, DHatP3 along columns
DHatPhi1 = np.tile(DHatP1, (1, n_e))
DHatPhi2 = np.tile(DHatP2, (1, n_e))
DHatPhi3 = np.tile(DHatP3, (1, n_e))

# Reshape coordinates based on elem
coorde1 = np.reshape(coord[0, elem.flatten(order='F')-1], (n_p, n_e), order='F')
coorde2 = np.reshape(coord[1, elem.flatten(order='F')-1], (n_p, n_e), order='F')
coorde3 = np.reshape(coord[2, elem.flatten(order='F')-1], (n_p, n_e), order='F')

# Coordinates around each integration point
coordint1 = np.kron(coorde1, np.ones((1, n_q)))
coordint2 = np.kron(coorde2, np.ones((1, n_q)))
coordint3 = np.kron(coorde3, np.ones((1, n_q)))

# Components of the Jacobian matrix
J11 = np.sum(coordint1 * DHatPhi1, axis=0).reshape(1, -1)
J12 = np.sum(coordint2 * DHatPhi1, axis=0).reshape(1, -1)
J13 = np.sum(coordint3 * DHatPhi1, axis=0).reshape(1, -1)
J21 = np.sum(coordint1 * DHatPhi2, axis=0).reshape(1, -1)
J22 = np.sum(coordint2 * DHatPhi2, axis=0).reshape(1, -1)
J23 = np.sum(coordint3 * DHatPhi2, axis=0).reshape(1, -1)
J31 = np.sum(coordint1 * DHatPhi3, axis=0).reshape(1, -1)
J32 = np.sum(coordint2 * DHatPhi3, axis=0).reshape(1, -1)
J33 = np.sum(coordint3 * DHatPhi3, axis=0).reshape(1, -1)

# Jacobian determinant
DET = (J11 * (J22 * J33 - J32 * J23) - J12 * (J21 * J33 - J23 * J31) +
       J13 * (J21 * J32 - J22 * J31))

# Inverse of Jacobian components
Jinv11 = (J22 * J33 - J23 * J32) / DET
Jinv12 = -(J12 * J33 - J13 * J32) / DET
Jinv13 = (J12 * J23 - J13 * J22) / DET
Jinv21 = -(J21 * J33 - J23 * J31) / DET
Jinv22 = (J11 * J33 - J13 * J31) / DET
Jinv23 = -(J11 * J23 - J13 * J21) / DET
Jinv31 = (J21 * J32 - J22 * J31) / DET
Jinv32 = -(J11 * J32 - J12 * J31) / DET
Jinv33 = (J11 * J22 - J12 * J21) / DET

# Replicate Jinv arrays along rows
Jinv11_rep = np.tile(Jinv11, (n_p, 1))
Jinv12_rep = np.tile(Jinv12, (n_p, 1))
Jinv13_rep = np.tile(Jinv13, (n_p, 1))

Jinv21_rep = np.tile(Jinv21, (n_p, 1))
Jinv22_rep = np.tile(Jinv22, (n_p, 1))
Jinv23_rep = np.tile(Jinv23, (n_p, 1))

Jinv31_rep = np.tile(Jinv31, (n_p, 1))
Jinv32_rep = np.tile(Jinv32, (n_p, 1))
Jinv33_rep = np.tile(Jinv33, (n_p, 1))

# Compute DPhi arrays
DPhi1 = Jinv11_rep * DHatPhi1 + Jinv12_rep * DHatPhi2 + Jinv13_rep * DHatPhi3
DPhi2 = Jinv21_rep * DHatPhi1 + Jinv22_rep * DHatPhi2 + Jinv23_rep * DHatPhi3
DPhi3 = Jinv31_rep * DHatPhi1 + Jinv32_rep * DHatPhi2 + Jinv33_rep * DHatPhi3

# Strain-displacement matrix B
n_b = 18 * n_p
vB = np.zeros((n_b, n_int))
vB[0:n_b-17:18, :] = DPhi1
vB[9:n_b-8:18, :] = DPhi1
vB[17:n_b:18, :] = DPhi1
vB[3:n_b-14:18, :] = DPhi2
vB[7:n_b-10:18, :] = DPhi2
vB[16:n_b-1:18, :] = DPhi2
vB[5:n_b-12:18, :] = DPhi3
vB[10:n_b-7:18, :] = DPhi3
vB[14:n_b-3:18, :] = DPhi3

# Indices for sparse matrix B
AUX = np.reshape(np.arange(1, 6 * n_int + 1), (6, n_int), order='F')
iB = np.tile(AUX, (3 * n_p, 1))

AUX1 = np.array([1, 1, 1]).reshape(3, 1) @ np.arange(1, n_p + 1).reshape(1, n_p)
AUX2 = np.array([2, 1, 0]).reshape(3, 1) @ np.ones((1, n_p))

AUX3 = 3 * elem[AUX1.flatten(order='F').T - 1, :] - np.kron(np.ones((1, n_e)), AUX2.flatten(order='F')).reshape(len(AUX2.flatten(order='F')), n_e, order='F')
jB = np.kron(AUX3, np.ones((6, n_q)))

# Construct sparse matrix B
B = scipy.sparse.csr_matrix((vB.flatten(), (iB.flatten() - 1, jB.flatten() - 1)), shape=(6*n_int, 3*n_n))
B = B.tocsc()

# Elastic stress-strain matrix D
IOTA = np.array([1, 1, 1, 0, 0, 0])
VOL = np.outer(IOTA, IOTA)
DEV = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) - VOL / 3
ELAST = 2 * np.outer(DEV.flatten(), matrix_G) + np.outer(VOL.flatten(), matrix_K)
WEIGHT = np.abs(DET) * np.tile(WF, n_e)

iD = np.tile(AUX, (6, 1))
jD = np.kron(AUX, np.ones((6, 1)))
vD = ELAST * WEIGHT

D = scipy.sparse.csr_matrix((vD.flatten(), (iD.flatten()-1, jD.flatten()-1)), shape=(6 * n_int , 6 * n_int))
D = D.tocsc()

# Elastic stiffness matrix K
K_elast = B.T.dot(D.dot(B))

# Assemble body force density vector f_V_int
f_V_int = np.array([np.zeros(n_int), -gamma * np.ones(n_int), np.zeros(n_int)])

# Total body force vector f_V
HatPhi = np.tile(HatP, (1, n_e))

vF1 = HatPhi * (WEIGHT * f_V_int[0, :])
vF2 = HatPhi * (WEIGHT * f_V_int[1, :])
vF3 = HatPhi * (WEIGHT * f_V_int[2, :])
iF = np.ones((n_p, n_int))
jF = np.kron(elem, np.ones((1, n_q)))

f_V = np.vstack([
    scipy.sparse.coo_matrix((vF1.flatten(), (iF.flatten()-1, jF.flatten()-1)), shape=(1, n_n)).toarray(),
    scipy.sparse.coo_matrix((vF2.flatten(), (iF.flatten()-1, jF.flatten()-1)), shape=(1, n_n)).toarray(),
    scipy.sparse.coo_matrix((vF3.flatten(), (iF.flatten()-1, jF.flatten()-1)), shape=(1, n_n)).toarray()
])

f = f_V

# Strength reduction method parameters
lambda_init = 1.20  # Initial reduction factor
d_lambda_init = 0.10  # Initial increment
d_lambda_min = 1e-4  # Minimum increment
step_max = 50  # Maximum steps

# Newton iteration parameters
it_newt_max = 30  # Maximum Newton iterations
it_damp_max = 10  # Maximum line search iterations
tol = 1e-6  # Relative tolerance for Newton iteration
r_min = tol / 100  # Basic regularization for stiffness matrix
r_damp = tol * 100  # Line search regularization for stiffness matrix

# Compute displacement, reduction factor, and control variables at each step using DCM
U2, lambda_hist2, omega_hist2 = DCM(
    lambda_init, d_lambda_init, d_lambda_min, step_max, it_newt_max,
    it_damp_max, tol, r_min, r_damp, WEIGHT, B, K_elast, Q, f,
    matrix_c0, matrix_phi, matrix_psi, matrix_G, matrix_K, matrix_lamlda
)

def fem_cloud_plot_all(coord, U2, elem, HatP, DHatP1, DHatP2, DHatP3, Xi, WF, E, v, scale=1.0, save_prefix="FEM_CloudPlot"):
    """
    FEM quadratic tetrahedral element post-processing and 3D cloud plot function (displacement, strain, stress output)
    
    Parameters:
        coord: ndarray (N_nodes, 3) or (3, N_nodes) Node coordinates
        U2: ndarray (N_nodes, 3) Node displacements
        elem: ndarray (10, N_elem) Element node indices (1-based)
        HatP, DHatP1, DHatP2, DHatP3: Shape functions and derivatives
        Xi, WF: Integration points and weights
        E, v: Elastic parameters
        scale: Displacement scaling factor
        save_prefix: Prefix for output image filenames
    """

    coord = coord.T
    U2 = U2.T

    # Elastic constitutive matrix
    G = E / (2*(1+v))
    K = E / (3*(1-2*v))
    lam = K - 2*G/3
    D = np.array([
        [lam+2*G, lam, lam, 0, 0, 0],
        [lam, lam+2*G, lam, 0, 0, 0],
        [lam, lam, lam+2*G, 0, 0, 0],
        [0,0,0,G,0,0],
        [0,0,0,0,G,0],
        [0,0,0,0,0,G]
    ])

    n_elem = elem.shape[1]
    n_q = Xi.shape[1]

    # Initialize storage arrays
    disp_points = []
    disp_values = []
    strain_values = []
    stress_values = []

    for e in range(n_elem):
        nodes_idx = elem[:, e] - 1
        coords_e = coord[nodes_idx, :]  # 10×3 element node coordinates
        Ue = U2[nodes_idx, :]           # 10×3 element node displacements

        for q in range(n_q):
            N = HatP[:, q].reshape(1, -1)
            uq = (N @ Ue).flatten() * scale
            disp_points.append(coords_e.mean(axis=0) + uq)

            # Compute derivatives of shape functions w.r.t global coordinates
            dNdxi = np.vstack((DHatP1[:, q], DHatP2[:, q], DHatP3[:, q]))
            J = dNdxi @ coords_e
            invJ = np.linalg.inv(J)
            dNdx = invJ @ dNdxi

            # Compute strain-displacement matrix B
            B = np.zeros((6, 30))
            for i in range(10):
                Bi = np.array([
                    [dNdx[0,i],0,0],
                    [0,dNdx[1,i],0],
                    [0,0,dNdx[2,i]],
                    [dNdx[1,i],dNdx[0,i],0],
                    [0,dNdx[2,i],dNdx[1,i]],
                    [dNdx[2,i],0,dNdx[0,i]]
                ])
                B[:, i*3:(i+1)*3] = Bi

            # Compute strain and stress at the integration point
            strain = B @ Ue.flatten()
            stress = D @ strain

            disp_values.append(np.linalg.norm(uq))
            strain_values.append(np.linalg.norm(strain))
            stress_values.append(np.linalg.norm(stress))

    disp_points = np.array(disp_points)
    disp_values = np.array(disp_values)
    strain_values = np.array(strain_values)
    stress_values = np.array(stress_values)

    # ===== Data rotation =====
    # Rotate 90° counterclockwise around X-axis
    theta = np.pi / 2
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    disp_points = disp_points @ Rx.T

    # Rotate 180° counterclockwise around Z-axis
    phi = np.pi
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    disp_points = disp_points @ Rz.T

    # ===== Plotting function =====
    def plot_and_save(points, values, field_name):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points[:,0], points[:,1], points[:,2],
                        c=values, cmap='jet', s=8)
        
        # Colorbar settings
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label(field_name, fontdict={'family':'Times New Roman', 'size':14})
        cbar.ax.tick_params(labelsize=12)
        for t in cbar.ax.get_yticklabels():
            t.set_fontname('Times New Roman')

        # Axis labels
        ax.set_xlabel('X', fontdict={'family':'Times New Roman', 'size':14})
        ax.set_ylabel('Y', fontdict={'family':'Times New Roman', 'size':14})
        ax.set_zlabel('Z', fontdict={'family':'Times New Roman', 'size':14})
        
        # 3D axis tick settings
        ax.tick_params(axis='x', labelsize=12, labelrotation=0)
        ax.tick_params(axis='y', labelsize=12, labelrotation=0)
        ax.tick_params(axis='z', labelsize=12, labelrotation=0)
        
        # Uniform font for all ticks
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            for tick in axis.get_ticklabels():
                tick.set_fontname('Times New Roman')

        # Save figure
        plt.savefig(f"{save_prefix}_{field_name}.png", dpi=1000, bbox_inches='tight')
        plt.show()
        
    # Plot displacement, strain, and stress
    plot_and_save(disp_points, disp_values, "displacement")
    plot_and_save(disp_points, strain_values, "strain")
    plot_and_save(disp_points, stress_values, "stress")


# Call the plotting function
fem_cloud_plot_all(
    coord=coord,
    U2=U2,
    elem=elem,
    HatP=HatP,
    DHatP1=DHatP1,
    DHatP2=DHatP2,
    DHatP3=DHatP3,
    Xi=Xi,
    WF=WF,
    E=3e7,       # Example: steel elastic modulus
    v=0.3,       # Poisson's ratio
    scale=1.0,   # Displacement scaling factor
    save_prefix="FEM_CloudPlot_HighDPI"
)
