import sys
import os
import numpy as np
import dill
from src.configuration import z_coords, x_coords, dx, dz, H_GATE, WIDTH, H_LAKE, n_x, n_z, n_freqs, f_max, cp, rho

def partition(lst, n):
    division = (len(lst)-1) / n
    return [0,*[round(division * (i+1)) for i in range(n)]]

def fluidmodes_free(Nf, depth):
    modes = np.arange(Nf)+1
    fshape_kx = np.zeros(Nf)
    fshape_x = np.zeros([Nf, int(WIDTH/dx)+1])
    for p in modes:
        fshape_kx[p-1] = (p-1)*np.pi/WIDTH
        fshape_x[p-1] = np.cos(fshape_kx[p-1]*x_coords)    
    fshape_kz = np.zeros(Nf)
    fshape_z = np.zeros([Nf, int(depth/dz)+1])
    zrange = tuple([z_coords<=depth])
    for r in modes:
        fshape_kz[r-1] = (2*r-1)*np.pi/(2*depth)
        fshape_z[r-1] = np.cos(fshape_kz[r-1]*z_coords[zrange]) 
    P = np.zeros([len(x_coords), len(z_coords[zrange]), Nf**2])
    DZ = np.zeros(Nf**2)
    for p in modes:
        for r in modes:
            pr = (p-1)*Nf+r
            P[:,:,pr-1] = np.outer(fshape_x[p-1],fshape_z[r-1].T)
            DZ[pr-1] = np.trapz(x=x_coords, y=fshape_x[p-1]**2)*np.trapz(x=z_coords[zrange], y=fshape_z[r-1]**2) 
    return fshape_kx, fshape_x.T, fshape_kz, fshape_z.T, P, DZ

def fluidmodes_closed(Nf, depth):
    modes = np.arange(Nf)+1
    fshape_kx = np.zeros(Nf)
    fshape_x = np.zeros([Nf, int(WIDTH/dx)+1])
    for p in modes:
        fshape_kx[p-1] = (p-1)*np.pi/WIDTH
        fshape_x[p-1] = np.cos(fshape_kx[p-1]*x_coords)    
    fshape_kz = np.zeros(Nf)
    fshape_z = np.zeros([Nf, int(depth/dz)+1])
    zrange = tuple([z_coords<=depth])
    for r in modes:
        fshape_kz[r-1] = (r-1)*np.pi/(depth)
        fshape_z[r-1] = np.cos(fshape_kz[r-1]*z_coords[zrange]) 
    P = np.zeros([len(x_coords), len(z_coords[zrange]), Nf**2])
    DZ = np.zeros(Nf**2)
    for p in modes:
        for r in modes:
            pr = (p-1)*Nf+r
            P[:, :, pr-1] = np.outer(fshape_x[p-1], fshape_z[r-1].T)
            DZ[pr-1] = np.trapz(x=x_coords, y=fshape_x[p-1]**2)*np.trapz(x=z_coords[zrange], y=fshape_z[r-1]**2) 
    return fshape_kx, fshape_x.T, fshape_kz, fshape_z.T, P, DZ

def modalforce(GATE, Wxz):
    intervals = np.zeros([n_x, n_z, int(1+WIDTH/dx), int(1+H_GATE/dz)])
    for ii in range(n_x):
        for jj in range(n_z):
            intervals[ii, jj, GATE.ii_x[ii]:GATE.ii_x[ii+1]+1, GATE.ii_z[jj]:GATE.ii_z[jj+1]+1] = 1
    Fmode = np.zeros([GATE.n_modes, n_x, n_z])
    for ii in range(n_x):
        for jj in range(n_z):
            for mode in range(GATE.n_modes):
                xrange = slice(GATE.ii_x[ii], GATE.ii_x[ii+1])
                zrange = slice(GATE.ii_z[jj], GATE.ii_z[jj+1])
                Fmode[mode, ii, jj] = np.trapz(x=x_coords[xrange],
                                             y=np.trapz(x=z_coords[zrange],
                                                        y=intervals[ii, jj, xrange, zrange]*-Wxz[xrange, zrange, mode],
                                                        axis=1))
    return Fmode

def separation_constant(kf, f_kx, f_kz, n_modes):
    f_ky = np.zeros([n_modes**2, n_freqs], dtype=complex)
    for p1 in range(n_modes):
        for p2 in range(n_modes):
            p = p1*n_modes+p2+1
            check = kf**2-f_kx[p1]**2-f_kz[p2]**2
            f_ky[p-1][check<0] = -np.sqrt(check[check<0])
            f_ky[p-1][check>=0] = np.sqrt(check[check>=0])+1e-20+1j*10**-20
    return f_ky

def fluid_structure_interaction(GATE, scia_results, fluidmodes=[5,6,5]):
    dry_eigenfreqs, Wxz, col_disp, stress_pos, stress_neg, shear, faces, coords = scia_results
    GATE.ii_x = partition(x_coords, n_x)
    GATE.ii_z = partition(z_coords, n_z)

    f1_kx, f1_x, f1_kz, f1_z, f1_P, f1_DZ = fluidmodes_closed(fluidmodes[0], H_GATE) # Closed because of overhang
    f2_kx, f2_x, f2_kz, f2_z, f2_P, f2_DZ = fluidmodes_free(fluidmodes[1],   H_GATE)
    f3_kx, f3_x, f3_kz, f3_z, f3_P, f3_DZ = fluidmodes_free(fluidmodes[2],   H_LAKE)

    Fmode = modalforce(GATE, Wxz)

    Q = np.zeros([fluidmodes[0]**2, GATE.n_modes])
    for mode in range(GATE.n_modes):
        for p1 in range(fluidmodes[0]):
            for p2 in range(fluidmodes[0]):
                p = p1*fluidmodes[0]+p2
                Q[p, mode] = -np.trapz(x=x_coords,
                                     y=np.trapz(x=z_coords,
                                                y=np.multiply(Wxz[:,:,mode], f1_P[:,:,p]), axis=1)
                                    ) # Why minus sign?

    T = np.zeros([fluidmodes[2]**2, GATE.n_modes])
    for mode in range(GATE.n_modes):
        for p1 in range(fluidmodes[2]):
            for p2 in range(fluidmodes[2]):
                p = p1*fluidmodes[2]+p2
                T[p, mode] = -np.trapz(x=x_coords,
                                     y=np.trapz(x=z_coords[z_coords<=H_LAKE],
                                                y=np.multiply(Wxz[:, :f3_P.shape[1], mode], f3_P[:,:,p]), axis=1)
                                    ) # Why minus sign?

    R_intxL = np.zeros([fluidmodes[1], fluidmodes[0]])   
    R_intzL = np.zeros([fluidmodes[1], fluidmodes[0]]) 
    for r in range(fluidmodes[1]):
        for p in range(fluidmodes[0]):
            R_intxL[r,p] = np.trapz(x=x_coords, y=f2_x[:,r]*f1_x[:,p])
            R_intzL[r,p] = np.trapz(x=z_coords, y=f2_z[:,r]*f1_z[:,p])

    R = np.zeros([fluidmodes[0]**2, fluidmodes[1]**2])
    for r1 in range(fluidmodes[1]):
        for r2 in range(fluidmodes[1]):
            r = r1*fluidmodes[1]+r2
            for p1 in range(fluidmodes[0]):
                for p2 in range(fluidmodes[0]):
                    p = p1*fluidmodes[0]+p2
                    R[p,r] = R_intxL[r1,p1]*R_intzL[r2,p2]    

    GATE.f_tf = np.linspace(0, f_max, n_freqs)
    omega_tf = GATE.f_tf*2*np.pi
    kf = np.array(omega_tf/cp, dtype=complex)
    
    f1_ky = separation_constant(kf, f1_kx, f1_kz, fluidmodes[0])
    f2_ky = separation_constant(kf, f2_kx, f2_kz, fluidmodes[1])
    f3_ky = separation_constant(kf, f3_kx, f3_kz, fluidmodes[2])
    
    omega_n = np.array(dry_eigenfreqs)*2*np.pi*(1+GATE.zeta*1j)
    
    def solve_fsi(w):
        Aln_RAO_w = np.zeros([n_x, n_z, GATE.n_modes], dtype=complex)
        GW = np.ones(GATE.n_modes)
        i = omega_tf.tolist().index(w)
        Ik = (omega_n**2-w**2)*GW
        Ck = (GATE.cdamp*1j*w)*GW          #Why set to zero
        AK_1_1 = np.diag(Ik)+np.diag(Ck)
        AK_1_2 = np.zeros([GATE.n_modes, GATE.n_modes], dtype=complex)
        BPmin_1 = np.zeros([GATE.n_modes, fluidmodes[0]**2], dtype=complex)
        BPplus_1 = np.zeros([GATE.n_modes, fluidmodes[0]**2], dtype=complex)
        for km in range(GATE.n_modes):
            for ln in range(GATE.n_modes):
                AK_1_2[ln, km] = rho*1j*w**2*sum(T[:, km]*T[:, ln]/(f3_ky[:, i]*f3_DZ))
        for km in range(GATE.n_modes):
            for ln in range(fluidmodes[0]**2):           
                BPmin_1[km,ln]  = -rho*1j*w*Q[ln, km]*np.exp(-1j*f1_ky[ln,i]*GATE.Ly)
                BPplus_1[km,ln] = -rho*1j*w*Q[ln, km]
        AK_1 = AK_1_1+AK_1_2

        BPmin_2 = np.diag(f1_ky[:,i]*f1_DZ*np.exp(-1j*f1_ky[:,i]*GATE.Ly))
        BPplus_2 = np.diag(-f1_ky[:,i]*f1_DZ)
        AK_2 = -w*Q

        BPmin_3_1 = np.diag(f1_DZ)
        BPplus_3_1 = np.diag(f1_DZ*np.exp(-1j*f1_ky[:,i]*GATE.Ly))

        BPmin_3_2 = np.zeros([fluidmodes[0]**2, fluidmodes[0]**2], dtype=complex)
        BPplus_3_2 = np.zeros([fluidmodes[0]**2, fluidmodes[0]**2], dtype=complex)
        for p in range(fluidmodes[0]**2):
            for q in range(fluidmodes[0]**2):
                BPmin_3_2[q,p] = f1_ky[p,i]*sum(R[p,:]*R[q,:]/(f2_ky[:,i]*f2_DZ))
                BPplus_3_2[q,p] = -f1_ky[p,i]*np.exp(-1j*f1_ky[p,i]*GATE.Ly)*sum(R[p,:]*R[q,:]/(f2_ky[:,i]*f2_DZ))
        BPmin_3 = BPmin_3_1+BPmin_3_2
        BPplus_3 = BPplus_3_1+BPplus_3_2
        AK_3 = np.zeros([fluidmodes[0]**2,GATE.n_modes])

        # Full Matrix Determination
        M = np.bmat([[AK_1, BPplus_1, BPmin_1],[AK_2, BPplus_2, BPmin_2],[AK_3, BPplus_3, BPmin_3]])
        for ii in range(n_x):
            for jj in range(n_z):
                F1n_vec = Fmode[:,ii,jj]
                Fm = np.bmat([F1n_vec, np.zeros(fluidmodes[0]**2), np.zeros(fluidmodes[0]**2)])
                MC = Fm*np.linalg.inv(M)
                Aln_RAO_w[ii,jj,:] = MC[0,:GATE.n_modes]
        # Two below aren't needed for gate solution
        #     Bplus_RAO = np.zeros([n_x, n_z, GATE.n_modes, n_freqs])
        #     Bmin_RAO = np.zeros([n_x, n_z, GATE.n_modes, n_freqs]) # Error in Matlab? Says Bplus
        return Aln_RAO_w

    GATE.FRF = np.array(list(map(solve_fsi, omega_tf))).transpose(1,2,3,0)

    return GATE

                    