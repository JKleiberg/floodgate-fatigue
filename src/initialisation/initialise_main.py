import dill
import os
import numpy as np
from src.configuration import t, dt, n_x, n_z, rho_s, dx, dz
from scipy.interpolate import InterpolatedUnivariateSpline
from src.initialisation.SCIA_coupling import write_xml_parameters, run_scia, read_scia_output
from src.initialisation.fsi_model import fluid_structure_interaction

def initialisation(GATE, overwrite=False):
    # Create directory if necessary
    scia_directory = '../data/05_SCIA/'+str(GATE.case)
    if not os.path.exists(scia_directory):
        os.mkdir(scia_directory)
        print("Created new directory at: "+str(scia_directory))
        flag_runmodel = True
    else:
        # If folder exists, check whether SCIA model has to be run
        with open(scia_directory+'/GATE.pkl', 'rb') as file:
            GATE_stored = dill.load(file)
        if hash(tuple(GATE.GEOMETRY.values())) == hash(tuple(GATE_stored.GEOMETRY.values())):
            if overwrite:
                print('Overwriting existing modes...')
                flag_runmodel = True
            else:
                print('Loading existing modes...')
                flag_runmodel = False
        else:
            print('Running SCIA to generate modes for new design...')
            flag_runmodel = True
    if flag_runmodel:
        GATE.x_coords = np.linspace(0, GATE.WIDTH, int(GATE.WIDTH/dx+1))
        GATE.z_coords = np.linspace(0, GATE.HEIGHT, int(GATE.HEIGHT/dz+1))
        write_xml_parameters(GATE)
        run_scia()
        GATE = read_scia_output(GATE)
        with open(scia_directory+'/GATE.pkl', 'wb') as file:
                dill.dump(GATE, file)

    ## Frequency Response Function(s)
    frf_directory = '../data/06_transferfunctions/'+str(GATE.case)
    frf_file = '/FRF_'+str(GATE.case)+'_'+str(GATE.n_modes)+'modes.npy'
    configfile = '/'+str(GATE.case)+'_properties.cp.pkl'
    if flag_runmodel:
        print('Creating new frequency response function(s)')
        if not os.path.exists(frf_directory):
            os.mkdir(frf_directory)
            print("Created new FRF directory at: "+str(frf_directory))
        GATE = fluid_structure_interaction(GATE, fluidmodes=[5,6,5])
        
        # Prepare interpolated FRF
        GATE.f_intpl = np.fft.rfftfreq(t.size, d=dt)
        FRF_intpl = np.zeros((n_x, n_z, GATE.n_modes, len(GATE.f_intpl)), dtype='complex') 
        for i in range(n_x):
            for j in range(n_z):
                for k in range(GATE.n_modes):
                    spl_imag = InterpolatedUnivariateSpline(GATE.f_tf, GATE.FRF[i][j][k].imag)
                    spl_real = InterpolatedUnivariateSpline(GATE.f_tf, GATE.FRF[i][j][k].real)
                    FRF_intpl[i,j,k,:] = 1j*spl_imag(GATE.f_intpl)+spl_real(GATE.f_intpl)
        with open(frf_directory+frf_file, 'wb') as file:
            np.save(file, FRF_intpl)
        print('Created new interpolated frequency response function(s) for case %s.'%GATE.case)
        GATE.MASS = rho_s*(GATE.GEOMETRY.Platethickness*GATE.HEIGHT*GATE.WIDTH +
            (GATE.GEOMETRY.RibThicknessHor+GATE.GEOMETRY.RibThicknessHor2+
             GATE.GEOMETRY.RibThicknessHor3+GATE.GEOMETRY.RibThicknessHor4)*GATE.WIDTH*GATE.GEOMETRY.LengthHorRibs + 
             (GATE.GEOMETRY.RibThicknessVert+GATE.GEOMETRY.RibThicknessVert2+
              GATE.GEOMETRY.RibThicknessVert3+GATE.GEOMETRY.RibThicknessVert4)*
             GATE.HEIGHT*GATE.GEOMETRY.LengthVertRibs -
             20*GATE.GEOMETRY.RibThicknessHor*GATE.GEOMETRY.RibThicknessVert*GATE.GEOMETRY.LengthVertRibs)
        with open(frf_directory+configfile, 'wb') as file:
            dill.dump(GATE, file)
    else:
        with open(frf_directory+configfile, 'rb') as file:
            GATE = dill.load(file)
            print('Loaded existing frequency response function(s) and gate properties for case %s.'%GATE.case)
    with open('../data/06_transferfunctions/current_case.pkl', 'wb') as file:
        dill.dump(GATE, file)
    analysis_folder = '../data/08_analysis/%s'%GATE.case
    if not os.path.exists(analysis_folder):
        os.mkdir(analysis_folder)
    return GATE