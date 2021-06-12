import clr
import os
import sys
import re
import subprocess
import cloudpickle
import numpy as np
import xml.dom.minidom as md
from scipy.interpolate import Rbf
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)
clr.AddReference(r"C:\Program Files (x86)\SCIA\Engineer19.1\OpenAPI_dll\Scia.OpenAPI.dll")
directory = os.path.dirname(os.path.dirname(root_dir))+'\\data\\05_SCIA\\SCIA_model'
from src.configuration import x_coords, z_coords

def write_xml_parameters(GATE): 
    """"Writes gate parameters to XML file which SCIA can read"""   
    with open(directory+r"\SCIA.xml") as xml:
        xmlNode = md.parse(xml)
    objects = xmlNode.getElementsByTagName('obj')
    counter = 0
    for obj in objects:
        for key in GATE.GEOMETRY.keys():
            if key==obj.getAttribute('nm'):
                val = obj.getElementsByTagName('p0')[0].setAttribute('v', str(GATE.GEOMETRY[key]))
                counter+=1
    if counter != len(objects):
        print('ERROR: Amount of inputs (%s) does not match XML entries (%s), check names.'
              %(len(objects), len(GATE.GEOMETRY.keys())))
    else:
        with open(directory+r"\SCIA.xml", "w") as new_xml:  
            xmlNode.writexml(new_xml, encoding="UTF-8")
    return

def run_scia():
    """Runs SCIA model based on gate parameters from XML and saves output in new XML"""
    print('Running SCIA analysis... (takes a few minutes)')
    executable = r"C:\\Program Files (x86)\\SCIA\\Engineer19.1\\Esa_XML.exe"
    analysis = "EIG"
    scia_file = "%s\\SCIA.esa"%directory
    inputs = "%s\\SCIA.xml"%directory
    outputs = "/x%s\\SCIAResults.xml"%directory
    newmodel = "/o%s\\Updated.esa"%directory
    process = subprocess.Popen([executable, analysis, scia_file, inputs, outputs], 
                               # newmodel], re-enable to store updated scia model
                               shell=True, stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE, close_fds=True)
    output, err = process.communicate()
    print('Finished SCIA simulation.')
    return

def read_scia_output(GATE):
    xmlNode = md.parse(directory+r"\\SCIAResults.xml")
    containers = xmlNode.getElementsByTagName("project")[0].getElementsByTagName('container')
    p1s = containers[0].getElementsByTagName('p1')
    freqs = [float(p1s.item(i).getAttribute('v')) for i in range(len(p1s))]

    disp_ids = [item.getAttribute('v') for item in containers.item(2).getElementsByTagName('p1')]
    d_nodes = [int(p_id.split()[1]) for p_id in disp_ids]
    n_disp = len(disp_ids)

    ## Coordinates
    xyz = ['p2','p3','p4']
    p_x, p_y, p_z = [containers.item(5).getElementsByTagName(axis) for axis in xyz]
    disp_coords = [tuple([float(p_x.item(j).getAttribute('v')),float(p_y.item(j).getAttribute('v')),float(p_z.item(j).getAttribute('v'))]) for j in range(n_disp)]
    coords = [[] for x in range(len(np.unique(d_nodes)))]
    for i, node in enumerate(d_nodes):
        coords[node-1] = disp_coords[i]
    coords = np.array(coords)

    ## Displacement
    u_y = np.zeros([GATE.n_modes, n_disp])
    for i in range(GATE.n_modes):
        # Skip the first two. Next 16 are displacement modes
        p_u_y = containers[i+2].getElementsByTagName('p7')
        u_y[i] = [float(p_u_y[j].getAttribute('v')) for j in range(len(d_nodes))]
    col_disp = np.zeros([len(np.unique(d_nodes)), GATE.n_modes])
    for i, node in enumerate(d_nodes):
        for j in range(GATE.n_modes):
            if abs(u_y[j,i]) > abs(col_disp[node-1,j]):
                col_disp[node-1,j] = u_y[j,i]

    ## 3D stress
    p_ids = containers.item(2+GATE.n_modes).getElementsByTagName('p1')
    ids = [p_id.getAttribute('v') for p_id in p_ids]#[1:]]
    s_nodes = [int("".join(re.findall('\d+', p_id.split(';')[1]))) for p_id in ids]
    elements = [int("".join(re.findall('\d+', p_id.split(';')[0]))) for p_id in ids]
    eq_plus = np.zeros([GATE.n_modes, len(s_nodes)])
    eq_min = np.zeros([GATE.n_modes, len(s_nodes)])
    tau_max = np.zeros([GATE.n_modes, len(s_nodes)])
    for i in range(GATE.n_modes):
        ii = i+2+GATE.n_modes
        p_eq_plus = containers[ii].getElementsByTagName('p10')
        p_eq_min = containers[ii].getElementsByTagName('p11')
        p_tau = containers[ii].getElementsByTagName('p14')
        eq_plus[i] = [float(p_eq_plus[j].getAttribute('v')) for j in range(len(s_nodes))]
        eq_min[i] = [float(p_eq_min[j].getAttribute('v')) for j in range(len(s_nodes))]
        tau_max[i] = [float(p_tau[j].getAttribute('v')) for j in range(len(s_nodes))]

    stress_pos = np.zeros([len(np.unique(s_nodes)), GATE.n_modes])
    stress_neg = np.zeros([len(np.unique(s_nodes)), GATE.n_modes])
    shear = np.zeros([len(np.unique(s_nodes)), GATE.n_modes])
    faces = np.zeros([len(np.unique(elements)), 4], dtype='uint16')
    counter = 0
    for i, node in enumerate(s_nodes):
        for j in range(GATE.n_modes):
            if abs(eq_plus[j,i]) > abs(stress_pos[node-1,j]): # Why largest? Ask Ruben
                stress_pos[node-1,j] = eq_plus[j,i]
            if abs(eq_min[j,i])  > abs(stress_neg[node-1,j]):
                stress_neg[node-1,j] = eq_min[j,i]
            if abs(tau_max[j,i]) > abs(shear[node-1,j]):
                shear[node-1,j]      = tau_max[j,i]
        faces[elements[i]-1, counter] = node
        if counter <3:
            counter+=1
        else:
            counter=0
    xx, zz = np.meshgrid(x_coords, z_coords, indexing='ij')
    x = coords[:,0]
    y = coords[:,2]
    Wxz = np.zeros([*xx.shape, GATE.n_modes])
    for mode in range(GATE.n_modes):
        z = col_disp[:,mode]
        spline = Rbf(x,y,z,function='thin_plate', smooth=0.00001)
        Wxz[:,:,mode] = spline(xx,zz)
            
    res = [freqs, Wxz, col_disp, stress_pos, stress_neg, shear, faces, coords]
    with open('../data/05_SCIA/'+str(GATE.case)+'/dry_modes.cp.pkl', 'wb') as file:
            cloudpickle.dump(res, file)
    print('Successfully read mode shapes from XML. File created at data/05_SCIA/%s'%GATE.case+r'/dry_modes.cp.pkl.')
    return res
