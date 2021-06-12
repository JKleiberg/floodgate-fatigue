import numpy as np
from ipywidgets import interact, FloatSlider
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models import ColumnDataSource, HoverTool, Range1d
import ipywidgets as widget

import os
import sys
output_notebook(hide_banner = True);

root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

from src.configuration import C_ts, cr, N_HOURS, H_GATE, dz
from src.spec import spectrum_generator
from src.pressure import pressure

def plot_wl():
    source = ColumnDataSource(
        data=dict(
            t = np.linspace(0, N_HOURS*3.6E6, 200000),
            y = 7+np.zeros(200000),
            hs = np.zeros(200000),
            hsp = np.zeros(200000)
        )
    )

    p = figure(title="Random hourly water level simulations", plot_height=500, plot_width=950, 
            y_range = Range1d(4,10,bounds = (2,12)), x_range = Range1d(0,N_HOURS*3.6E6, 
            bounds = (0,N_HOURS*3.6E6)), x_axis_type='datetime')
    
    p.line(x='t', y='y', source=source, color="midnightblue", line_width=1.5, alpha=0.8, legend_label = 'Water level')
    p.line(x='t', y='hsp', alpha = 1, source = source, color = 'skyblue', line_width=3, legend_label = 'Hm0')
    p.line([0, N_HOURS*3.6E6], H_GATE, alpha = 1, color = 'dimgray', line_width=3, legend_label = 'Overhang level')

    p.toolbar.logo = None
    p.title.text_font_size = '18pt'
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Elevation [m]'
    p.xaxis.formatter = DatetimeTickFormatter(
            seconds = ['%Ss'],
            minsec = [':%M:%S'],
            minutes = [':%M', '%Mm'],
            hourmin = ['%H:%M'],
            hours = ['%H:%M'],
            days = ['%H:%M'])
    
    p.add_tools(HoverTool(tooltips=[("Water level", "@y"),("Sig. wave height", "@hs")]))
    p.legend.location = "top_left"
    p.legend.click_policy="hide"


    def update(U=10, d=10):
        ts = 0.1
        f,amp,k,Hs,Tp = spectrum_generator(U, d, spec = 'TMA')
        amp_refl = (1+cr)*amp
        eta = np.fft.irfft(amp_refl)*len(k)
        Hs_refl = (1+cr)*Hs
        source.data = {'t': np.linspace(0, N_HOURS*3.6E6, len(eta)),
                       'y': eta + d,
                       'hs': Hs_refl*np.ones(len(eta)),
                       'hsp': (Hs_refl/2+d)*np.ones(len(eta))
                       }
        push_notebook();

    show(p, notebook_handle=True);
    
    interact(update, 
             U    = widget.FloatSlider(value=10, min=0, max=40, step=0.1, continuous_update = False, description='$u_{10}$ [m/s]', readout_format='.1f'),
             d  = widget.FloatSlider(value=7,    min=4, max=9,  step=0.1, continuous_update = False, description='depth [m]', readout_format='.1f')
    );
    
## Quasi-static plot
def plot_qs():
    source_qs = ColumnDataSource(
        data=dict(
            f = np.linspace(0,2,1000),
            a = np.zeros(1000),))

    p = figure(title="Quasi-static pressure spectra", plot_height=500, plot_width=950)#, 
            #y_range = Range1d(4,10,bounds = (2,12)), x_range = Range1d(0,N_HOURS*3.6E6, 
            #bounds = (0,N_HOURS*3.6E6)))
    r = p.line(x='f', y='a', source=source_qs, color="midnightblue", line_width=1.5, alpha=0.8)
    p.x_range = Range1d(0,r.data_source.data['f'].max())
    p.y_range = Range1d(0,100)

    # g = 9.81
    # p.line(np.sqrt(2*np.pi*g/(20*7)*np.tanh(2*np.pi/20))/(2*np.pi), [0,100], alpha = 1, color = 'red')
    # p.line(np.sqrt(2*np.pi*g/(2*7)*np.tanh(2*np.pi/2))/(2*np.pi), [0,100], alpha = 1, color = 'black')

    p.toolbar.logo = None
    p.title.text_font_size = '18pt'
    p.xaxis.axis_label = 'Frequency [Hz]'
    p.yaxis.axis_label = 'Amplitude [N/m^2]'
    p.add_tools(HoverTool(tooltips=[("amplitude", "@a")]))

    def update_qs(U=10, hsea=7, Z = 4):
        i = int(Z/dz)
        f, pqs_f, p_imp_f,Hs,Tp = pressure(U,hsea);
        r.data_source.data = {'f': f[1:], 
                              'a': abs(pqs_f[i,1:])}
        p.x_range.end = r.data_source.data['f'].max()
        p.y_range.end = 1.2*r.data_source.data['a'].max()

        push_notebook();

    show(p, notebook_handle=True);
    interact(update_qs, 
                U    = FloatSlider(value = 10, min = 0.1, max=40, step=0.1, continuous_update = False),
                hsea  = FloatSlider(value = 7,   min = 4,   max=9,   step=0.1, continuous_update = False),
                Z     = FloatSlider(value = 4,   min = 0,   max=7.5, step=0.1, continuous_update = False)
            );