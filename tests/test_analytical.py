# -*- coding: utf-8 -*-
"""
A series of test based on an analytical solution to simple
network problem.


"""

import pywr.core
import datetime
import numpy as np

def make_simple_model(supply_amplitude, demand, frequency,
                      initial_volume):
    """
    Make a simlpe model,
        supply -> reservoir -> demand.
        
    supply is a annual cosine function with amplitude supply_amplitude and
    frequency
    
    """
    
    model = pywr.core.Model()
    
    S = supply_amplitude
    w = frequency
    
    def supply_func(parent, index):
        t = parent.model.timestamp.timetuple().tm_yday
        return S*np.cos(t*w)+S
    
    supply = pywr.core.Supply(model, name='supply', max_flow=supply_func)
    demand = pywr.core.Demand(model, name='demand', demand=demand)
    res = pywr.core.Reservoir(model, name='reservoir')
    res.properties['max_volume'] = pywr.core.ParameterConstant(1e6)
    res.properties['current_volume'] = pywr.core.Variable(initial_volume)
    
    supply_res_link = pywr.core.Link(model, name='link1')
    res_demand_link = pywr.core.Link(model, name='link2')
    
    supply.connect(supply_res_link)
    supply_res_link.connect(res)
    res.connect(res_demand_link)
    res_demand_link.connect(demand)
    
    return model
    
def test_run_analytical():
    """
    Run the test model though a year with analytical solution values to 
    ensure reservoir just contains sufficient volume.
    """
    
    S = 100.0 # supply amplitude
    D = S # demand    
    w = 2*np.pi/365 # frequency (annual)
    V0 = S/w  # initial reservoir level
    
    model = make_simple_model(S, D, w, V0)
    
    model.timestamp = datetime.datetime(2015, 1, 1)
    
    # TODO include first timestep
    T = np.arange(1,365)
    V_anal = S*(np.sin(w*T)/w+T) - D*T + V0
    V_model = np.empty(T.shape)
    
    for i,t in enumerate(T):
        model.step()
        for node in model.nodes():
            if 'current_volume' in node.properties:
                V_model[i] = node.properties['current_volume'].value()        
                
        
    return T, V_model, V_anal
    
    
    
if __name__ == '__main__':
    
    t, Vm, Va = test_run_analytical()
    
    error = np.abs(Vm-Va)/Va
    
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2,sharex=True)  
    
    ax1.plot(t, Va, label='Analytical')
    ax1.plot(t, Vm, '-o', label='Model')    
    
    ax1.set_ylabel('Volume')
    
    ax2.plot(t, error, '-o', label='Error')    
    ax2.set_ylabel('Error [%]')
    ax2.set_xlabel('Day of Year')
    
    plt.show()
