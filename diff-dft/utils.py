import numpy as np
import plotly.graph_objects as go

J_per_Ha = 4.3597447222071e-18
eV_per_Ha = J_per_Ha / 1.602176634e-19
A_per_B = 0.529177249

def plot3d(wave, cell):
    t = np.real(wave.detach().numpy())
    fig = go.Figure(data=[go.Scatter3d(
        x=cell.dx3[0].flatten(),
        y=cell.dx3[1].flatten(),
        z=cell.dx3[2].flatten(),
        mode='markers',
        marker=dict(
            size=12,
            color=t.flatten(),               
            colorscale='inferno', 
            opacity= 0.1
        ),
    )])

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()