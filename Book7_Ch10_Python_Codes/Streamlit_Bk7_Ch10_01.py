import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import streamlit as st

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

def generate_PDFs(mu1,sigma1,mu2,sigma2,P_C1):
    # meshgrid for plotting
    xx1, xx2 = np.mgrid[-6:6:0.1, -6:6:0.1]
    xx1_xx2  = np.dstack((xx1, xx2))
    
    
    P_C2 = 1 - P_C1;
    
    pdf_x_C1 = multivariate_normal(mu1, sigma1)
    pdf_x_C2 = multivariate_normal(mu2, sigma2)
    
    pdf_x_C1_yy = pdf_x_C1.pdf(xx1_xx2)
    pdf_x_C2_yy = pdf_x_C2.pdf(xx1_xx2)
    
    pdf_x_and_C1 = pdf_x_C1_yy*P_C1;
    pdf_x_and_C2 = pdf_x_C2_yy*P_C2;
    pdf_diff = pdf_x_and_C1 - pdf_x_and_C2;
    return pdf_x_and_C1,pdf_x_and_C2,xx1,xx2,pdf_diff

def plot_contours(pdf_x_and_C1,pdf_x_and_C2,xx1,xx2,pdf_diff):
     
    
    fig, ax = plt.subplots(figsize = (5,5))
    ax.contour(xx1, xx2, pdf_x_and_C1, alpha = 0.8, levels = 15, cmap="RdBu_r")
    ax.contour(xx1, xx2, pdf_x_and_C2, alpha = 0.8, levels = 15, cmap="RdBu_r")
    ax.contour(xx1, xx2, pdf_diff, [0], alpha = 1, colors = 'k')

    
    # Figure decorations
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    plt.axis('equal')
    plt.show()
    return fig, ax

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

#%%

with st.sidebar:
    st.title('高斯判别分析')
    
    P_C1_0 = st.slider('Pr of C1', 0.1, 0.9, 0.4, 0.1)
    
    mu1_x = st.slider('mu1, X', -2.0, 2.0, -1.0, 0.1)
    mu1_y = st.slider('mu1, Y', -2.0, 2.0, -1.0, 0.1)
    
    sigma1_x = st.slider('sigma1, X', 0.2, 3.0, 1.0, 0.1)
    sigma1_y = st.slider('sigma1, Y', 0.2, 3.0, 1.0, 0.1)
    rho1_xy  = st.slider('rho1', -0.9, 0.9, 0.0, 0.1)
    
    SIGMA1 = [[sigma1_x**2, sigma1_x * sigma1_y * rho1_xy],
              [sigma1_x * sigma1_y * rho1_xy, sigma1_y**2]]
    SIGMA1 = np.array(SIGMA1)
    
    mu1 = [mu1_x, mu1_y]
    mu1 = np.array(mu1)

    mu2_x = st.slider('mu2, X', -2.0, 2.0, 1.0, 0.1)
    mu2_y = st.slider('mu2, Y', -2.0, 2.0, 1.0, 0.1)
    
    sigma2_x = st.slider('sigma2, X', 0.2, 3.0, 2.0, 0.1)
    sigma2_y = st.slider('sigma2, Y', 0.2, 3.0, 2.0, 0.1)
    rho2_xy  = st.slider('rho2', -0.9, 0.9, 0.0, 0.1)
    
    SIGMA2 = [[sigma2_x**2, sigma2_x * sigma1_y * rho1_xy],
              [sigma2_x * sigma2_y * rho1_xy, sigma2_y**2]]
    SIGMA2 = np.array(SIGMA2)
    
    mu2 = [mu2_x, mu2_x]
    mu2 = np.array(mu2)  
    
#%%
st.latex(r'\mu_1 = ' + bmatrix(mu1))
st.latex(r'\Sigma_1 = ' + bmatrix(SIGMA1))
st.latex(r'\mu_2 = ' + bmatrix(mu2))
st.latex(r'\Sigma_2 = ' + bmatrix(SIGMA2))



pdf_x_and_C1,pdf_x_and_C2,xx1,xx2,new_pdf_diff = generate_PDFs(mu1,SIGMA1,mu2,SIGMA2,P_C1_0)

C_plot,C_ax = plot_contours(pdf_x_and_C1,pdf_x_and_C2,xx1,xx2,new_pdf_diff)

st.pyplot(C_plot)
