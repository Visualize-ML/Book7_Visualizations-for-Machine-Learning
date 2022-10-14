
###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def generate_PDFs(mu1,sigma1,mu2,sigma2,P_C1):
    # meshgrid for plotting
    xx1, xx2 = np.mgrid[-6:6:0.1, -6:6:0.1]
    xx1_xx2 = np.dstack((xx1, xx2))
    
    
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
     
    
    fig, ax = plt.subplots()
    ax.contour(xx1, xx2, pdf_x_and_C1, alpha = 0.8, levels = 15, cmap="RdBu_r")
    ax.contour(xx1, xx2, pdf_x_and_C2, alpha = 0.8, levels = 15, cmap="RdBu_r")
    ax.contour(xx1, xx2, pdf_diff, [0], alpha = 1, colors = 'k')

    
    # Figure decorations
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    plt.axis('scaled')
    plt.show()
    return fig, ax

#%%

# main function
# Centroids of distributions, mean of the distributions
mu1 = [-1, 1];
mu2 = [1, -1];

# # hyperbola and degenerate hyperbola
sigma1 = [[1, 0], [0, 9]];
sigma2 = [[9, 0], [0, 1]];

P_C1_0 = 0.4;

pdf_x_and_C1,pdf_x_and_C2,xx1,xx2,new_pdf_diff = generate_PDFs(mu1,sigma1,mu2,sigma2,P_C1_0)

C_plot,C_ax = plot_contours(pdf_x_and_C1,pdf_x_and_C2,xx1,xx2,new_pdf_diff)

# plot a axis for slider
axcolor = 'lightgoldenrodyellow'

ax_P_C1 = plt.axes([0.2, 0.925, 0.5, 0.03], facecolor=axcolor)
#  [left, bottom, width, height]

s_P_C1 = Slider(ax_P_C1, 'PC1', 0.1, 0.9, valinit=0.1, valstep=0.05)
# Define a slider, value in range of [2, 60], step size = 2

def update(val):

    new_P_C1 = s_P_C1.val
    new_pdf_x_and_C1,new_pdf_x_and_C2,xx1,xx2,new_pdf_diff = generate_PDFs(mu1,sigma1,mu2,sigma2,new_P_C1)
    
    C_ax.cla()
    C_ax.contour(xx1, xx2, new_pdf_x_and_C1, alpha = 0.8, levels = 15, cmap="RdBu_r")
    C_ax.contour(xx1, xx2, new_pdf_x_and_C2, alpha = 0.8, levels = 15, cmap="RdBu_r")
    C_ax.contour(xx1, xx2, new_pdf_diff, [0], alpha = 1, colors = 'k')

    # Figure decorations
    C_ax.set_xlim(xx1.min(), xx1.max())
    C_ax.set_ylim(xx2.min(), xx2.max())
    
    C_ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    C_ax.set_xlabel('$x_1$')
    C_ax.set_ylabel('$x_2$')

    plt.axis('scaled')
    plt.show()
    print(new_P_C1)
    print(new_pdf_x_and_C2.max())


s_P_C1.on_changed(update)

# add a reset button
reset_ax = plt.axes([0.8, 0.925, 0.075, 0.03])
button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_P_C1.reset()

button.on_clicked(reset)

# Figure decorations
C_ax.set_xlim(xx1.min(), xx1.max())
C_ax.set_ylim(xx2.min(), xx2.max())

C_ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
C_ax.set_xlabel('$x_1$')
C_ax.set_ylabel('$x_2$')

plt.axis('scaled')
plt.show()
