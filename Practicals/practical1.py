## Derivation of Double Pebdulum Equations of Motion
#
# Derive the equations of motion of this system which is in the form of:
#  
# $$M(q) \ddot q + C(q, \dot q) \dot q + G(q) = 0$$ 
# 
# What are the formulas for matrices $M(q), C(q, \dot q)$ and $G(q)$? 

import numpy as np 
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.pyplot as plt
import sympy as sym
from sympy import *
init_printing(use_unicode=True)


# Generalized coordinates 
q1, q2, dq1, dq2, ddq1, ddq2 = symbols('q1 q2 dq1 dq2 ddq1 ddq2')

# System parameters
l1, l2, m1, m2, g = symbols('l1 l2 m1 m2 g')

""" Kinematics """
# Position and velocity of m1
print('-------------------------')
print('Position and velocity of m1:')
x1 = l1 * sym.sin(q1)
y1 = -l1 * sym.cos(q1)
dx1 = sym.diff(x1, q1) * dq1
dy1 = sym.diff(y1, q1) * dq1

print('x1:  ', x1)
print('y1:  ', y1)
print('dx1: ', dx1)
print('dy1: ', dy1)

# Position and velocity of m2
print('-------------------------')
print('Position and velocity of m2:')
x2 = l1 * sym.sin(q1) + l2 * sym.sin(q2)
y2 = -l1 * sym.cos(q1) - l2 * sym.cos(q2)
dx2 = sym.diff(x2, q1) * dq1 + sym.diff(x2, q2) * dq2
dy2 = sym.diff(y2, q1) * dq1 + sym.diff(y2, q2) * dq2

print('x2:  ', x2)
print('y2:  ', y2)
print('dx2: ', dx2)
print('dy2: ', dy2)

""" Dynamics """
# Kinetic and potential energies of m1 
print('-------------------------')
print('Kinetic and potential energies of m1:')
T1 = 1 / 2 * m1 * (dx1**2 + dy1**2)
V1 = -m1 * l1 * sym.cos(q1) * g
T1 = sym.simplify(T1)
V1 = sym.simplify(V1)

print('T1: ', T1)
print('V1: ', V1)

# Kinetic and potential energies of m2
print('-------------------------')
print('Kinetic and potential energies of m2:')
T2 = 1 / 2 * m2 * (dx2**2 + dy2**2);
V2 = -m2 * (l1 * sym.cos(q1) + l2 * sym.cos(q2)) * g
T2 = sym.simplify(T2) 
V2 = sym.simplify(V2)

print('T2: ', T2)
print('V2: ', V2)


""" Recall Lagrangian:
 $$L(q, \dot q) = T(q, \dot q) - V(q)$$
 Lagrange equations of motion:
 $\frac{d}{dt}(\frac{\partial L}{\partial \dot q_i }) - \frac{\partial L}{\partial q_i} = 0$ 
         for i = 1, 2
"""
print('-------------------------')
print('Calculate the Lagrangian of the system: ')
T = T1 + T2
T = sym.simplify(T)
V = V1 + V2
V = sym.simplify(V)
L = T - V

print('L: ', L)

"""
We use $dLddq$ as short for $\frac{\partial L}{\partial \dot q}$ and $dLdq$ 
for $\frac{\partial L}{\partial q}$. 
"""
print('-------------------------')
print('Calculate the partial derivatives of Lagrangian:')
dLddq1 = sym.diff(L, dq1);
dLddq2 = sym.diff(L, dq2);
dLdq1 = sym.diff(L, q1);
dLdq2 = sym.diff(L, q2);
dLddq1 = sym.simplify(dLddq1)
dLddq2 = sym.simplify(dLddq2)
dLdq1 = sym.simplify(dLdq1)
dLdq2 = sym.simplify(dLdq2)

"""
We use dLddq_dt for $\frac{d}{dt}(\frac{\partial L}{\partial \dot q})$
"""
print('-------------------------')
print('dLddq1', dLddq1)
dLddq1_dt = sym.diff(dLddq1, q1) * dq1   + sym.diff(dLddq1, q2) * dq2 + \
            sym.diff(dLddq1, dq1) * ddq1 + sym.diff(dLddq1, dq2) * ddq2

dLddq2_dt = sym.diff(dLddq2, q1) * dq1   + sym.diff(dLddq2, q2) * dq2 + \
            sym.diff(dLddq2, dq1) * ddq1 + sym.diff(dLddq2, dq2) * ddq2
print('dLddq1_dt', dLddq1_dt)
print('-------------------------')
print('Calculate equations of motion: ')
Eq1 = dLddq1_dt - dLdq1;
Eq2 = dLddq2_dt - dLdq2;
Eq1 = sym.simplify(Eq1)
Eq2 = sym.simplify(Eq2)

print('Eq1: ', Eq1)
print('Eq2: ', Eq2)


# Use the "subs" function 
print('-------------------------')
print('Calculate Mass matrix (M), Coriolis and gravity terms (C and  G):')
G = zeros(2,1) 
G[0,0] = Eq1.subs([(ddq1,0), (ddq2,0), (dq1,0), (dq2,0)]).simplify()
G[1,0] = Eq2.subs([(ddq1,0), (ddq2,0), (dq1,0), (dq2,0)]).simplify()
print('-------------------------\nG:\n',G)

M = zeros(2,2) 
M[0,0] = (Eq1.subs([(ddq1,1), (ddq2,0), (dq1,0), (dq2,0)]) - G[0])
M[0,1] = (Eq1.subs([(ddq1,0), (ddq2,1), (dq1,0), (dq2,0)]) - G[0])
M[1,0] = (Eq2.subs([(ddq1,1), (ddq2,0), (dq1,0), (dq2,0)]) - G[1])
M[1,1] = (Eq2.subs([(ddq1,0), (ddq2,1), (dq1,0), (dq2,0)]) - G[1])
M = simplify(M)
print('-------------------------\nM:\n', M)

C = zeros(2,2) 
C[0,0] = (Eq1.subs([(ddq1,0), (ddq2,0), (dq2,0)]) - G[0]) / dq1
C[0,1] = (Eq1.subs([(ddq1,0), (ddq2,0), (dq1,0)]) - G[0]) / dq2
C[1,0] = (Eq2.subs([(ddq1,0), (ddq2,0), (dq2,0)]) - G[1]) / dq1
C[1,1] = (Eq2.subs([(ddq1,0), (ddq2,0), (dq1,0)]) - G[1]) / dq2
C = simplify(C)
print('-------------------------\nC:\n', C)
print('-------------------------')

# create M, C, G functions to evaluate at certain points
eval_M = lambdify((l1,l2,m1,m2,q1,q2),M)
eval_C = lambdify((dq1,dq2,l1,l2,m2,q1,q2), C)
eval_G = lambdify((g,l1,l2,m1,m2,q1,q2), G)


def set_parameters():
    # sample parameters
    m1 = 1
    m2 = 1
    l1 = 0.5
    l2 = 0.5
    g = 9.81
    return m1, m2, l1, l2, g


def dynamics(y,t):
    # create dynamics
    m1, m2, l1, l2, g = set_parameters()

    q  = np.array([y[0],y[1]])
    dq = np.array([y[2],y[3]])

    M = eval_M(l1,l2,m1,m2,q[0],q[1])
    C = eval_C(dq[0],dq[1],l1,l2,m2,q[0],q[1])
    G = eval_G(g,l1,l2,m1,m2,q[0],q[1])

    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]
    dy[2:] = np.linalg.solve(M, (-C @ dq ).reshape(2,1) - G)[:,0]
    return dy

# create time array to evaluate dynamics
t = np.linspace(0,10,1001)
global dt 
dt = t[1] - t[0]
states = odeint(dynamics, y0=[0,np.pi/2,0,0], t=t)

fig = plt.figure()
plt.plot(t, states[:,0], label="q1")
plt.plot(t, states[:,1], label="q2")
plt.legend()
# plt.show()

# functions to return mass locations (x1,y1) (x2,y2)
eval_x1 = lambdify((q1,l1),x1)
eval_y1 = lambdify((q1,l1),y1)
eval_x2 = lambdify((q1,q2,l1,l2),x2) 
eval_y2 = lambdify((q1,q2,l1,l2),y2) 

def get_x1y1_x2y2(th1,th2):
    _, _, l1, l2, _ = set_parameters()
    return (eval_x1(th1,l1), 
            eval_y1(th1,l1),
            eval_x2(th1,th2,l1,l2),
            eval_y2(th1,th2,l1,l2))

x1, y1, x2, y2 = get_x1y1_x2y2(states[:,0], states[:,1])

fig = plt.figure()
ax = plt.gca()
ln1, = plt.plot([], [], 'ro-', lw=3, markersize=8)
ax.set_xlim(-1.2,1.2)
ax.set_ylim(-1.2,1.2)
plt.gca().set_aspect('equal', adjustable='box')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def animate(i):
    global dt
    ln1.set_data([0,x1[i],x2[i]], [0, y1[i], y2[i]])
    time_text.set_text('time = %.1f' % (float(i)*dt))

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=int(dt*1000))
plt.show()

