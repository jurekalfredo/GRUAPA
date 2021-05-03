
from numpy import pi
import numpy as np


LARGO_PUENTE = 20000
CANT_PUNTOS = 100
#primer punto del arco
px_3 = -(LARGO_PUENTE/1000)
py_3 = LARGO_PUENTE/2
pz_3 = 0
#segundo punto del arco
px_2 = 0
py_2 = 0
pz_2 = 0
#tercer punto del arco
px_1 = -(LARGO_PUENTE/1000)
py_1 = (LARGO_PUENTE/2)*-1
pz_1 = 0


def rotation_matrix(axis, theta):
    
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - (cc) - (dd), 2 * (bc + ad), 2 * (bd - (ac))],
                     [2 * (bc - ad), aa + cc - (bb) - (dd), 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - (bb) - (cc)]])

A = np.array([px_1, py_1, pz_1])
B = np.array([px_2, py_2, pz_2])
C = np.array([px_3, py_3, pz_3])


a = np.linalg.norm(C - (B))
b = np.linalg.norm(C - (A))
c = np.linalg.norm(B - (A))



s = (a + b + c) / 2
R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
radio=int(R)
#print('radio '+str(radio))
b1 = a*a * (b*b + c*c - (a*a))
b2 = b*b * (a*a + c*c - (b*b))
b3 = c*c * (a*a + b*b - (c*c))
P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
######print(P)
P /= (b1)+(b2)+(b3)
Px = P[0]
Py = P[1]
Pz = P[2]

px_ini = px_1 - Px
py_ini = py_1 - Py
pz_ini = pz_1 - Pz
px_mid = px_2 - Px
py_mid = py_2 - Py
pz_mid = pz_2 - Pz
px_fin = px_3 - Px
py_fin = py_3 - Py
pz_fin = pz_3 - Pz


a_vec = np.array([px_ini, py_ini, pz_ini])/np.linalg.norm(np.array([px_ini, py_ini, pz_ini]))
b_vec = np.array([px_fin, py_fin, pz_fin])/np.linalg.norm(np.array([px_fin, py_fin, pz_fin]))
c_vec = np.array([px_mid, py_mid, pz_mid])/np.linalg.norm(np.array([px_mid, py_mid, pz_mid]))


vec_a = np.array([a_vec[0], a_vec[1], a_vec[2]])
vec_b = np.array([b_vec[0], b_vec[1], b_vec[2]])
vec_c = np.array([c_vec[0], c_vec[1], c_vec[2]])

axis = np.cross(a_vec, c_vec)

axis_p = np.array([axis[0], axis[1], axis[2]])
#resultado  = c_vec[0]*c_vec[1]



vec_a = np.array([round(a_vec[0],3), round(a_vec[1],3), round(a_vec[2],3)])
vec_b = np.array([round(b_vec[0],3), round(b_vec[1],3), round(b_vec[2],3)])

vec_1 = np.dot(a_vec,c_vec)
vec_2 = np.dot(b_vec,c_vec)

angulo_ab1 = np.arccos(np.dot(c_vec,a_vec))
angulo_ab2 = np.arccos(np.dot(c_vec,b_vec))
angulo_ab = (angulo_ab1) + (angulo_ab2)

v = [px_ini, py_ini, pz_ini]


c_vec = np.cross(a_vec,b_vec)


vec_c = np.array([c_vec[0], c_vec[1], c_vec[2]])
radio_=  np.array([P[0], P[1], P[2]])  
angulo_ab_grados = np.degrees(angulo_ab)

numWayPts = int(abs(b/100))   
print('CANT DE PUNTOS '+str(numWayPts))
theta_Grados = (angulo_ab_grados / (numWayPts))

grados_actual = theta_Grados
print('puntos: '+str(A))
for i in range(numWayPts):

    theta = np.radians(grados_actual)
    #print( theta)
    resul = rotation_matrix(axis, theta)
    #print( resul)
    new_pt = np.dot(rotation_matrix(axis, theta), v)
    #print( new_pt)
    lCX = round(new_pt[0] + Px,2)
    lCY = round(new_pt[1] + Py,2)
    lCZ = round(new_pt[2] + Pz,2)  
    grados_actual += theta_Grados

    pos_F = lCX, lCY,lCZ
    print('puntos: '+str(pos_F))
    

