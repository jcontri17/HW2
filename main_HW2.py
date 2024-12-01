import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt





###------PROBLEM SETUP
L = 4       			#Domain limit
dx = 0.1    			#size of dx (x-step)
xstart = -L
xstop = L+dx
xspan = np.arange(xstart, xstop, dx)	#array of x-values
K = 1				#given in problem statement
n = 5				#first n normalized eigenfunctions(phi_n) and eigenvalues(eps_n)
tol = 1e-6			#standard tolerance       



###------SCHRODINGER ODE SYSTEM FUNCTION
def schrodinger(x, y, eps):
    phi = y[0]
    phi_prime = y[1]
    dphi_dx = phi_prime
    dphi_prime_dx = (K*x**2 - eps) * phi
    return [dphi_dx, dphi_prime_dx]



###-----SHOOTING METHOD FUNCTION
def shooting_method(eps_guess):
    #Initial Conditions
    phi0 = 0        #phi(-L) = 0
    phi_prime0 = 1  #guess phi'(0) to start

    #Integrate Schrodinger equation using solve_ivp
    sol = solve_ivp(schrodinger, [xstart,xstop], [phi0, phi_prime0], t_eval=xspan, args=(eps_guess,))
    return sol.y[0] #returns phi(x)



###-----FUNCTION TO SOLVE FOR EIGENVALUES WITH BISECTION METHOD
def eigValFun(eps_min, eps_max, mode):
	eps_mid = (eps_max + eps_min) /2
	psi_x = shooting_method(eps_mid)[-1]
	while abs(psi_x) > tol:
		print("eps_min =",eps_min,"eps_max =",eps_max,"psi_eps=",psi_x)
		#eps_mid = (eps_max + eps_min) /2
		#psi_x = shooting_method(eps_mid)[-1]
		if (-1)**(mode+1)*psi_x > 0:
			eps_max = eps_mid
		else:
			eps_min = eps_mid
		
		eps_mid = (eps_max + eps_min) /2
		psi_x = shooting_method(eps_mid)[-1]
	return eps_mid



###-----NORMALIZE THE EIGENFUNTIONS
def normalize(phi):
	print(phi)
	norm = np.sqrt(np.trapz(phi**2, xspan))
	return phi/norm



###-----MAIN PROGRAM
eigval = np.zeros(n)			#creates an nx1 array for the eigenvalues
eigfun = np.zeros((xspan.size,n))	#creates a 2L/dx x 1 array for the eigenfunction values

eps_start = 0				#Initial guess for lower bound of epsilon in eigenvalue function
eps_offset = 2				#Some offset value to create an initial upper bound for finding the epsilon


for i in range(n):
	#eigval[i] = eigValFun(2*i+eps_start,2*(i+1)-eps_start,i)
	eigval[i] = eigValFun(eps_start, (i+1)*eps_offset, i)
	print("Eigenvalue",i+1,":", eigval[i])

	#Plot the Eigenfunctions
	eigfun[:,i] = normalize(shooting_method(eigval[i])) #phi(eps) 2L/dx x 1 array
	#eigVal = eigval[i][0]
	eigVal = eigval[i]
	label_i = (f"$\\phi_{{{i+1}}}: \t \\epsilon_{{{i+1}}} = {eigVal:.2f}$")
	plt.plot(xspan, eigfun[:,i], label=label_i)
	eps_start = eigval[i]		#Sets the next initial guess of epsilon



###------SAVING THE ANSWERS
A1 = eigfun
print("A1 size = ", A1.shape)
np.save('A1.npy', A1)

A2 = eigval.T
#print("A2 size = ", A2.shape)
np.save('A2.npy', A2)



###-----FINISHING PARAMETERS FOR PLOTTING
plt.xlabel("x")
plt.ylabel("$\\phi(x)$")
plt.legend()
plt.grid(True)
plt.show()


#print(2*L/dx)
print(np.zeros(n))
print(len(xspan))




