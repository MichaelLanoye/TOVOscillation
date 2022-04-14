import numpy as np
import math
from scipy.optimize import curve_fit, newton, brentq, minimize
import cmath
import re
import matplotlib.pyplot as plt

### rk4 class for scaled TOV ###
class STOVStep:
    "RK4 class for Chandra's dimensionless TOV."

    def __init__(self, r, y, dr, n, a0):
        "Initializes the Scaled TOV Step class."

        self.r = r
        self.y = y
        self.dr = dr
        self.n = n
        self.a0 = a0

        self.f = np.zeros(2, dtype=float)
        self.k1 = np.zeros(2, dtype=float)
        self.k2 = np.zeros(2, dtype=float)
        self.k3 = np.zeros(2, dtype=float)
        self.k4 = np.zeros(2, dtype=float)

    def _deriv(self, r, y):
        "Calculates the derivative for a given r and y."

        #0: theta => e and p are functions of theta
        #1: mass
        self.f[0] = -(self.a0 + y[0])*(y[1]/(r**2) + np.abs(y[0])**(self.n + 1)*r/self.a0)/(
            (self.n + 1)*(1 - 2*y[1]/r)
            )
        self.f[1] = (np.abs(y[0])**self.n)*(r**2)

        return self.f

    def rk4step(self):
        "Processes a single step in the RK4 routine."

        if self.y[0] < 10**(-8):
            self.dr = 10**(-11)
        elif self.y[0] < 10**(-6):
            self.dr = 10**(-8)
        elif self.y[0] < 10**(-4):
            self.dr = 10**(-6)
        else:
            self.dr = 10**(-4)

        self.k1 = self.dr * self._deriv(
            self.r,
            self.y
        )

        self.k2 = self.dr * self._deriv(
            self.r + self.dr/2,
            self.y + self.k1/2
        )

        self.k3 = self.dr * self._deriv(
            self.r + self.dr/2,
            self.y + self.k2/2
        )

        self.k4 = self.dr * self._deriv(
            self.r + self.dr,
            self.y + self.k3
        )

        self.dy = (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)/6
        self.y += self.dy
        self.r += self.dr

        return self.r, self.y

### rk4 class for Tooper TOV ###
class TooperTOV:
    "RK4 class for Tooper's TOV."

    def __init__(self, y, xi, dxi, n, sigma):
        "Initializes the TOV Step class."

        self.y = y
        self.xi = xi
        self.dxi = dxi
        self.n = n
        self.sigma = sigma
        self.previous = 0

        self.k1 = np.zeros(2, dtype='float')
        self.k2 = np.zeros(2, dtype='float')
        self.k3 = np.zeros(2, dtype='float')
        self.k4 = np.zeros(2, dtype='float')

    def _deriv(self, xi, y):
        "y[0] = v, y[1] = theta"

        if y[1] < 0:
            y[1] = self.previous
        else:
            self.previous = y[1]
        self.f = np.zeros(2, dtype='float')
        self.f[0] = (xi**2)*(y[1]**self.n)
        self.f[1] = -(y[0] + self.sigma*xi*y[1]*self.f[0])*(1 + self.sigma*y[1])/(xi*(xi - 2*self.sigma*(self.n + 1)*y[0]))
        return self.f

    def rk4step(self):
        "A single step in the RK4 routine."

        # if self.y[1] < 10**(-4):
        #     self.dxi = 10**(-8)
        if self.y[1] < 10**(-2):
            self.dxi = 10**(-6)
        else:
            self.dxi = 10**(-4)

        self.k1 = self.dxi * self._deriv(
            self.xi,
            self.y
        )
        self.k2 = self.dxi * self._deriv(
            self.xi + self.dxi/2,
            self.y + self.k1/2
        )
        self.k3 = self.dxi * self._deriv(
            self.xi + self.dxi/2,
            self.y + self.k2/2
        )
        self.k4 = self.dxi * self._deriv(
            self.xi + self.dxi,
            self.y + self.k3
        )

        self.dy = (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)/6
        self.y += self.dy
        self.xi += self.dxi

        return self.xi, self.y

### rk4 class for oscillation equations ###
class OscilStep:
    "RK4 class for the system of space-time perturbation equations."

    def __init__(self, r, nu, y, n, e, p, m, sigma2, rstep, drstep):
        "Initializes the Step class for the space-time oscillations."

        self.r = r
        self.nu = nu
        self.y = y
        self.e = e
        self.p = p
        self.m = m
        self.sigma2 = sigma2
        self.rstep = rstep
        self.drstep = drstep
        self.n = n
        self.enu = []
        self.mur = 0
        self.nur = 0
                
        self.dr = []
        self.de = []
        self.dp = []
        self.dm = []
        self.dnu = []
        self.counter = 1
        self.q = 0
        self.marker = 0
        for i, x in enumerate(r):
            if i < (len(r) - 1):
                self.dr.append(self.r[i+1] - x)
                self.de.append(self.e[i+1] - self.e[i])
                self.dp.append(self.p[i+1] - self.p[i])
                self.dm.append(self.m[i+1] - self.m[i])
                self.dnu.append(self.nu[i+1] - self.nu[i])
                if self.dr[-1] == 0:
                    self.dr[-1] = self.dr[-2]
            elif i == (len(r) - 1):
                self.dr.append(self.r[i] - self.r[i-1])
                self.de.append(self.e[i] - self.e[i-1])
                self.dp.append(self.p[i] - self.p[i-1])
                self.dm.append(self.m[i] - self.m[i-1])
                self.dnu.append(self.nu[i] - self.nu[i-1])
                if self.dr[-1] == 0:
                    self.dr[-1] = self.dr[-2]

        self.k1 = np.zeros(5, dtype=float)
        self.k2 = np.zeros(5, dtype=float)
        self.k3 = np.zeros(5, dtype=float)
        self.k4 = np.zeros(5, dtype=float)

    def _deriv(self, r, y):
        "The system of derivatives for the perturbation equations."
        
        ### Locates where in the list of r the current integration is. ###
        for index in range(self.marker, len(self.r)):
            if r <= self.r[0]:
                i = 0
                break
            if r > self.r[-1]:
                i = -1
                break
            if r < self.r[index]:
                break
            if(r > self.r[index]) and (r <= self.r[index+1]):
                i = index
                self.marker = index

        self.energy = self.e[i] + (self.de[i]/self.dr[i])*(r - self.r[i])
        self.pressure = self.p[i] + (self.dp[i]/self.dr[i])*(r - self.r[i])
        self.mass = self.m[i] + (self.dm[i]/self.dr[i])*(r - self.r[i])
        nu = self.nu[i] + (self.dnu[i]/self.dr[i])*(r - self.r[i])
        
        self.q = self.de[i]/self.dp[i]
        
        self.emu = r/(r - 2*self.mass)      # exp(2mu_2)
        self.enu = np.exp(-2*nu)            # exp(-2nu)
        self.nur = (self.emu*(2*self.pressure*r**2 + 1) - 1)/(2*r)          # nu'
        self.mur = (1 - (1 - 2*self.energy*(r**2))*self.emu)/(2*r)          # mu_2'
        self.emunu = self.emu*self.enu      #e^2(mu2-nu)
        # self.q1 = self.q*(self.r[-1] - r)
        # self.q1nur = self.q1*self.nur

        self.f = np.zeros(5, dtype=float)       #0:X', 1:X'', 2:N', 3:G', 4:L'
        self.f[0] = y[1]
        self.f[1] = -(2/r + self.nur - self.mur)*self.f[0] - self.sigma2*self.emunu*y[0] - (self.n/(r**2))*self.emu*(y[2] + y[4])
        self.f[2] = (y[3]/self.nur - (self.f[0] + self.nur*(y[2] - y[4]))
            - (1/(self.nur*r**2))*(self.emu-1)*(y[2] - r*self.f[0] 
            - (r**2)*y[3]) + self.emu*(self.energy + self.pressure)*y[2]/self.nur 
            - 0.5*self.sigma2*self.emunu*(y[2] + y[4] + ((r**2)/self.n)*y[3] 
            + (1/self.n)*(r*self.f[0] + (2*self.n + 1)*y[0]))/self.nur)
        self.f[3] = (self.n*self.nur*(y[2] - y[4]) + (self.n/r)*(self.emu - 1)*(y[2] + y[4]) +
            r*(self.nur - self.mur)*self.f[0] + self.sigma2*self.emunu*r*y[0] - 2*r*y[3])/(r**2)
        self.f[4] = (-(self.f[2] + 2*self.f[0]) - (1/r - self.nur)*(-y[2] + 3*y[4] + 2*y[0]) 
            - (2/r - (self.q + 1)*self.nur)*(y[2] - y[4] + ((r**2)/self.n)*y[3] + (1/self.n)*(r*self.f[0] + y[0])))

        return self.f

    def rk4step(self):
        "RK4 routine"

        if self.rstep > (1-10**(-8))*self.r[-1]:
            self.drstep = 10**(-11)
        elif self.rstep > (1-10**(-6))*self.r[-1]:
            self.drstep = 10**(-8)
        elif self.rstep > (1-10**(-4))*self.r[-1]:
            self.drstep = 10**(-6)
        else:
            self.drstep = 10**(-4)

        self.k1 = self.drstep * self._deriv(
            self.rstep,
            self.y
        )
        self.k2 = self.drstep * self._deriv(
            self.rstep + self.drstep / 2,
            self.y + self.k1 / 2
        )
        self.k3 = self.drstep * self._deriv(
            self.rstep + self.drstep / 2,
            self.y + self.k2 / 2
        )
        self.k4 = self.drstep * self._deriv(
            self.rstep + self.drstep,
            self.y + self.k3
        )
        self.dy = (1 / 6) * (self.k1 + self.k2 * 2 + self.k3 * 2 + self.k4)
        self.rstep += self.drstep
        self.y += self.dy

        return self.rstep, self.y

    def W(self):
        "Calculates W at the current value of r."
        self.w = (self.rstep**2)*self.y[3]/self.n + (self.y[2] - self.y[4]) + (self.rstep*self.y[1] + self.y[0])/self.n

        return self.w

    def Wr(self):
        "Calculates W' at the current value of r."
        self.wr = self.f[2] - self.f[4] + (self.nur - 1/self.rstep)*self.y[2] - (self.nur + 1/self.rstep)*self.y[4]

        return self.wr

### rk4 class for Zerilli equations ###
class Zerilli:
    "RK4 class for the Zerilli equations."

    def __init__(self, y, r, dr, R, M, n, sigma, X, Xprime, L, Lprime):
        "Solves the Zerilli equation."

        self.y = y
        self.r = r #+ 2*M*log(np.abs(r/(2*M) - 1)) # self.r is r_star
        self.dr = dr
        self.R = R
        self.M = M
        self.n = n
        self.sigma = sigma
        self.X = Xprime
        self.Xprime = Xprime
        self.L = L
        self.Lprime = Lprime
        # self.drstar = 0

        self.k1 = np.zeros(2, dtype=float)
        self.k2 = np.zeros(2, dtype=float)
        self.k3 = np.zeros(2, dtype=float)
        self.k4 = np.zeros(2, dtype=float)
        self.f = np.zeros(2, dtype=float)


    def _deriv(self, y, r):

        r_star = 1 - (2*self.M/self.r)      # Conversion from r to r_*.
        self.delta = r**2 - 2*self.M*r
        self.V = (2*self.delta/(r**5*(self.n*r + 3*self.M)**2))*(
            self.n**2*(self.n + 1)*r**3 + 3*self.M*self.n**2*r**2 + 9*self.M**2*self.n*r + 9*self.M**3
        )

        #[0]: Z', [1]:Z''
        self.f[0] = self.y[1]
        self.f[1] = ((-2*self.M/self.r**2)*self.f[0]/r_star + (self.V - self.sigma**2)*self.y[0])/r_star**2      #(self.V - self.sigma**2)*self.y[0]

        return self.f

    def rk4stepZerilli(self):
        "RK4 routine"
        
        self.k1 = self.dr*self._deriv(
            self.y,
            self.r
        )
        self.k2 = self.dr*self._deriv(
            self.y + self.k1/2,
            self.r + self.dr/2
        )
        self.k3 = self.dr*self._deriv(
            self.y + self.k2/2,
            self.r + self.dr/2
        )
        self.k4 = self.dr*self._deriv(
            self.y + self.k3,
            self.r + self.dr
        )

        self.dy = (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)/6
        self.y += self.dy
        self.r += self.dr

        return self.y, self.r

### Solves Chandra's TOV for n and alpha ###
def TOV_Polytrope_Scaled(n, alpha):
    "Solves the dimensionless TOV equations. Inputs are n and alpha(central energy density over central pressure). Outputs are mass and radius in km."
    
    r0 = 10**(-4)
    dr = 10**(-4)
    theta0 = 1
    e0 = 1
    theta2 = -e0*(1 + alpha)*(1/3 + 1/alpha)/(2*(n+1))
    theta4 = e0*(theta2/(4*(n + 1)))*(4*(n + 1)/3 - (
        1/3 + 1/alpha + (alpha + 1)*(n/3 + (n+1)/alpha)
    ))
    theta = theta0 + theta2*r0**2 + theta4*r0**4
    m = (1/3)*theta**n*r0**3

    y = np.zeros(2, dtype=float)
    y[0] = theta
    y[1] = m

    star = STOVStep(r0, y, dr, n, alpha)

    while y[0] > 0:
        r, y = star.rk4step()
        if y[0] <= 0:
            break
        percent = (1 - y[0]/theta)*100
        print('%.4f %%'% percent, end='\r')
    
    return y[1], r

### Solves Chandra's TOV and returns a dictionary of lists for radius, mass, pressure, and energy density ###
def _STOVData(n, alpha):
    "Solves the dimensionless TOV equations. Inputs are n and alpha(central energy density over central pressure). Output is a dictionary containing lists of radius, mass, pressure, and energy density."
    
    r0 = 10**(-4)
    dr = 10**(-4)
    theta0 = 1
    e0 = 1
    p0 = (theta0**(n+1))/alpha
    theta2 = -e0*(1 + alpha)*(1/3 + 1/alpha)/(2*(n+1))
    theta4 = e0*(theta2/(4*(n + 1)))*(4*(n + 1)/3 - (
        1/3 + 1/alpha + (alpha + 1)*(n/3 + (n+1)/alpha)
    ))
    theta = theta0 + theta2*r0**2 + theta4*r0**4
    m = (1/3)*theta**n*r0**3

    y = np.zeros(2, dtype=float)
    y[0] = theta
    y[1] = m

    star = STOVStep(r0, y, dr, n, alpha)

    data = {}
    data['Radius'] = [r0]
    data['Mass'] = [y[1]]
    data['Pressure'] = [p0]
    data['Energy'] = [e0]
    while y[0] > 0:
        r, y = star.rk4step()
        if y[0] <= 0:
            break
        e = y[0]**n
        p = (y[0]**(n+1))/alpha
        data['Radius'].append(r)
        data['Mass'].append(y[1])
        data['Pressure'].append(p)
        data['Energy'].append(e)
        percent = (1 - y[0]/theta)*100
        print('%.4f %%'% percent, end='\r')
    
    return data

### Solves Tooper's TOV ###
def TOV_Polytrope_Real(n, alpha, e0):
    "Solves the TOV equations for a real mass and radius in solar masses and km. Inputs are n, alpha, and central energy density."

    theta = 1
    xi = 10**(-6)
    v = 0
    dxi = 10**(-4)
    sigma = 1/alpha
    msun = 1.4762

    y = np.zeros(2, dtype='float')
    y[0] = v
    y[1] = theta

    star = TooperTOV(y, xi, dxi, n, sigma)

    while y[1] > 0:
        xi, y = star.rk4step()
        percent = (1-y[1])*100
        print("%.4f %%"% percent, end='\r')

    A = np.sqrt(4*np.pi*e0/((n + 1)*sigma))
    R = xi/A
    M = sigma*(n + 1)*y[0]*R/(msun*xi)

    return M, R

### Solves for the complex GW frequencies ###
def OscilFreq(sigma_start, sigma_step=0.01, points=10, l=2, filename=None, r=None, p=None, e=None, m=None, n=None, alpha=None):
    "Solves the space-time perturbation equations. Must provide either a file or lists with radius, mass, pressure, and energy density or n and alpha for the polytropic EoS. Output are the real and imaginary frequencies. Files must contain columns separated by commas with headers identifying each column."
    
    if(filename is not None):
        ## Extract data from file
        data = {}
        try:
            with open(filename, 'r') as file:
                header = file.readline().split(',')     #column separator
                # print(header)
                for key, val in enumerate(header):
                    val = val.strip()
                    data[val] = []
                while True:
                    info = file.readline()
                    infolist = info.split(',')
                    if len(info) == 0:
                        break
                    for i, x in enumerate(header):
                        x = x.strip()
                        data[x].append(float(infolist[i]))
        except FileNotFoundError:
            print('The file you entered does not exist.')
            quit()

        print(data.keys())
        
        for val in list(data.keys()):
            if re.search(r'radius', val, re.I):
                r = data[val]
            elif re.search(r'mass', val, re.I):
                m = data[val]
            elif re.search(r'pressure', val, re.I):
                p = data[val]
            elif re.search(r'energy', val, re.I) or re.search(r'density', val, re.I):
                e = data[val]

        print('Radius:', r[-1])
        print('Mass:', m[-1])
        print('Pressure:', p[-1])
        print('Energy:', e[-1])

    elif((n is not None) and (alpha is not None)):
        ## Solve TOV then solve Oscillation
        PolyData = _STOVData(n, alpha)
        r = PolyData['Radius']
        m = PolyData['Mass']
        p = PolyData['Pressure']
        e = PolyData['Energy']

    if ((r is None) or (m is None) or (p is None) or (e is None)):
        ## Check if all data is present
        print('Must provide Radius, Mass, Pressure, and Energy Density!')
        quit()

    n = (l - 1)*(l + 2)/2
    e0 = e[0]
    p0 = p[0]
    m0 = m[0]
    r0 = r[0]

    # p2 = -(e[0] + p[0])*(3*p[0] + e[0])/3
    p2 = 2*(p[2]-p[0])/r[2]**2
    e2 = 2*(e[2]-e[0])/r[2]**2

    rmax = r[-1]

    a = p0 + e0/3
    b = 2 * e0/3
    b2 = (4/9)*e0**2 - (2/5)*e2
    b22 = (4/9) * e0**2 - (4/5) * e2
    a2 = (2/3)*e0*(p0 + e0/3) - (p2 + e2/5)
    Q = e2/p2  #9*1.5/(2.5*thetaList[0])

    dp = []
    dr = []
    for i, x in enumerate(p):
        if i < (len(p) - 1):  #== 0:
            dp.append(p[i+1] - p[i])
            dr.append(r[i+1] - r[i])
        elif i == (len(p) - 1):
            dp.append(p[i] - p[i-1])
            dr.append(r[i] - r[i-1])

    nu = 0
    nuarray = []
    enu = []

    # Integrating dp/(p+e)
    for i, x in enumerate(r):
        nu += dp[i]/(p[i] + e[i])
        nuarray.append(nu)

    # Calculating nu: exp(2nu) @ r = R is 1-(2M/R)
    nu0 = math.log(1 - 2*m[-1]/r[-1])/2 + nuarray[-1]

    # nu = integral of (dp/(p+e)) + nu_0
    for i, x in enumerate(nuarray):
        nuarray[i] = nu0 - x
    for i, x in enumerate(nuarray):
        enu.append(math.exp(-2*x))

    def _SurfOscil(sigma):
        sigma2 = sigma**2
        sigma0 = sigma2 * np.exp(-2*nu0)

        ### Solving 1st Interior Solution ###

        X0 = -1*n/(l*(l+1))
        N0 = 1
        G0 = ((l-1)/2)*(a + b - (l*(a - b) + sigma0)/(l*(l+1)))
        L0 = 0

        # # # # # # # # # # # # # # # # # # # # # # # # # #
        # AX = C => X = A^(-1)C                           #
        # A = [a11 a12 a13 a14]  X = [X2]  C = [const11]  #
        #     [a21 a22 a23 a24]      [G2]      [const21]  #
        #     [a31 a32 a33 a34]      [N2]      [const31]  #
        #     [a41 a42 a43 a44]      [L2]      [const41]  #
        # # # # # # # # # # # # # # # # # # # # # # # # # #

        a11 = (l + 2)*(l + 3)
        a12 = 0
        a13 = n
        a14 = n
        a21 = (l + 2)*(a - b) + sigma0
        a22 = -(l + 4)
        a23 = n*(a + b)
        a24 = -(a - b)*n
        a31 = (l + 2)*(a - b) + sigma0*((l + 1)**2)/(2*n)
        a32 = -1
        a33 = (l + 1)*a + sigma0/2
        a34 = sigma0/2
        a41 = (l + 3)*l*(l + 1)/n
        a42 = 0
        a43 = l + 3
        a44 = l + 3
        const11 = -l*(a - b)*X0 - n*b*(N0 + L0) - sigma0*X0
        const21 = -(a2 + b2)*n*N0 + n*(a2 - b2)*L0 - l*(a2 - b22)*X0 - sigma0*(b - a)*X0
        const31 = (-(a2*l + a**2 + b2 - b*(e0 + p0) + (e2 + p2) + (1/2)*sigma0*(b - a))*N0
            - ((1/2)*sigma0*(b - a) - a**2)*L0 - (l*(a2 - b2) + sigma0*(b - a)*(l**2 + 2*l - 1)/(2*n))*X0
            - (sigma0/(2*n) - b)*G0)
        const41 = a*(Q*N0 - (Q - 2)*L0 + (2 + (l + 1)*(Q + 1)/n)*X0) - 2*G0/n

        matA = [[a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]]
        matC = [[const11], [const21], [const31], [const41]]
        matX = np.matmul(np.linalg.inv(matA), matC)

        X2 = matX[0][0]
        G2 = matX[1][0]
        N2 = matX[2][0]
        L2 = matX[3][0]

        y = np.zeros(5, dtype=float)
        y[0] = (X0*r[0]**l + X2*r[0]**(l+2))                                #X
        y[1] = (l*X0*r[0]**(l-1) + (l+2)*X2*r[0]**(l+1))                    #X'
        y[2] = (N0*r[0]**l + N2*r[0]**(l+2))                                #N
        y[3] = (G0*r[0]**l + G2*r[0]**(l+2))                                #G
        y[4] = (L0*r[0]**l + L2*r[0]**(l+2))

        rstep = r0
        drstep = 10**(-4)
        starN = OscilStep(r, nuarray, y, n, e, p, m, sigma2, rstep, drstep)

        print('sigma =', sigma, '                     ')
        while rstep < r[-1]:
            rstep, y = starN.rk4step()
            percent = (rstep/(2*rmax))*100
            print('Internal Solution: %.2f %%' % percent, end='\r')

        WN = starN.W()
        WrN = starN.Wr()

        ### Solving 2nd Interior Solution ###

        X0 = -1*n/(l*(l+1))
        N0 = 0
        G0 = -((l-1)/2)*(a - b + (l*(a - b) + sigma0)/(l*(l+1)))
        L0 = 1

        # # # # # # # # # # # # # # # # # # # # # # # # # #
        # AX = C => X = A^(-1)C                           #
        # A = [a11 a12 a13 a14]  X = [X2]  C = [const11]  #
        #     [a21 a22 a23 a24]      [G2]      [const21]  #
        #     [a31 a32 a33 a34]      [N2]      [const31]  #
        #     [a41 a42 a43 a44]      [L2]      [const41]  #
        # # # # # # # # # # # # # # # # # # # # # # # # # #

        a11 = (l + 2)*(l + 3)
        a12 = 0
        a13 = n
        a14 = n
        a21 = (l + 2)*(a - b) + sigma0
        a22 = -(l + 4)
        a23 = n*(a + b)
        a24 = -(a - b)*n
        a31 = (l + 2)*(a - b) + sigma0*((l + 1)**2)/(2*n)
        a32 = -1
        a33 = (l + 1)*a + sigma0/2
        a34 = sigma0/2
        a41 = (l + 3)*l*(l + 1)/n
        a42 = 0
        a43 = l + 3
        a44 = l + 3
        const11 = -l*(a - b)*X0 - n*b*(N0 + L0) - sigma0*X0
        const21 = -(a2 + b2)*n*N0 + n*(a2 - b2)*L0 - l*(a2 - b22)*X0 - sigma0*(b - a)*X0
        const31 = (-(a2*l + a**2 + b2 - b*(e0 + p0) + (e2 + p2) + (1/2)*sigma0*(b - a))*N0
            - ((1/2)*sigma0*(b - a) - a**2)*L0 - (l*(a2 - b2) + sigma0*(b - a)*(l**2 + 2*l - 1)/(2*n))*X0
            - (sigma0/(2*n) - b)*G0)
        const41 = a*(Q*N0 - (Q - 2)*L0 + (2 + (l + 1)*(Q + 1)/n)*X0) - 2*G0/n

        matA = [[a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]]
        matC = [[const11], [const21], [const31], [const41]]
        matX = np.matmul(np.linalg.inv(matA), matC)

        X2 = matX[0][0]
        G2 = matX[1][0]
        N2 = matX[2][0]
        L2 = matX[3][0]

        y = np.zeros(5, dtype=float)
        y[0] = (X0*r[0]**l + X2*r[0]**(l+2))                                #X
        y[1] = (l*X0*r[0]**(l-1) + (l+2)*X2*r[0]**(l+1))                    #X'
        y[2] = (N0*r[0]**l + N2*r[0]**(l+2))                                #N
        y[3] = (G0*r[0]**l + G2*r[0]**(l+2))                                #G
        y[4] = (L0*r[0]**l + L2*r[0]**(l+2))

        rstep = r0
        drstep = 10**(-4)
        starL = OscilStep(r, nuarray, y, n, e, p, m, sigma2, rstep, drstep)

        while rstep < r[-1]:
            rstep, y = starL.rk4step()
            percent = ((rstep+rmax)/(2*rmax))*100
            print('Internal Solution: %.2f %%' % percent, end='\r')

        WL = starL.W()
        WrL = starL.Wr()

        return WN, WL, WrN, WrL, starN.y, starL.y, starN.f, starL.f

    def _ExtSol(Z, ZPrimeStar, X, Xprime, L, Lprime, sigma):
        yExt = np.zeros(2, dtype = float)
        yExt[0] = Z
        yExt[1] = ZPrimeStar
        drOuter = .01
        rOuter = r[-1]

        StarExt = Zerilli(yExt, rOuter, drOuter, r[-1], m[-1], n, sigma, X, Xprime, L, Lprime)

        rExt = [rOuter]
        ZExt = [Z]
        while rOuter < 100*r[-1]:
            yExt, rOuter = StarExt.rk4stepZerilli()
            rExt.append(rOuter)
            ZExt.append(yExt[0])
            ticker = rOuter/r[-1]
            print('External Solution: %.2f %%' % ticker, end='\r')

        def func(x, a0, b0):
            y = []
            for i in x:
                a1 = (n+1)*b0/sigma
                a2 = (n*(n+1)*a0 - 3*m[-1]*sigma*(1+2/n)*b0)/(2*sigma**2)
                b1 = (n+1)*a0/sigma
                b2 = (n*(n+1)*b0 + 3*m[-1]*sigma*(1+2/n)*a0)/(2*sigma**2)
                r_star = i + 2*m[-1]*np.log(np.abs(i/(2*m[-1]) - 1))
                # print('log:', log(np.abs(i/(2*m[-1]) - 1)))
                # print('r_star:', r_star)
                #### What is r_star!! ####
                y.append((a0 - a1/i - a2/i**2)*np.cos(sigma*r_star) - (b0 + b1/i - b2/i**2)*np.sin(sigma*r_star))

            return y

        def detweiler(x, a, b):
            rstar = x + 2*m[-1]*np.log(np.abs(x/(2*m[-1]) - 1))
            a1 = 1j*(n+1)/sigma
            a2 = (n*(n+1) + 1.5j*m[-1]*sigma*(1+2/n))/(2*sigma**2)
            b2 = (n*(n+1) - 1.5j*m[-1]*sigma*(1+2/n))/(2*sigma**2)
            zplus = np.exp(1j*sigma*rstar)*(1 + a1/x - a2/x**2)
            zminus = np.exp(-1j*sigma*rstar)*(1 - a1/x - b2/x**2)
            z = a*zminus + b*zplus
            zreal = np.real(z)
            zimag = np.imag(z)

            return np.hstack([zreal, zimag])

        params, error = curve_fit(func, rExt[int(3*len(rExt)/4):], ZExt[int(3*len(ZExt)/4):]) 
        amplitude = params[0]**2 + params[1]**2
        amplitude2 = (max(ZExt[int(3*len(ZExt)/4):]) - min(ZExt[int(3*len(ZExt)/4):]))/2
        Z_real = np.real(ZExt[int(3*len(ZExt)/4):])
        Z_imag = np.imag(ZExt[int(3*len(ZExt)/4):])
        Z_complex = np.hstack([Z_real, Z_imag])
        params2, error2 = curve_fit(detweiler, rExt[int(3*len(rExt)/4):], Z_complex)

        print('Standing Wave Amplitude =', amplitude)

        return amplitude, amplitude2**2, params2[1]

    amplitude = []
    amplitude2 = []
    det_gamma = []
    def _Solve(sigma):

        W_N, W_L, Wr_N, Wr_L, y_N, y_L, f_N, f_L = _SurfOscil(sigma)
        q = -Wr_L/Wr_N
        X = q*y_N[0] + y_L[0]
        Xprime = q*f_N[0] + f_L[0]
        L = q*y_N[4] + y_L[4]
        Lprime = q*f_N[4] + f_L[4]
        Z = (rmax/(n*rmax + 3*m[-1]))*((3*m[-1]/n)*X - rmax*L)
        Zprime = ((1/(n*rmax + 3*m[-1])**2)*((3*m[-1]/n)*(n*rmax + 3*m[-1])*rmax*Xprime + (9*m[-1]**2/n)*X
            - (n*rmax + 6*m[-1])*rmax*L - rmax**2*(n*rmax + 3*m[-1])*Lprime))
        ZPrimeStar = (1 - 2*m[-1]/rmax)*Zprime
        amp, amp2, gamma_det = _ExtSol(Z, ZPrimeStar, X, Xprime, L, Lprime, sigma)
        amplitude.append(amp)
        amplitude2.append(amp2)
        det_gamma.append(gamma_det)

        return amp, gamma_det

    sigma = sigma_start

    prevValue = 0
    amp, value = _Solve(sigma)
    prevSig = 0

    def parabola(x, a, b, c):
        return a*((x - b)**2 + c**2)

    x_data = []
    y_data = []
    while True:
        prevValue = value
        prevSig = sigma
        sigma += sigma_step
        amp, value = _Solve(sigma)
        if (np.abs(value) > np.abs(prevValue)) and value*prevValue > 0:
            sigma_step *= -1
        if value*prevValue < 0:
            guess = sigma - value*(sigma - prevSig)/(value - prevValue)
            print('\nBeginning Fine Scan.\n')
            for x in np.linspace(guess - np.abs(sigma_step)*0.2, guess + np.abs(sigma_step)*0.2, points):
                amp, value = _Solve(x)
                x_data.append(x)
                y_data.append(amp)
            break

    try:
        popt, pcov = curve_fit(parabola, x_data, y_data, p0=[2000, guess, 10**-3])
    except RuntimeError:
        print('Curve Fit Failed!')
        return None, None

    plt.title('Breit-Wigner Fit')
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_data, parabola(x_data, *popt), label="Curve Fit")
    plt.xlabel(r"Frequency, $\sigma$")
    plt.ylabel("Standing Wave Amplitude")
    plt.legend(loc='best')
    plt.savefig("Breit-Wigner Fit-Frequency=%.6f.png" % popt[1])
    plt.close()

    # amp, value = _Solve(popt[1])

    return popt[1], np.abs(popt[2])