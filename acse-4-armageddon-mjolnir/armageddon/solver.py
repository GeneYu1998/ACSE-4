import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'
        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option
        Cd : float, optional
            The drag coefficient
        Ch : float, optional
            The heat transfer coefficient
        Q : float, optional
            The heat of ablation (J/kg)
        Cl : float, optional
            Lift coefficient
        alpha : float, optional
            Dispersion coefficient
        Rp : float, optional
            Planet radius (m)
        rho0 : float, optional
            Air density at zero altitude (kg/m^3)
        g : float, optional
            Surface gravity (m/s^2)
        H : float, optional
            Atmospheric scale height (m)
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0

        inverse_file = pd.read_csv('./data/ChelyabinskEnergyAltitude.csv', skiprows=1, header=None, sep=',')
        self.inverse_data = inverse_file.to_numpy()
        inverse_data_sorted = self.inverse_data[self.inverse_data[:,0].argsort()]
        self.inverse_fcn = interpolate.UnivariateSpline(inverse_data_sorted[:, 0], inverse_data_sorted[:, 1], k=1, ext=3)

        # set function to define atmospheric density
        if atmos_func == 'exponential':
            self.rhoa = lambda z: self.rho0*np.exp(-z/self.H)
        elif atmos_func == 'tabular':
            self.df = pd.read_csv('./data/AltitudeDensityTable.csv',
                                  skiprows=6, header=None, sep=' ').to_numpy()
            def rhoa(z):
                z = np.array(z)
                i = np.where((z/10.).astype(np.int)<8600, (z/10.).astype(np.int), 8600)
                z_i = self.df[i, 0]
                rho_i = self.df[i, 1]
                H_i = self.df[i, 2]
                return rho_i*np.exp((z_i-z)/H_i)
            self.rhoa = rhoa
        elif atmos_func == 'constant':
            self.rhoa = lambda x: rho0
        else:
            raise NotImplementedError(
                "atmos_func must be 'exponential', 'tabular' or 'constant'")

    def determine_parameters(self, rho0=3300, theta0=18.3, v0=19.2e3, radians=False):
        """
        Determine the strength (Y) and radius (r), given initial data rho0, theta0, v0,
        and observed energy deposition curve.

        Parameters
        ----------
        rho0 : float, optional
            The density.
        theta0 : float, optional
            The impact angle.
        v0 : float, optional,
            The entry velocity
        radians : bool, optional
            Whether theta0 is in radians or not

        Returns
        -------
        Y : float
            Predicted strength
        r : float
            Predicted radius
        """
        import matplotlib.pyplot as plt

        if not radians:
            theta0 = np.pi*(theta0/180)
   
        def get_loss(x):
            """
            Parameters
            ----------
            x : np.array
                '1 x 2' np.array

            Returns
            -------
            answer : float
            """
            Y, r = x[0], x[1]
            result = self.calculate_energy(self.solve_atmospheric_entry(radius=r, velocity=v0, density=rho0, strength=Y, angle=theta0))
            z = result['altitude'].to_numpy()
            energy = result['dedz'].to_numpy()
            # plt.scatter(z * 1e-3, energy,color='r')
            # plt.scatter(z * 1e-3, self.inverse_fcn(z),color='b')
            # plt.show()
            answer = np.mean((self.inverse_fcn(z)-energy)**2)
            return answer
        x0 = np.array([1e6, 10])
        get_loss(x0)
        res = minimize(get_loss, x0, method='Nelder-Mead', tol=1, options={'maxiter': 100, 'disp': True})
        Y, r = res.x[0], res.x[1]
        return Y, r

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters
        velocity : float
            The entery speed of the asteroid in meters/second
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians
        init_altitude : float, optional
            Initial altitude in m
        dt : float, optional
            The output timestep, in s
        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """

        # Enter your code here to solve the differential equations
        if not radians:
            angle = np.pi*(angle/180)

        # initialize vector
        y0 = np.zeros(6)
        y0[0] = velocity
        y0[1] = (4/3)*np.pi*(radius**3)*density
        y0[2] = angle
        y0[3] = init_altitude
        y0[4] = 0.
        y0[5] = radius

        # solving through time stepping
        y, t = self.RK45(self.f, y0, 0, 800, dt, strength, density)

        if not radians:
            y[:, 2] = 180*(y[:, 2]/np.pi)

        return pd.DataFrame({'velocity': y[:, 0],
                             'mass': y[:, 1],
                             'angle': y[:, 2],
                             'altitude': y[:, 3],
                             'distance': y[:, 4],
                             'radius': y[:, 5],
                             'time': t}, index=range(len(t)))

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude

        Returns
        -------
        result : DataFrame
        """

        result = result.copy()
        result = result.to_numpy()
        # kinetic energy = 0.5*m*v^2
        dedz = np.array((0.5*result[1:, 1]*(result[1:, 0]**2) -
                         0.5*result[:-1, 1]*(result[:-1, 0]**2)
                         )/(result[1:, 3] - result[:-1, 3]))
        dedz = np.insert(dedz, 0, 0)
        result = pd.DataFrame({'velocity': result[:, 0],
                               'mass': result[:, 1],
                               'angle': result[:, 2],
                               'altitude': result[:, 3],
                               'distance': result[:, 4],
                               'radius': result[:, 5],
                               'time': result[:, 6],
                               'dedz': dedz / 4.184e9})
        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats

        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
 
        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key ``outcome`` (which should contain one of the following
                                 strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``),
                as well as the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_distance``,
            ``burst_energy``
        """
        max_dedz = result.dedz.max()
        index_dedz = result.dedz.idxmax()
        burst_altitude = result.altitude[index_dedz]
        burst_distance = result.distance[index_dedz]
        E0 = result['mass'][0]*result['velocity'][0]**2
        E1 = result['mass'][index_dedz]*result['velocity'][index_dedz]**2
        burst_total_ke_lost = 1/2*(E0 - E1) / 4.184e12

        outcome = {}
        if burst_altitude > 5000:
             #Airbust above 5km
            outcome["outcome"] = "Airburst"
            outcome["burst_energy"] = burst_total_ke_lost

        else:
            #Calculating residual kinetic energy at peak
            mass = result.mass[index_dedz]
            velocity = result.velocity[index_dedz]
            residual_KE = 0.5 * mass * velocity ** 2 / 4.184e9
            outcome["burst_energy"] = max(residual_KE, burst_total_ke_lost)

            if (burst_altitude < 5000 and burst_altitude > 0):
                #Low-altitude (below 5km) airburst
                outcome["outcome"] = "Airburst and cratering"

            else:
                #Cratering event, i.e. "bursts" on the ground
                outcome["outcome"] = "Cratering"
        
        outcome["burst_peak_dedz"] = max_dedz
        outcome["burst_altitude"] = burst_altitude
        outcome["burst_distance"] = burst_distance
        return outcome

    # implement RK4 algorithm
    def RK4(self, f, y0, t0, t_max, dt, strength, density):
        """
        Solves ODE using explicit 4-stage, 4th order Runge-Kutta method

        Parameters
        ----------
        f : function
            Returns derivative
        y0 : np.array
            '1 x j' array
            Initial vector of j parameters. In this script j=6, namely
            velocity, mass, angle, altitude, distance, and radius
        t0 : float
            Initial time for time-stepping
        t_max : float
            Final time for time-stepping
        dt : float
            Time interval for time-stepping
        strength : float
            The strength of the asteroid
        density : float
            The density of the asteroid

        Returns
        -------
        y : np.array
            'n x j' array of solution over full time frame
        t : np.array
            '1 x n' array of times matching array y, up to t_max
        """
        n = int((t_max-t0)/dt) + 1
        y = np.zeros((n, len(y0)))
        y[0, :] = y0
        t = np.zeros(n)
        t[0] = t0
        for i in range(n-1):
            k1 = dt*f(t[i], y[i, :], strength, density)
            k2 = dt*f(t[i] + 0.5*dt, y[i, :] + 0.5*k1, strength, density)
            k3 = dt*f(t[i] + 0.5*dt, y[i, :] + 0.5*k2, strength, density)
            k4 = dt*f(t[i] + dt, y[i, :] + k3, strength, density)
            y[i+1, :] = y[i, :] + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)
            t[i+1] = t[i] + dt
        return y, t

    def f(self, t, y, strength, density):
        """
        Calculates derivative of input vector y, as given in equations of
        motion found in AirburstSolver.ipynb

        Parameters
        ----------
        t : float
            Current time
        y : np.array
            '1 x 6' array containing current velocity, mass, angle, altitude,
            distance, and radius
        strength : float
            Strength of the asteroid
        density : float
            Density of the asteroid

        Returns
        -------
        out : np.array
            '1 x 6' array of current derivatives of velocity, mass, angle,
            altitude, distance, and radius
        """
        # initialise derivative vector
        out = np.zeros(6)

        # area of asteroid
        A = np.pi*(y[5]**2)

        ra = self.rhoa(y[3])

        # velocity derivative
        out[0] = -self.Cd*ra*A*(y[0]**2)/(2*y[1]) + self.g*np.sin(y[2])

        # mass derivative
        out[1] = -self.Ch*ra*A*(y[0]**3)/(2*self.Q)

        # angle derivative
        out[2] = (self.g*np.cos(y[2])/y[0] -
                  self.Cl*ra*A*y[0]/(2*y[1]) -
                  y[0]*np.cos(y[2])/(self.Rp + y[3]))

        # altitude derivative
        out[3] = -y[0]*np.sin(y[2])

        # distance derivative
        out[4] = y[0]*np.cos(y[2])/(1+y[3]/self.Rp)

        # radius derivative
        if ra*(y[0]**2) >= strength:
            out[5] = y[0]*((3.5*self.alpha*ra/density)**0.5)
        else:
            out[5] = 0

        return out

    def RK45(self, f, y0, t0, t_max, dt, strength, density, tol=1.e-7,
             error_out=False):
        """
        Solves ODE using explicit 7-stage, 5th order Runge-Kutta method,
        aka Dormand-Prince method, with adaptive time-stepping
        Parameters
        ----------
        f : function
            Returns derivative
        y0 : np.array
            '1 x 6' array
            Initial vector of velocity, mass, angle, altitude, distance, and
            radius
        t0 : float
            Initial time for time-stepping
        t_max : float
            Final time for time-stepping
        dt : float
            Time interval for time-stepping
        strength : float
            The strength of the asteroid
        density : float
            The density of the asteroid
        tol : float, optional
            Tolerance for error of solution
        error_out : bool, optional
            Whether error wants to be part of the output
        Returns
        -------
        y_out : np.array
            'n x 6' array of velocities, masses, angles, altitudes, distances,
            and radii over full time frame
        t_out : float
            '1 x n' array of times through time-stepping, up to t_max
        e_out : array, conditional
            '1 x n' array of corresponding errors. Only part of output
            if error_out is True
        """
        # specify constants
        a21 = 1. / 5.

        a31 = 3. / 40.
        a32 = 9. / 40.

        a41 = 44. / 45.
        a42 = -56. / 15.
        a43 = 32. / 9.

        a51 = 19372. / 6561.
        a52 = -25360. / 2187.
        a53 = 64448. / 6561.
        a54 = -212. / 729.

        a61 = 9017. / 3168.
        a62 = -355. / 33.
        a63 = 46732. / 5247.0
        a64 = 49. / 176.
        a65 = -5103. / 18656.

        a71 = 35. / 384.
        a72 = 0.
        a73 = 500. / 1113.
        a74 = 125. / 192.
        a75 = -2187. / 6784.
        a76 = 11. / 84.

        c2 = 1. / 5.
        c3 = 3. / 10.
        c4 = 4. / 5.
        c5 = 8. / 9.
        c6 = 1.
        c7 = 1.

        b1a = 35. / 384.
        b2a = 0.
        b3a = 500. / 1113.
        b4a = 125. / 192.
        b5a = -2187. / 6784.0
        b6a = 11. / 84.0
        b7a = 0.0

        b1b = 5179. / 57600.0
        b2b = 0.0
        b3b = 7571. / 16695.0
        b4b = 393. / 640.0
        b5b = -92097. / 339200.
        b6b = 187. / 2100.
        b7b = 1. / 40.

        # specify output variables
        y_out = np.array(y0)
        t_out = [t0]
        e_out = [0]
        dt_out = dt

        # specify y and t for iterations
        y = y0
        t = t0
        i = 1
        y_prev = y0
        t_prev = t0
        e_prev = 0

        # iterate over time steps
        while t < t_max and y[3] >= 0:
            k1 = dt*f(t, y, strength, density)
            k2 = dt*f(t + c2*dt, y + a21*k1, strength, density)
            k3 = dt*f(t + c3*dt, y + a31*k1 + a32*k2, strength, density)
            k4 = dt*f(t + c4*dt, y + a41*k1 + a42*k2 + a43*k3,
                      strength, density)
            k5 = dt*f(t + c5*dt, y + a51*k1 + a52*k2 + a53*k3 + a54*k4,
                      strength, density)
            k6 = dt*f(t + c6*dt, y + a61*k1 + a62*k2 + a63*k3 + a64*k4 +
                      a65*k5,
                      strength, density)
            k7 = dt*f(t + c7*dt, y + a71*k1 + a72*k2 + a73*k3 + a74*k4 +
                      a75*k5 + a76*k6,
                      strength, density)
            y5 = y + b1a*k1 + b2a*k2 + b3a*k3 + b4a*k4 + b5a*k5 + b6a*k6 \
                + b7a*k7
            y4 = y + b1b*k1 + b2b*k2 + b3b*k3 + b4b*k4 + b5b*k5 + b6b*k6 \
                + b7b*k7
            # error control
            e = np.linalg.norm(y5 - y4)/np.sqrt(y.size)
            if e <= tol:
                if e > 1.89e-4:
                    dt_next = 0.9 * (tol/e)**(1./5.) * dt
                else:
                    dt_next = 5. * dt
            else:
                dt_next = 0.9 * (tol/e)**(1./4.) * dt
                if dt >= 0:
                    dt_next = max(dt_next, 0.1*dt)
                else:
                    dt_next = min(dt_next, 0.1*dt)
            if e < tol:
                y = y5
                t = t + dt
                if t >= t0 + i*dt_out:
                    tx = t0 + i*dt_out
                    n_intervals = int((t-tx)/float(dt_out))
                    for j in range(n_intervals+1):
                        new_y = y_prev + (y - y_prev)*(tx - t_prev)/dt
                        y_out = np.vstack((y_out, new_y))
                        t_out.append(tx)
                        e_out.append(e_prev + (e - e_prev)*(tx - t_prev)/dt)
                        # stop if altitude falls below 0,
                        # or velocity becomes 0,
                        # or mass becomes 0
                        if new_y[3] < 0 or new_y[0] <= 0 or new_y[1] <= 0:
                            break
                        i += 1
                        tx = t0 + i*dt_out
                y_prev = y
                t_prev = t
                e_prev = e
            dt = dt_next
        if not error_out:
            return y_out, t_out
        else:
            return y_out, t_out, e_out

    def error_convergence(self, radius, velocity, density, strength, angle,
                          tol_list, init_altitude=100e3, dt=0.05,
                          radians=False):
        """
        Calculates errors depending on the tolerance limit

        Parameters
        ----------
        tol_list : array
            '1 x n' array of n tolerances to test for solver

        Returns
        -------
        err_list : array
            '1 x n' array of n errors corresponding to n tolerances
        """
        if not radians:
            angle = np.pi*(angle/180)

        # initialize vector
        y0 = np.zeros(6)
        y0[0] = velocity
        y0[1] = (4/3)*np.pi*(radius**3)*density
        y0[2] = angle
        y0[3] = init_altitude
        y0[4] = 0.
        y0[5] = radius

        err_list = []
        # iterate through tol_list
        for tol in tol_list:
            print("Currently testing tolerance = " + str(tol))
            y, t, e = self.RK45(self.f, y0, 0, 50, dt, strength, density, tol,
                                error_out=True)
            err_list.append(np.linalg.norm(e)/np.sqrt(len(e)))

        return err_list