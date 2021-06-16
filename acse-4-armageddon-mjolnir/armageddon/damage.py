import numpy as np
import pandas as pd
from armageddon.solver import Planet
from armageddon.locator import PostcodeLocator

def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii for the input damage levels

    Examples
    --------

    >>> import armageddon
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> armageddon.damage_zones(outcome, 52.79, -2.95, 135, pressures=[1e3, 3.5e3, 27e3, 43e3])
    """
    # the radius of the earth
    R_p = 6371000

    # Convert parameters from degrees to radian
    lat = np.radians(lat)
    lon = np.radians(lon)
    beta = np.radians(bearing)

    # other parameters for use
    E_k = outcome['burst_energy']
    z_b = outcome['burst_altitude']
    r = outcome['burst_distance']

    ratio = r / R_p

    # evaluate surface zero point
    blat = np.sin(lat) * np.cos(ratio) + np.cos(lat) * np.sin(ratio) * np.cos(beta)

    blon = np.sin(beta) * np.sin(ratio) * np.cos(lat)

    blon /= np.cos(ratio) - np.sin(lat) * blat

    blon = np.arctan(blon) + lon

    blat = np.arcsin(blat)

    blat, blon = np.degrees([blat, blon])

    blat = float(blat)

    blon = float(blon)

    if isinstance(pressures, float):
        pressures = [pressures]

    # a list(array) to store the radius
    damrad = [0] * len(pressures)
    damrad = np.array(damrad)


    # define the non-linear equation, as well as its 1st and 2nd derivatives
    non_linear_eq_deri = lambda x: 3.14e11 * (-1.3) * x**(-2.3) + 1.8e7 * (-0.565) * x **(-1.565)
    non_linear_eq_2nd_deri = lambda x: 3.14e11 * (-1.3)*(-2.3) * x**(-3.3) + 1.8e7 * (-0.565)*(-1.565) * x **(-2.565)

    # use 2nd order Newton-Raphson method to solve damage radius from the Empirical formula
    for i, A in enumerate(pressures):
        non_linear_eq = lambda x: 3.14e11* x**(-1.3) + 1.8e7 * x**(-0.565) - A
        root = second_order_Newton(non_linear_eq, non_linear_eq_deri, non_linear_eq_2nd_deri, A, A)
        damrad[i] = root

    damrad = damrad * E_k**(2/3) - z_b**2

    damrad[damrad < 0] = 0

    damrad = np.sqrt(damrad)

    damrad = damrad.tolist()


    return blat, blon, damrad


def second_order_Newton(f, dfdx, df2dx2, initial_guess, scaling = 1, max_iterstep = 1e6):
    """
    Solving a nolinear equation f(x) = 0 using the 2nd order Newton-Raphson method,
    with iteration function: phi(x) = x + f(x)*[u(x) + f(x)*w(x)],
    we could show by theoretical analysis that u(x) = - f(x)/f'(x) and
    w(x) = - 2f''(x)/ [f'(x)]^3 enables a cubic convergence for this method.
    but to note choosing the initial guess is tricky for this method.

    Parameters
    ----------

    f: function
        the non-linear function we need to solve, (eg. f(x) = 0)
    dfdx: function
        the 1st derivative for function f
    df2dx2: function
        the 2nd derivative for function f
    intial_guess: float
        the start point of the iteration
    scaling (Optional): float
        a scaling factor to ensure convergence, depending on the specific function we have.
        the point here to make the absolute value of f, dfdx, df2dx2 at the fix point
        C to be small, (eg. |f(C)|, |f'(C)|, |f''(C| is not too big)
    max_iterstep: int
        the max iterations that you can accept for this solver, by default it's 10^6.

    Returns
    -------

    root: float
        the numerical solution for the Newton method

        """

    itermax = max_iterstep
    x_0 = initial_guess / scaling
    this_x = x_0
    gap = 1
    count = 0

    while (gap > 1e-6 and count < itermax):
        f_x = f(this_x) / scaling
        f_x_1st = dfdx(this_x) / scaling
        f_x_2nd = df2dx2(this_x) / scaling
        next_x = this_x - (f_x / f_x_1st) - (1/2) * (f_x / f_x_1st )**2 * (f_x_2nd / f_x_1st )
        gap = next_x - this_x
        this_x = next_x
        count = count + 1

    if count == itermax:
        print('Newton does not convergent in limited iterations.')

    return next_x

fiducial_means = {'radius': 10, 'angle': 20, 'strength': 1e6,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 51.5, 'lon': 1.5, 'bearing': -45.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e5,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def impact_risk(planet, means=fiducial_means, stdevs=fiducial_stdevs,
                pressure=27.e3, nsamples=100, sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector

    Parameters
    ----------
    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry

    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    pressure: float
        The pressure at which to calculate the damage zone for each impact

    nsamples: int
        The number of iterations to perform in the uncertainty analysis

    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes

    Returns
    -------
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    """

    input_planets = []
    input_damage_zones = []

    gen_rand_nums = lambda means_dict, sdevs_dict, nsamples, klist : \
    [np.random.normal(means_dict[key], sdevs_dict[key], nsamples) for key in klist]

    list1 = ['radius', 'angle', 'strength', 'density', 'velocity']

    list2 = ['lat', 'lon', 'bearing']

    rand_lists_planet = gen_rand_nums(means, stdevs, nsamples, list1)

    rand_lists_zones = gen_rand_nums(means, stdevs, nsamples, list2)

    rand_lists_planet = np.array(rand_lists_planet).T

    rand_lists_planet = rand_lists_planet.tolist()

    rand_lists_zones = np.array(rand_lists_zones).T

    rand_lists_zones = rand_lists_zones.tolist()

    for i in range(nsamples):
        input_planets.append(dict(list(zip(list1, rand_lists_planet[i]))))
        input_damage_zones.append(dict(list(zip(list2, rand_lists_zones[i]))))

    surface_zeros = [0] * nsamples
    radiis = [0] * nsamples

    for i in range(nsamples):
        input_solver = input_planets[i]
        input_damage_zone = input_damage_zones[i]

        result = planet.solve_atmospheric_entry(radius=input_solver['radius'],
                                       angle=input_solver['angle'],
                                       strength=input_solver['strength'],
                                       density=input_solver['density'],
                                       velocity=input_solver['velocity'])

        result = planet.calculate_energy(result)

        outcome = planet.analyse_outcome(result)

        blast_lat, blast_lon, damage_rad = damage_zones(outcome,
                                                           lat=input_damage_zone['lat'],
                                                           lon=input_damage_zone['lon'],
                                                           bearing=input_damage_zone['bearing'],
                                                           pressures=pressure)

        surface_zeros[i] = (blast_lat, blast_lon)

        radiis[i] = damage_rad

    print(surface_zeros, radiis)

    locator = PostcodeLocator()

    columns = ["sector", "population", "risk"]
    df = pd.DataFrame(columns=columns)
    total = {}

    for index, point in enumerate(surface_zeros):

        r = [radiis[index]]

        if not sector: #fix here
            lists_of_sectors = locator.get_postcodes_by_radius(point, r)
            lists_of_population = locator.get_population_of_postcode(lists_of_sectors)
        else:
            lists_of_sectors = locator.get_postcodes_by_radius(point, r, True)
            lists_of_population = locator.get_population_of_postcode(lists_of_sectors, True)

        sectors = lists_of_sectors[0]
        populations = lists_of_population[0]

        for i, sec in enumerate(sectors):

            population_num = populations[i]
            row = pd.Series({'sector': sec, 'population': population_num, 'risk': 0})
            df = df.append(row, ignore_index=True)
            if sec in total:
                total[sec] += 1
            else:
                total[sec] = 1

    df.drop_duplicates(subset='sector', inplace=True)

    for key in total.keys():
        df['risk'][df.sector == key] = (total[key] / nsamples) * \
                                       df['population'][df.sector == key]

    result = df[['sector', 'risk']]

    return result
