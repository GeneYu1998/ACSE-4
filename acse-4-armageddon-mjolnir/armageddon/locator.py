"""Module dealing with postcode information."""
import math
import numpy as np
import pandas as pd


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of 
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)
    >>> with numpy.printoptions(formatter={'all', fmt}): print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    '[1.286e+05 6.378e+04]'

    """
    R_p = 6371000

    # convert latlon1, 2 to numpy.ndarrays
    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)

    distance = np.empty((len(latlon1), len(latlon2)), float)

    # turn degree to radian
    latlon1 *= math.pi / 180
    latlon2 *= math.pi / 180

    # deal with the special case that one of the input array is a single point
    if latlon1.ndim == 1:
        distance = np.empty(len(latlon2), float)
        loc_1 = latlon1

        for j in range(len(distance)):
            loc_2 = latlon2[j]
            # compute the result using Vincenty formula
            sum_1 = np.cos(loc_2[0]) * np.sin(np.abs(loc_1[1] - loc_2[1]))
            sum_2 = np.cos(loc_1[0]) * np.sin(loc_2[0]) - np.sin(loc_1[0]) * np.cos(loc_2[0]) * np.cos(
                np.abs(loc_1[1] - loc_2[1]))
            distance[j] = sum_1 ** 2 + sum_2 ** 2
            distance[j] = np.sqrt(distance[j])
            distance[j] /= np.sin(loc_1[0]) * np.sin(loc_2[0]) + np.cos(loc_1[0]) * np.cos(loc_2[0]) * np.cos(
                np.abs(loc_1[1] - loc_2[1]))
            distance[j] = np.arctan(distance[j])
            distance[j] *= R_p

    elif latlon2.ndim == 1:
        distance = np.empty(len(latlon1), float)
        loc_1 = latlon2
        latlon2 = latlon1

        for j in range(len(distance)):
            loc_2 = latlon2[j]
            sum_1 = np.cos(loc_2[0]) * np.sin(np.abs(loc_1[1] - loc_2[1]))
            sum_2 = np.cos(loc_1[0]) * np.sin(loc_2[0]) - np.sin(loc_1[0]) * np.cos(loc_2[0]) * \
                np.cos(np.abs(loc_1[1] - loc_2[1]))
            distance[j] = sum_1 ** 2 + sum_2 ** 2
            distance[j] = np.sqrt(distance[j])
            distance[j] /= np.sin(loc_1[0]) * np.sin(loc_2[0]) + np.cos(loc_1[0]) * np.cos(loc_2[0]) * \
                np.cos(np.abs(loc_1[1] - loc_2[1]))
            distance[j] = np.arctan(distance[j])
            distance[j] *= R_p

            # the general case, latlon1, 2 both lists of points
    else:
        for i in range(distance.shape[0]):
            loc_1 = latlon1[i]
            for j in range(distance.shape[1]):
                loc_2 = latlon2[j]
                sum_1 = np.cos(loc_2[0]) * np.sin(np.abs(loc_1[1] - loc_2[1]))
                sum_2 = np.cos(loc_1[0]) * np.sin(loc_2[0]) - np.sin(loc_1[0]) * np.cos(loc_2[0]) * np.cos(
                    np.abs(loc_1[1] - loc_2[1]))
                distance[i, j] = sum_1 ** 2 + sum_2 ** 2
                distance[i, j] = np.sqrt(distance[i, j])
                distance[i, j] /= np.sin(loc_1[0]) * np.sin(loc_2[0]) + np.cos(loc_1[0]) * np.cos(loc_2[0]) * np.cos(
                    np.abs(loc_1[1] - loc_2[1]))
                distance[i, j] = np.arctan(distance[i, j])
                distance[i, j] *= R_p

    return distance


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='./resources/full_postcodes.csv',
                 census_file='./resources/population_by_postcode_sector.csv',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in latitude-longitude space.

        """
        self.norm = norm
        self.postcode_file = postcode_file
        self.census_file = census_file
        self.postcodes_db = pd.read_csv(self.postcode_file)
        self.census = pd.read_csv(self.census_file)

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Return (unit or sector) postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements of radii to the location X.


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD', 'SW7 5HF', 'SW7 5HG', 'SW7 5HQ']]
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.4e3, 0.2e3], True)
        [['SW7 1', 'SW7 4', 'SW7 3', 'SW7 2', 'SW7 9', 'SW7 5'], ['SW7 1', 'SW7 4', 'SW7 3', 'SW7 2', 'SW7 9', 'SW7 5']]
        """
        location_latitude = np.array(self.postcodes_db['Latitude'])
        location_longitude = np.array(self.postcodes_db['Longitude'])
        locations = np.vstack((location_latitude, location_longitude))
        locations = locations.T
        distance = self.norm(locations, X)
        result = []

        for r in radii:

            self.postcodes_db['distance_center_less'] = distance - r
            impacts = self.postcodes_db.loc[self.postcodes_db.distance_center_less < 0]
            impact_postcodes = impacts['Postcode'].tolist()

            if sector:
                impact_postcodes = [i[:-2] for i in impact_postcodes]
                # extract the unique elements
                impact_postcodes = list(set(impact_postcodes))

            result.append(impact_postcodes)

        self.postcodes_db.drop('distance_center_less', axis=1, inplace=True)

        return result

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_population_of_postcode([['SW7 2AZ','SW7 2BT','SW7 2BU','SW7 2DD']])
        [[18.71311475409836, 18.71311475409836, 18.71311475409836, 18.71311475409836]]
        >>> locator.get_population_of_postcode([['SW7  2']], True)
        [[2283]]
        """
        self.census["geography_remove"] = self.census["geography"].apply(lambda x: x.replace(" ", ""))
        all_population = []
        if sector:

            for arr in postcodes:
                population_list = []

                for postsec in arr:

                    postsec = postsec.replace(" ", "")
                    population = self.census.loc[self.census.geography_remove == postsec]
                    if population.shape[0] == 0:
                        population_list.append(0)
                    else:
                        population_list.append(
                            population.iloc[0]['Variable: All usual residents; measures: Value'])

                all_population.append(population_list)

        else:

            self.postcodes_db["sector"] = self.postcodes_db["Postcode"].apply(lambda x: x[0:-2].replace(" ", ""))

            for arr in postcodes:

                arr = [i[:-2].replace(" ", "") for i in arr]
                population_list = [0] * len(arr)
                temp_set = set(arr)

                for key in temp_set:

                    population = self.census.loc[self.census.geography_remove == key]
                    relevant_rows = self.postcodes_db.loc[self.postcodes_db.sector == key]
                    post_unit_total = relevant_rows.shape[0]

                    for i in range(len(arr)):

                        if key == arr[i]:

                            if post_unit_total == 0:
                                population_list[i] = 0
                            else:
                                if population.shape[0] == 0:
                                    population_list[i] = 0
                                else:
                                    population_list[i] = \
                                        population.iloc[0]['Variable: All usual residents; measures: Value'] \
                                        / post_unit_total

                all_population.append(population_list)

            self.postcodes_db.drop('sector', axis=1, inplace=True)

        self.census.drop('geography_remove', axis=1, inplace=True)

        return all_population
