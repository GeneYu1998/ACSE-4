import folium
import random

from armageddon import locator


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    #>>> import folium
    #>>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)
        map.add_child(folium.LatLngPopup())

    for i in radius:
        folium.Circle([lat, lon], i, color=randomcolor(), fill=True, fillOpacity=0.6, **kwargs).add_to(map)

    # folium.Circle([lat, lon], radius, fill=True, fillOpacity=0.6, **kwargs).add_to(map)

    return map


def randomcolor():
    """
    Randomly select a color for plotting

    Returns
    -------
    A random color code

    """
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def plot_polyline(lat, lon, blat, blon, map=None):
    """
    Plot a line between the meteoroid entry point and the surface zero location
    on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)

    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)

    map: folium.Map
        existing map object

    Returns
    -------
    Folium map object

    """
    if not map:
        map = folium.Map(location=[blat, blon], control_scale=True)
        map.add_child(folium.LatLngPopup())

    folium.PolyLine(
        locations=[
            [lat, lon],
            [blat, blon],
        ],
        popup=f'great circle distance: {locator.great_circle_distance([lat, lon], [[blat, blon]])[0]}',
        icon=folium.Icon(color='red'),
        color='black'
    ).add_to(map)

    folium.Marker(
        location=[lat, lon],
        popup=f'enter point:{[lat, lon]}',
        icon=folium.Icon(color='red'),
    ).add_to(map)

    folium.Marker(
        location=[blat, blon],
        popup=f'centre point:{[blat, blon]}',
        icon=folium.Icon(color='blue'),
    ).add_to(map)

    return map
