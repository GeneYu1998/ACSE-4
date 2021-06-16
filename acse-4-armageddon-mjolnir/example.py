import armageddon

#######################
### Airburst Solver ###
#######################

# Initialise the Planet class
earth = armageddon.Planet(atmos_func='constant')

# Solve the atmospheric entry problem (for something similar to Chelyabinsk). 
result = earth.solve_atmospheric_entry(radius=10, angle=20,
                                       strength=1e6, density=3000,
                                       velocity=19e3)

# Calculate the kinetic energy lost per unit altitude and add it
# as a column to the result dataframe
result = earth.calculate_energy(result)

# Determine the outcomes of the impact event
outcome = earth.analyse_outcome(result)

#####################
### Damage Mapper ###
#####################

# Calculate the blast location and damage radius for several pressure levels
blast_lat, blast_lon, damage_rad = armageddon.damage_zones(outcome,
                                                           lat=51.2, lon=0.7, bearing=-35.,
                                                           pressures=[1e3, 3.5e3, 27e3, 43e3])


# Plot a circle to show the limit of the lowest damage level
damage_map = armageddon.plot_circle(blast_lat, blast_lon, damage_rad)
damage_map = armageddon.plot_polyline(lat=51.2, lon=0.7,blat=blast_lat,blon=blast_lon,map=damage_map)
damage_map.save("damage_map.html")

# The PostcodeLocator tool
locator = armageddon.PostcodeLocator()

# Find the postcodes in the damage radii
postcodes = locator.get_postcodes_by_radius((blast_lat, blast_lon), radii=damage_rad)

# Find the population in each postcode
population = locator.get_population_of_postcode(postcodes)

# Alternatively find the postcode sectors in the damage radii, and populations of the sectors
sectors = locator.get_postcodes_by_radius((blast_lat, blast_lon), radii=damage_rad, sector=True)
population_sector = locator.get_population_of_postcode(sectors, sector=True)

