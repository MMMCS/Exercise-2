import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def analyse_molecular_geometry(path_to_outfiles):
    """ This code takes the path of the directory containing either the H2O or H2S outfiles (not both) as an input
and outputs 1) a potential energy surface and 2) frequencies for the bend and symmetric stretch normal modes.
User input of path can be done by pasting path and replacing backslashes by forward slashes.
This programme requires os, numpy and matplotlib packages to be installed.
The programme outputs a text file with the normal mode frequencies and an image file of the energy surface."""

    # Set user input as current working directory
    os.chdir(path_to_outfiles)
    currentdirectory = os.getcwd()
    directory = os.fsencode(currentdirectory)

    # Create list of files within working directory
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        files.append(filename)

    # Parse files for necessary data where theta is bond angle, r is bond length, energies is energy of geometry
    temporary_list, energies, r, theta = [], [], [], []
    for file in files:
        if "theta" in file: # Opens file if theta is in filename
            opened_data_file = open(str(file), "r")
            for line in opened_data_file:
                if 'Input=' in line:
                    characters_in_line = list(line) # List characters of the whole line
                    temporary_list = [''.join(characters_in_line[12:16])] # Extract the bond length data, r
                    r.append(float(temporary_list[0])) # Append to list of bond lengths
                    temporary_list = [''.join( characters_in_line[21:25])] # Extract the bond angle data, theta
                    theta.append(float(temporary_list[0] + "0")) # Append to list of bond angles
                if 'E(RHF)' in line:
                    words_in_line = line.split() # List 'words' of the whole line
                    energies.append(float(words_in_line[4])) # Append to list of energies

    # If data set is for H2S, trim the lists made above of the extremes of data to give a clearer surface plot
    if 'H2S' in filename:
        energies, r, theta = energies[637:], r[637:], theta[637:]

    # Parsed data lists are currently in order from theta 100 upwards first and then up to 100
    # reorder the data by increasing theta to give clearer surface plot
    theta, r, energies = zip(*sorted(zip(theta, r, energies))) # Unzip, sort by theta, rezip

    # Generate arrays necessary for the surface plot
    # Generate lists spanning ascending unique values in r and theta
    r_unique_values, theta_unique_values = [], []
    for r_list_member in r: # If a member of r is not yet in list of unique r values, add it
        if r_list_member not in r_unique_values:
            r_unique_values.append(r_list_member )
    for theta_list_member in theta:
        if theta_list_member not in theta_unique_values:
            theta_unique_values.append(theta_list_member)
    X, Y = np.meshgrid(r_unique_values, theta_unique_values) # Create X and Y axes with correct values
    Z = np.zeros((len(theta_unique_values),(len(r_unique_values)))) # Initialise empty Z 2D array with dimensionality of X and Y
    for row_number in range(0,len(theta_unique_values)):
        Z[row_number] = energies[(len(r_unique_values)* row_number):(len(r_unique_values) * (row_number + 1))] # Populate array with the energy corresponding to (r, theta)

    # Plot energy surface and adjust visuals
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    energy_surface = ax.plot_surface(X, Y, Z, rstride=6, cstride=1, cmap='jet', linewidth=0.5, edgecolor="black", alpha=0.85) # Plot surface with colour scheme to aid visualisation
    plt.title('Energy Surface for ' + filename[0:3] + '\n \n') # Plot title
    ax.set_xlabel('\n r / Angstroms') # Axes labels
    ax.set_ylabel('\n Theta / degrees')
    for tick in ax.zaxis.get_majorticklabels(): # Stop the negative sign in front of energies overlapping with z ticks
        tick.set_horizontalalignment("left")
    if 'H2S' in filename: # Set H2S default viewing angle
        ax.view_init(elev=40, azim=-61)
        ax.set_zlabel('\n \n \n \n \n \n Energy / Hartree') # Hack z label away from z tick labels for neatness
        pad_amount = 0.08
    else: # Set H2O default viewing angle
        ax.view_init(elev=40, azim=-73)
        ax.set_zlabel('\n \n \n \n Energy / Hartree') # Hack z label away from z tick labels for neatness
        pad_amount = 0.02
    fig.colorbar(energy_surface, shrink=0.5, aspect=5, pad=pad_amount) # Move colour bar slightly to see z label if H2S for neatness

    # Find minimum point on energy surface
    surface_points = []
    for i, j, k in zip(energies, r, theta): # Generate list of (z, x, y) coordinates
        surface_points.append((i, j, k))
    surface_min_point = min(surface_points) # Find coordinates for minimum energy point (minimal z coordinate) on surface
    E0, r0, theta0 = (surface_min_point[0]), round(surface_min_point[1],2), surface_min_point[2]

    # Calling constants
    pi = 3.141592654
    u = 1.66053904020e-27 # amu
    c = 2.99792458e10 # Speed of light
    deg_to_rad = pi / 180 # Degrees to radians conversion factor
    hartee_to_joule = 4.35974e-18 # Hartree to joules conversion factor
    ang_to_metres = 1e-10 # Angstrom to metres conversion factor

    # Finding normal mode frequencies by curve fitting close to energy minimum along r and theta axes
    energy_at_fixed_r, theta_at_fixed_r, energy_at_fixed_theta, r_at_fixed_theta = [], [], [], []
    for energy_list_member, r_list_member, theta_list_member in zip(energies, r, theta):
        if r_list_member == r0: # At r0, consider data along theta axis
            if 0.95 * theta0 < theta_list_member < 1.05 * theta0: # Limit theta to maximise validity of harmonic approximation
                energy_at_fixed_r.append((energy_list_member - E0) * hartee_to_joule) # adds energy - energy0 to list
                theta_at_fixed_r.append((theta_list_member - theta0) * deg_to_rad) # adds theta - theta0 to list
        if theta_list_member == theta0: # At theta0, consider data along r axis
            if 0.8 * r0 < r_list_member < 1.2 * r0: # Limit r to maximise validity of harmonic approximation
                energy_at_fixed_theta.append((energy_list_member - E0) * hartee_to_joule) # adds energy - energy0 to list
                r_at_fixed_theta.append((r_list_member - r0) * ang_to_metres) # adds r - r0 to list

    # Fit data generated above to quadratics where coefficients of second order terms are spring constants
    k_for_symm = np.polyfit(r_at_fixed_theta, energy_at_fixed_theta, 2)
    k_for_bend = np.polyfit(theta_at_fixed_r, energy_at_fixed_r, 2)

    SHO_symm_freq = ((1 / (2 * pi)) * ((2 * k_for_symm[0] / ((2 * u)))) ** 0.5) / c
    SHO_bend_freq = ((1 / (2 * pi)) * (((2 * k_for_bend[0] / (0.5 * u * (r0 * ang_to_metres) ** 2))) ** 0.5)) / c

    # Print frequencies of normal modes and potential energy surface to output files
    if 'H2S' in filename:  # name output files as appropriate to molecule analysed
        data_output_file = "H2Soutput.txt"
        plot_output_file = "H2Splot.png"
    if 'H2O' in filename:
        data_output_file = "H2Ooutput.txt"
        plot_output_file = "H2Oplot.png"

    plt.savefig(plot_output_file, format='png', dpi=1700)  # save plot to file

    print("", filename[0:3], "Symmetric Stretch via fitted curve:", round(SHO_symm_freq, 2), "cm-1", "\n",
          filename[0:3], "Bend via fitted curve:", round(SHO_bend_freq, 2), "cm-1", "\n",
          file=open(data_output_file, "w"))

analyse_molecular_geometry("C:/Users/User/Documents/Cambridge University Work/Part II/Practical/Exercise2/venv/H2Soutfiles")