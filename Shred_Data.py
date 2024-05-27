"""
Project SHRED 2024
Author: (c) Seth Gerow, Embry-Riddle Aeronautical University
email: gerows@my.erau.edu
This is the main data analysis module for Project SHRED. Functions used within this module are for importing data from the wind tunnel tests, performing geometric and data analysis, and storing data and metadata in "Fin" objects.
"""
import pandas as pd
import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

class Fin():
    """
    The SHRED Fin Object stores wind tunnel data, STL data of the fin, plotting metadata (axis titles, date of test, units, ect.).
    Future versions of SHRED will expand the Fin Object to store geometric data such as NACA classifications, and custom SHRED 
    classifications such as profile shape and performance metrics.
    Future Versions of SHRED will hopefully also include methods to convert between unit types, [USC/SI], and store multiple sets of data
    If you are interested in contributing these would be good places to start.
    """
    def __init__(self, name, data_file, STL_file):
        self.data = {'Wind Tunnel' : import_new_data(data_file)[0],
                     'STL': import_new_stl(STL_file)
                    }
        self.meta = {'Name' : name,
                     'Data File' : data_file,
                     'Date of Test': import_new_data(data_file)[1],
                     'Units Type': import_new_data(data_file)[2],
                     'NACA Descriptor': 'Yet to be implemented',
                     'Axis Titles': 'Yet to be implemented',
                     'Profile Curve': 'Yet to be implemented',
                    }
    def show_stl(self):
        figure = plt.figure()
        print(f"defining axes...")
        axes = figure.add_subplot(projection='3d')
        print("done")
        print(f"loading {self.meta['Name']}...")
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.data['STL'].vectors))
        print("done")
        print(f"scaling axes...")
        scale = self.data['STL'].points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)
        print("done")
        plt.show()
        

def get_date(df): 
    """
    this function pulls the calendar day in format: month day, year from a Shred data excel file
    """
    time = df.Time.str.split(pat=None,n=7,expand=False,regex=None)
    date = df.Time.get(1)
    date = date.split(" ")[0]
    year = int(date[0:2])+2000
    calendar = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    month = calendar[int(date[3:4])-1]
    day = date[4:6]
    cal_date = (month + " " + str(day) + ", " + str(year))
    return date, cal_date
    
def normalize_time(df):
    """
    This function may be able to be replaced with the datetime package but it changes the time strings given in the wind tunnel data to integer times starting at 0 and ending at the end of the test. Since this is not astrophysics, date and time of day are not important and can be removed to be stored in the metadata. This is what the getdate function does.
    """
    starttime = df.Time[1]
    starttime = (int(starttime[0:2])*3600) + (int(starttime[3:5])*60) + (int(starttime[6:8])) + (int(starttime[9:])/1000)
    for i in range(1,len(df)):
        time = df.Time[i]
        df.Time[i] = (int(time[0:2])*3600) + (int(time[3:5])*60) + (int(time[6:8])) + (int(time[9:])/1000)
        df.Time[i] = df.Time[i]-starttime
        
def import_new_data(filename):
    """
    This function reads a .xlxs data sheet from the wind tunnel and formats it by removing NaN data as well as creating some metadata for plotting
    """
    nanvalue = float("NAN")
    df = pd.read_excel(filename)
    df.replace(0,nanvalue,inplace=True)
    df.Yaw.replace(nanvalue, 0, inplace=True)
    df.dropna(axis=1, inplace=True)
    (date, cal_date) = get_date(df)
    df.Time = df.Time.str.removeprefix(date)
    df.Time = df.Time.str.removeprefix(" ")
    normalize_time(df)
    units = df['Units'][1]
    df = df.drop(['Units','Type'], axis = 1)
    return df, cal_date, units

def import_new_stl(stl_file):
    """
    This function reads an STL file and performs linear algebra to center the fin and prepare the file for profile analysis
    This function's output will be stored in the self.data of the SHRED.Fin object and can be used to 3D print the fin
    """
    if stl_file.endswith('.stl') == False:
        raise ValueError('That file-type is not supported')
    else:
        print(f"importing STL data from {stl_file}...")
        STL = mesh.Mesh.from_file(stl_file)
        print("done")
        print(f"calculating center of mass...")
        center_of_mass = STL.get_mass_properties()[1]
        print("done")
        print(f"centering fin at the origin...")
        vector_shape = STL.vectors.shape
        cog_vector = np.full((vector_shape), center_of_mass) 
        STL.vectors = STL.vectors - cog_vector #changes the vectors array to be centered with the cog at the origin
        ### I need to find a way to optimize this process, there has to be a more efficient way of doing these next few steps###
        print("done")
        print(f"building point cloud...")
        x=[]
        y=[]
        z=[]
        for i in range(0, len(STL.points)):
            x.append(STL.points[i][0])
            x.append(STL.points[i][3])
            x.append(STL.points[i][6])
            y.append(STL.points[i][1])
            y.append(STL.points[i][4])
            y.append(STL.points[i][7])
            z.append(STL.points[i][2])
            z.append(STL.points[i][5])
            z.append(STL.points[i][8])
        PC = np.stack([x,y,z]) #PC stands for point cloud
        print("done")
        print(f"calculating covariances and eigenvalues...")
        center = np.mean(PC, axis=1)
        cov = np.cov(PC)
        eigvalues, eigvectors =np.linalg.eig(cov)
        print("done")
        print(f"alligning fin with cartesian coordinate system")
        PC_centered = PC - center[:,np.newaxis]

        cart_alligned = np.matmul(eigvectors.T, PC_centered) #this alligns the point cloud with the cartesian coord system

        alligned_points = []
        for i in range(0,len(cart_alligned[0])):
            alligned_points.append([cart_alligned[0][i],cart_alligned[1][i],cart_alligned[2][i]])
        alligned_points=np.stack(alligned_points)
        STL.vectors = np.reshape(alligned_points, vector_shape)
        print("done")
        return STL     