from matplotlib import pyplot as plt
import math
import numpy as np
import matplotlib.style as mplstyle
#This file is a cleaned version of my testing in 'TestingPythonCompatibility.ipynb' which is acting as a sort of playground
#The point of this file is to create a random set of points in a point cloud, move it to the origin, and rotate it to be vertical
#I will plot each step on a new plot which will appear when the script is ran

#Rotation Matrix Exploration Figure 1 is the point cloud on its own
plt.figure(1)
plt.suptitle('Point Cloud with no changes made')

#for the first version of this I want to work with this specific cloud, but future versions can generate random clouds to prove the ability to work on any set of points
point_cloud_start = [(3.7, 1.7), 
                   (4.1, 3.8), 
                   (4.7, 2.9), 
                   (5.2, 2.8), 
                   (6.0, 4.0), 
                   (6.3, 3.6), 
                   (9.7, 6.3), 
                   (10.0, 4.9), 
                   (11.0, 3.6), 
                   (12.5, 6.4)]
x = []
y = []
for i in range(len(point_cloud_start)):
    x.append(point_cloud_start[i][0])
    y.append(point_cloud_start[i][1])
plt.subplot(111)
plt.scatter(x,y)
plt.axis((-20,20,-20,20))

#Rotation Matrix Exploration Figure 2 is the point cloud with its center point
plt.figure(2)
plt.suptitle('Point Cloud and bounding box')

restacted_cloud = np.stack((x, y), axis = 0)
cov = np.cov(restacted_cloud) #covariance matrix
eigvalues, eigvectors = np.linalg.eig(cov)
teigvect = np.transpose(eigvectors)
#by dotting the original point cloud with the eigenvectors we can allign all of the points in the cloud with either the x and y
#axis, which is used to get the minimum and maximum values of the cloud
point_cloud_rot = np.dot(point_cloud_start, np.linalg.inv(teigvect))

min = np.min(point_cloud_rot, axis =0)
max = np.max(point_cloud_rot, axis= 0)

diff = (max-min)*0.5
center = min + diff

center = np.dot(center,teigvect)

plt.subplot(111)
plt.scatter(x,y)
plt.scatter([center[0]],[center[1]])    
plt.axis((-20,20,-20,20))

#Rotation Matrix Exploration Figure 3 is the point cloud moved to the center
plt.figure(3)
plt.suptitle('Point Cloud moved to origin')

center_x = round(center[0],1)
center_y = round(center[1],1)
for i in range(len(point_cloud_start)):
    x[i] = x[i] - center_x
    y[i] = y[i] - center_y

plt.subplot(111)
plt.scatter(x,y)
plt.axis((-20,20,-20,20))

#Rotation Matrix Exploration Figure 4 is the point cloud at the center with bounding box
plt.figure(4)
plt.suptitle('Point Cloud at origin with bounding box')

point_cloud_origin = []
for i in range(len(point_cloud_start)):
    point_cloud_origin.append((x[i], y[i]))

restacted_cloud_o = np.stack((x, y), axis = 0)
cov = np.cov(restacted_cloud_o) #covariance matrix of the cloud at origin
eigvalues, eigvectors = np.linalg.eig(cov)
teigvect = np.transpose(eigvectors)
#by dotting the original point cloud with the eigenvectors we can allign all of the points in the cloud with either the x and y
#axis, which is used to get the minimum and maximum values of the cloud
point_cloud_origin_rot = np.dot(point_cloud_origin, np.linalg.inv(teigvect))

min = np.min(point_cloud_origin_rot, axis =0)
max = np.max(point_cloud_origin_rot, axis= 0)

diff = (max-min)*0.5
center = min + diff

#This uses the long and short differences to find the corners with respect to the center
corners = np.array([center+[-diff[0],-diff[1]],
                    center+[diff[0],-diff[1]],
                    center+[diff[0],diff[1]],
                    center+[-diff[0],diff[1]],
                    center+[-diff[0],-diff[1]]])

corners = np.dot(corners,teigvect)
center = np.dot(center,teigvect)

plt.subplot(111)
plt.scatter(x,y)
plt.scatter([center[0]],[center[1]])
plt.plot(corners[:,0],corners[:,1],'-')
plt.axis((-20,20,-20,20))


#Rotation Matrix Exploration Figure 5 is the point cloud at the center with bounding box
plt.figure(5)
plt.suptitle('Point Cloud at origin with bounding box rotated vertical')

theta = angle = np.arctan2(corners[1][1]-corners[0][1],corners[1][0]-corners[0][0])
R = [[math.cos(-theta),-math.sin(-theta)],
    [math.sin(-theta), math.cos(-theta)]]
for i in range(len(corners)):
    corners[i] = np.matmul(R,corners[i])
for i in range(len(point_cloud_origin)):
    point_cloud_origin[i] = np.matmul(R,point_cloud_origin[i])
center = np.matmul(R,center)

x=[]
y=[]
for i in range(len(point_cloud_origin)):
    x.append(point_cloud_origin[i][0])
    y.append(point_cloud_origin[i][1])

plt.subplot(111)
plt.scatter([center[0]],[center[1]])
plt.scatter(x,y)
plt.plot(corners[:,0],corners[:,1],'-')
plt.axis((-20,20,-20,20))
plt.show()