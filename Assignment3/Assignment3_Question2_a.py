import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import math
import random

# For the purpose of illustration
init_pixel=np.array([500,500])
scale=1000
display_width = 1000
display_height = 1000

# initial state of the robot
pose=np.array([[0],[0],[0]])
pose_gt=np.array([[0],[0],[0]])
pose_measure=np.array([[0],[0],[0]])
pose_k=np.array([[0],[0],[0]])

mu=np.zeros((2))
cov=np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)

poses=[(init_pixel[0],init_pixel[1])]
poses_gt=[(init_pixel[0],init_pixel[1])]
poses_measure=[(init_pixel[0],init_pixel[1])]
poses_kalman=[(init_pixel[0],init_pixel[1])]

center=np.array([[10],[10]])
print(pose[0][0])
n_particles=100
particles=np.random.multivariate_normal([pose[0][0],pose[1][0]],[[0.0001,0],[0,0.0001]],n_particles)
weights=np.zeros((n_particles,1))

def convert2pixel(position, sc=1):
    # xsc & ysc in case we cannot visualize properly
    loc=position*scale/sc+init_pixel
    print(loc)
    return [int(loc[0]),int(loc[1])]

def gaussian(X, sigma=0.5):
    return np.exp(-((X[0] - pose[0])**2 + (X[1] - pose[1])**2) / (2 * sigma**2))

def motion_model(x, u, noise_binary=1, particle=0):
    r=0.1;T=1/8
    T=1/8
    r=0.1
    L=0.3
    A=np.eye(3)
    B=np.array([[r*T*np.cos(x[2,0]),0]
                ,[r*T*np.sin(x[2,0]),0]
                ,[0,T*r/L]])
    x_new=np.matmul(A,x)+np.matmul(B,u) + noise_binary*T*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
    return x_new

def update_belief():
    global pose
    global pose_gt
    pre_pose=pose
    # pre_pose_gt=pose_gt
    # pose of the robot
    # defining the jacobian G
    #G=np.array([[1,0,-T*r*uw*np.sin(theta)],[0,1,T*r*uw*np.sin(theta)],[0,0,1]])


    ur=ul=1

    print(math.dist([pose[0,0]*10,pose[1,0]*10],center))
    if (math.dist([pose[0,0],pose[1,0]],center)<10):
        ur=1
        ul=0
    if (math.dist([pose[0,0],pose[1,0]],center)>11):
        ur=0
        ul=1

    u=np.array([[(ur+ul)/2],[ur-ul]])

    pose=motion_model(pose, u, noise_binary=1)
    pose_gt=motion_model(pose_gt, u, noise_binary=0)

    w_total=np.sum(weights)
    for p in range(n_particles):
        temp=motion_model(np.array([[particles[p][0]],
                                   [particles[p][1]],
                                   pose[2]]),
                                   u, noise_binary=1,
                                   particle=1)
        particles[p][0]=temp[0]
        particles[p][1]=temp[1]
        particles[p]+=weights[p]*particles[p]/w_total


def sensor_model(p=np.zeros((2,1))):
    c_t=np.array([[1,0,0],[0,2,0],[0,0,1]]).astype(float)
    R=np.array([[np.random.normal(0,0.05)],
        [np.random.normal(0,0.075)],
        [0]])
    p_out=np.matmul(c_t,pose)+R
    return p_out

def update_observation():
    global pose_measure, pose
    z=sensor_model(pose_gt)
    pose_measure=z

def particle():
    global cov, pose, weights, particles
    particle_update=[]
    sum_=sum(weights)[0]+0.001
    print(sum_)
    for i in range(len(weights)):
        #print(np.array(particles[i]).shape)
        weights[i]=gaussian(sensor_model(particles[i]))
        particle_update.append(np.array(weights[i]/sum_)*np.array(particles[i]))
    particles+=particle_update

def resample():
    global weights, n_particles, particles
    #w=weights/np.array(weights).sum()
    #for i in range(1,n_particles):
    #    w[i]+=w[i-1]
    indices=random.choices(range(n_particles),  weights = weights, k=n_particles)
    particles=particles[indices]

if __name__=="__main__":
    pygame.init()
    pygame.display

    # the surface or our canvas for adding objs
    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('Python Simulation')

    # color definition
    white=(255,255,255)
    blue=(0,0,255)
    red=(255,0,0)
    green=(0,255,0)
    black=(0,0,0)

    gameDisplay.fill(white)

    # used for fps and sense of time
    clock = pygame.time.Clock()

    # just an init
    crashed = False

    # iterator for frequency
    t=0
    # our game loop
    while (not crashed): # One meter to the right
        update_belief()
        #update_cov()
        if t%8==0:
            particles=np.random.multivariate_normal([pose[0][0],pose[1][0]],[[0.0001,0],[0,0.0001]],n_particles)
            update_observation()
            particle()
        resample()

        # plotting position
        print(pose)

        poses.append(convert2pixel(np.array([pose[0,0],pose[1,0]])))
        pygame.draw.lines(gameDisplay,red,False,poses,5)

        # plotting ground truth
        poses_gt.append(convert2pixel(np.array([pose_gt[0,0],pose_gt[1,0]])))
        pygame.draw.lines(gameDisplay,blue,False,poses_gt,5)

        # plotting measurements
        poses_measure.append(convert2pixel(np.array([pose_measure[0,0],pose_measure[1,0]]),2))
        pygame.draw.lines(gameDisplay,green,False,poses_measure,5)

        for p in range(len(weights)):
            pygame.draw.circle(gameDisplay, blue, convert2pixel(particles[p]),1 )  #weights[p][0]*3


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        pygame.display.update() 

        # resetting screen to remove previous ellipses
        gameDisplay.fill(white)

        # this is the fps or frequency of operation
        clock.tick(8)

        # iterator for measurement
        t+=1


