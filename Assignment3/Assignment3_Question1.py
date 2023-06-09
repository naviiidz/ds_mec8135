import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random

# For the purpose of illustration
init_pixel=np.array([50,50])
scale=1000
display_width = 1000
display_height = 1000

# initial state of the robot
pose=np.zeros(2)
pose_gt=np.zeros(2)
pose_measure=np.zeros(2)
pose_k=np.zeros(2)

mu=np.zeros((2))
cov=np.array([[0,0],[0,0]]).astype(float)

poses=[(init_pixel[0],init_pixel[0])]
poses_gt=[(init_pixel[0],init_pixel[0])]
poses_measure=[(init_pixel[0],init_pixel[0])]
poses_kalman=[(init_pixel[0],init_pixel[0])]

n_particles=100
particles=np.random.multivariate_normal(pose,[[0.0001,0],[0,0.0001]],n_particles)
weights=np.zeros((n_particles,1))

def convert2pixel(position, xsc=1, ysc=1):
    # xsc & ysc in case we cannot visualize properly
    loc=position*scale+init_pixel
    return [int(loc[0]/xsc),int(loc[1]/ysc)]

def gaussian(X, sigma=0.5):
    return np.exp(-((X[0] - pose[0])**2 + (X[1] - pose[1])**2) / (2 * sigma**2))


def motion_model(x, noise_binary=1):
    r=0.1;T=1/8
    u=np.array([1,1])
    B_t=np.ones((2,2))*r*T/2
    A_t=np.eye(2)
    e=np.array([np.random.normal(0,0.1),np.random.normal(0,0.15)])
    x_new=np.matmul(B_t, u)+np.matmul(A_t, x)+e*T*noise_binary
    return x_new

def update_belief():
    global pose
    global pose_gt
    T=1/8
    pre_pose_gt=pose_gt
    pre_pose=pose
    # pose of the robot
    pose=motion_model(pre_pose)
    # ground truth pose
    pose_gt=motion_model(pre_pose_gt,0)
    w_total=np.sum(weights)
    for p in range(n_particles):
        particles[p]=motion_model(particles[p])
        particles[p]+=weights[p]*particles[p]/w_total

def sensor_model(p=np.zeros((2,1))):
    c_t=np.array([[1,0],[0,2]]).astype(float)   
    p_out=np.matmul(c_t,p)+np.array([np.random.normal(0,0.05),np.random.normal(0,0.075)])
    return p_out

def update_observation():
    global pose_measure, pose, particles
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
    while (np.linalg.norm(pose)<=1) and (not crashed): # One meter to the right
        update_belief()
        if t%8==0:
            particles=np.random.multivariate_normal(pose,[[0.0001,0],[0,0.0001]],n_particles)
            update_observation()
            particle()
        resample()
            



        # plotting ground truth
        poses_gt.append(convert2pixel(pose_gt))
        pygame.draw.lines(gameDisplay,blue,False,poses_gt,5)

        # plotting measurements
        poses_measure.append(convert2pixel(pose_measure,1,2))
        pygame.draw.lines(gameDisplay,green,False,poses_measure,5)


        # plotting position
        poses.append(convert2pixel(pose))
        pygame.draw.lines(gameDisplay,red,False,poses,5)


        for p in range(len(weights)):
            pygame.draw.circle(gameDisplay, blue, convert2pixel(particles[p]),1 )  #weights[p][0]*3

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

        pygame.display.update() 

        # resetting screen to remove previous ellipses
        gameDisplay.fill(white)

        # this is the fps or frequency of operation
        clock.tick(2)

        # iterator for measurement
        t+=1


