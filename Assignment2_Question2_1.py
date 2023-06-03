import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import math

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

def convert2pixel(position, sc=1):
    # xsc & ysc in case we cannot visualize properly
    loc=position*scale/sc+init_pixel
    print(loc)
    return [int(loc[0]),int(loc[1])]

def update_cov():
    global cov
    r=0.1;T=1/8
    A=np.eye(3)
    Rt=np.array([[0.01,0,0],[0,0.1,0],[0,0,0]])*T
    cov=np.matmul(A,np.matmul(cov,np.transpose(A)))+Rt

def kalman():
    global cov, pose

    # values we already have:
    C=np.array([[1,0,0],[0,2,0],[0,0,1]]).astype(float)    
    z=pose_measure

    Qt=np.array([[0.05,0,0],[0,0.075,0],[0,0,0]])
    temp=np.matmul(C,np.matmul(cov,np.transpose(C)))+Qt

    k=np.matmul(cov,np.transpose(C))/temp
    k[np.isnan(k)] = 0
    print(k)
    temp2=z-np.matmul(C,pose)
    pose=pose+np.matmul(k,temp2)
    cov=np.matmul(np.eye(3)-np.matmul(k,C),cov)


def update_belief():
    global pose
    global pose_gt
    pre_pose=pose
    # pre_pose_gt=pose_gt
    # pose of the robot
    # defining the jacobian G
    #G=np.array([[1,0,-T*r*uw*np.sin(theta)],[0,1,T*r*uw*np.sin(theta)],[0,0,1]])
    T=1/8
    r=0.1
    L=0.3

    ur=ul=1

    print(math.dist([pose[0,0]*10,pose[1,0]*10],center))
    if (math.dist([pose[0,0],pose[1,0]],center)<10):
        ur=1
        ul=0
    if (math.dist([pose[0,0],pose[1,0]],center)>11):
        ur=0
        ul=1

    A=np.eye(3)

    B=np.array([[r*T*np.cos(pose[2,0]),0]
                ,[r*T*np.sin(pose[2,0]),0]
                ,[0,T*r/L]])
    B_gt = np.array([[r*T*np.cos(pose[2,0]),0]
                     ,[r*T*np.sin(pose[2,0]),0]
                     ,[0,T*r/L]])

    u=np.array([[(ur+ul)/2],[ur-ul]])

    pose=np.matmul(A,pose)+np.matmul(B,u) + T*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
    pose_gt=np.matmul(A,pose_gt)+np.matmul(B_gt,u)


def update_observation():
    global pose_measure, pose
    c_t=np.array([[1,0,0],[0,2,0],[0,0,1]]).astype(float)    
    R=np.array([[np.random.normal(0,0.05)],
              [np.random.normal(0,0.075)],
              [0]])
    z=np.matmul(c_t,pose)+R
    pose_measure=z

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
            update_observation()
            kalman()

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

        # plotting uncertainty ellipse
        #h=abs(abs(cov[0,0])*scale)
        #w=abs(abs(cov[1,1])*scale)
        #ellipse_size = (pose[0]*scale+init_pixel[0]-h/2,pose[1]*scale+init_pixel[1]-w/2,
        #                h,w)
        #pygame.draw.ellipse(gameDisplay, red, ellipse_size, 1)  

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


