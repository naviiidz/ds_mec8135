import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

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


def convert2pixel(position, xsc=1, ysc=1):
    # xsc & ysc in case we cannot visualize properly
    loc=position*scale+init_pixel
    return [int(loc[0]/xsc),int(loc[1]/ysc)]

def update_cov():
    global cov
    r=0.1;T=1/8
    A=np.eye(2)
    Rt=np.array([[0.1,0],[0,0.15]])*T
    cov=np.matmul(A,np.matmul(cov,np.transpose(A)))+Rt

def kalman():
    global cov, pose

    # values we already have:
    C=np.array([[1,0],[0,2]]).astype(float)    
    z=pose_measure

    Qt=np.array([[0.05,0],[0,0.075]])
    temp=np.matmul(C,np.matmul(cov,np.transpose(C)))+Qt

    k=np.matmul(cov,np.transpose(C))/temp
    k[np.isnan(k)] = 0
    print(k)
    temp2=z-np.matmul(C,pose)
    pose=pose+np.matmul(k,temp2)
    cov=np.matmul(np.eye(2)-np.matmul(k,C),cov)


def update_belief():
    global pose
    global pose_gt
    pre_pose=pose
    # pre_pose_gt=pose_gt
    r=0.1;T=1/8
    u=np.array([0.1,0.1])
    B_t=np.ones((2,2))*r*T/2
    A_t=np.eye(2)
    # pose of the robot
    e=np.array([np.random.normal(0,0.1),np.random.normal(0,0.15)])
    pose=np.matmul(B_t, u)+np.matmul(A_t, pre_pose)+e*T
    # ground truth pose
    pose_gt=np.matmul(B_t, u)+np.matmul(A_t, pre_pose)

def update_observation():
    global pose_measure, pose
    c_t=np.array([[1,0],[0,2]]).astype(float)    
    z=np.matmul(c_t,pose)+np.array([np.random.normal(0,0.05),np.random.normal(0,0.075)])
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
    while (np.linalg.norm(pose)<=1) and (not crashed): # One meter to the right
        update_belief()
        update_cov()
        if t%8==0:
            update_observation()
            kalman()

        # plotting position
        poses.append(convert2pixel(pose))
        pygame.draw.lines(gameDisplay,red,False,poses,5)

        # plotting ground truth
        poses_gt.append(convert2pixel(pose_gt))
        pygame.draw.lines(gameDisplay,blue,False,poses_gt,5)

        # plotting measurements
        poses_measure.append(convert2pixel(pose_measure,1,2))
        pygame.draw.lines(gameDisplay,green,False,poses_measure,5)

        # plotting uncertainty ellipse
        h=abs(abs(cov[0,0])*scale)
        w=abs(abs(cov[1,1])*scale)
        ellipse_size = (pose[0]*scale+init_pixel[0]-h/2,pose[1]*scale+init_pixel[1]-w/2,
                        h,w)
        pygame.draw.ellipse(gameDisplay, red, ellipse_size, 1)  

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


