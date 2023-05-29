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
cov=np.array([[0,0],[0,0]])

poses=[(init_pixel[0],init_pixel[0])]
poses_gt=[(init_pixel[0],init_pixel[0])]
poses_measure=[(init_pixel[0],init_pixel[0])]
poses_kalman=[(init_pixel[0],init_pixel[0])]


def convert2pixel(position, xsc=1, ysc=1):
    # xsc & ysc in case we cannot visualize properly
    loc=position*scale+init_pixel
    return [int(loc[0]/xsc),int(loc[1]/ysc)]
    

def kalman():
    global mu, cov, pose_k, pose

    # values we already have:
    r=0.1;T=1/8
    B=np.ones((2,2))*r*T/2
    A=np.eye(2)
    C=np.array([[1,0],[0,2]]).astype(float)    
    z=pose_measure
    u=np.array([1,1])
   

    Rt=np.array([[np.random.normal(0,0.1),0],[0,np.random.normal(0,0.15)]])
    Qt=np.array([[np.random.normal(0,0.05),0],[0,np.random.normal(0,0.075)]])


    mu_b=np.matmul(A,mu)+np.matmul(B,u)
    cov_b=np.matmul(A,np.matmul(cov,np.transpose(A)))+Rt
    temp=np.matmul(C,np.matmul(cov_b,np.transpose(C)))+Qt
    k=np.matmul(cov_b,np.transpose(C))*np.linalg.pinv(temp)
    k[np.isnan(k)] = 0
    temp2=z-np.matmul(C,mu_b)
    mu=mu_b+np.matmul(k,temp2)
    cov=np.matmul(np.eye(2)-np.matmul(k,C),cov_b)
    pose_k=mu


def update_belief():
    global pose
    global pose_gt
    pre_pose=pose
    pre_pose_gt=pose_gt
    r=0.1;T=1/8
    u=np.array([1,1])
    B_t=np.ones((2,2))*r*T/2
    A_t=np.eye(2)
    # pose of the robot
    e=np.array([np.random.normal(0,0.1),np.random.normal(0,0.15)])
    pose=np.matmul(B_t, u)+np.matmul(A_t, pre_pose)+e*T
    # ground truth pose
    pose_gt=np.matmul(B_t, u)+np.matmul(A_t, pre_pose)
    #z=update_observation(pre_pose,c_t)
    #kalman(A_t,B_t,c_t,u,z,Rt)
    #print(pose_gt)
    
def update_observation():
    global pose_measure, pose
    c_t=np.array([[1,0],[0,2]]).astype(float)    
    z=np.matmul(c_t,pose)+np.array([np.random.normal(0,0.05),np.random.normal(0,0.075)])
    pose_measure=z


pygame.init()

pygame.display

# the surface or our canvas for adding objs
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Python Simulation')

# for changing color
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
t=0
# our game loop
while (np.linalg.norm(pose_gt)<=1) and (not crashed): # One meter to the right
    # hiding previous state of robot using fill method
    #gameDisplay.fill(white, pygame.draw.circle(gameDisplay, blue, pose_pre.astype(int)  , 5))

    update_belief()
    if t%8==0:
        update_observation()
        kalman()
    #print(pose)


    poses.append(convert2pixel(pose))
    pygame.draw.lines(gameDisplay,red,False,poses,5)

    poses_gt.append(convert2pixel(pose_gt))
    pygame.draw.lines(gameDisplay,blue,False,poses_gt,5)

    poses_measure.append(convert2pixel(pose_measure,1,2))
    pygame.draw.lines(gameDisplay,green,False,poses_measure,5)

    poses_kalman.append(convert2pixel(pose_k,1,2))
    pygame.draw.lines(gameDisplay,black,False,poses_kalman,5)

    h=abs(cov[0,0]*scale*2)
    w=abs(cov[1,1]*scale*2)
    #print(w,h)
    #ellipse_size = (pose[0]*scale+init_pixel[0]-h/2,pose[1]*scale+init_pixel[1]-w/2,
    #                h,w)
    print(cov)
    #a=pygame.draw.ellipse(gameDisplay, red, ellipse_size, 1)  
    #pygame.draw.circle(gameDisplay, red, loc, 5)
    for event in pygame.event.get():
        # event is sort of all you do in the window
        if event.type == pygame.QUIT:
            crashed = True
        #print(event)


    
    # update a specific area of screen
    # note: use pygame.display.flip() 
    pygame.display.update() 
    gameDisplay.fill(white)
    # this is the fps
    clock.tick(8)
    t+=1


