import numpy as np
import math
import matplotlib.pyplot as plt
import random

pi=np.pi

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def make_ngon(n):
    if n == 4:
        return ([1, 1, -1, -1, 1],[-1, 1, 1, -1, -1])
    # Makes the heaxagonal bounds
    tabX=[1]
    tabY=[0]
    for i in range(1,n+1):
        tabX+=[np.cos(i*2*np.pi/n)]
        tabY+=[np.sin(i*2*np.pi/n)]
    # print(tabX)
    # print(tabY)
    return (tabX,tabY)

def getLines(tabX,tabY,n):
    # Gets the vertices and vectors of the hexagonal table for plotting
    lineEqs=[]
    for i in range(0,n):
        vX=tabX[i+1]-tabX[i]
        vY=tabY[i+1]-tabY[i]
        lineEqs.append([tabX[i],tabY[i],vX,vY])
    return lineEqs

def mat_mul(Rinv,T,R,vX,vY,vS):
    # Matrix multiplication function
    V=[[vS],[vX],[vY]]
    V= np.matmul(R,V)
    V = np.matmul(T,V)
    V = np.matmul(Rinv,V)
    return (float(V[1][0]),float(V[2][0]),float(V[0][0]))

def rotate(x):
    # Creates a rotational matrix
    return np.matrix([[1,0,0],[0,cos(x),sin(x)],[0,-sin(x),cos(x)]],dtype=float)

def reflect(pX,pY,vX,vY,vS,r,distanceCenters,type,intersectAng1,intesectAng2,moonBilliards):
    # No-slip reflection on the disperser type 0 is the unit disperser (left one)
    #type 1 is the disperser on the right with radius r
    vSVPP = vS

    # Calculates velocity ang for Phi=velocityAng - normalAng
    if vX==0 and vY>0:
        velocityAng = np.pi/2
    elif vX==0 and vY<0:
        velocityAng = np.pi*3/2
    else:
        if (vX>0 and vY>=0):
            velocityAng=np.arctan(vY/vX)
        elif(vX<0 and vY>=0):
            velocityAng=np.pi+np.arctan(vY/vX)
        elif (vX<0 and vY<=0):
            velocityAng=np.pi+np.arctan(vY/vX)
        else:
            velocityAng=2*np.pi+np.arctan(vY/vX)

    if type == 0:
        Tr = [[-cos(eta*pi),sin(eta*pi),0],[sin(eta*pi),cos(eta*pi),0],[0,0,-1]]
        if pX==0 and pY>0:
            Rinv=rotate(0)
            R=rotate(0)
            normalAng = 0
            (vX,vY,vS)=mat_mul(Rinv,Tr,R,vX,vY,vS)
        elif pX==0 and pY<0:
            Rinv=rotate(-np.pi)
            R=rotate(np.pi)
            normalAng = np.pi
            (vX,vY,vS)=mat_mul(Rinv,Tr,R,vX,vY,vS)
        else:
            if (pX>0 and pY>=0):
                normalAng=np.arctan(pY/pX)
            elif(pX<0 and pY>=0):
                normalAng=np.pi+np.arctan(pY/pX)
            elif (pX<0 and pY<=0):
                normalAng=np.pi+np.arctan(pY/pX)
            else:
                normalAng=2*np.pi+np.arctan(pY/pX)
            Rinv=rotate(np.pi/2-normalAng)
            R=rotate(normalAng-np.pi/2)
            (vX,vY,vS)=mat_mul(Rinv,Tr,R,vX,vY,vS)
        if moonBilliards:
            arc = normalAng-intersectAng1
        else:
            if pY>=0:
                arc = normalAng
            else:
                arc = 2*intersectAng1 - (2*np.pi - normalAng) + (2*np.pi - 2*intersectAng2)*r
        return (vX,vY,vS,velocityAng-normalAng,vSVPP,arc)
    else:
        pX=pX-distanceCenters
        Tr = [[-cos(eta*pi),sin(eta*pi),0],[sin(eta*pi),cos(eta*pi),0],[0,0,-1]]
        if pX==0 and pY>0:
            Rinv=rotate(0)
            R=rotate(0)
            normalAng = 0
            (vX,vY,vS)=mat_mul(Rinv,Tr,R,vX,vY,vS)
        elif pX==0 and pY<0:
            Rinv=rotate(-np.pi)
            R=rotate(np.pi)
            normalAng = np.pi
            (vX,vY,vS)=mat_mul(Rinv,Tr,R,vX,vY,vS)
        else:
            if (pX>0 and pY>=0):
                normalAng=np.arctan(pY/pX)
            elif(pX<0 and pY>=0):
                normalAng=np.pi+np.arctan(pY/pX)
            elif (pX<0 and pY<=0):
                normalAng=np.pi+np.arctan(pY/pX)
            else:
                normalAng=2*np.pi+np.arctan(pY/pX)
            Rinv=rotate(np.pi/2-normalAng)
            R=rotate(normalAng-np.pi/2)
            (vX,vY,vS)=mat_mul(Rinv,Tr,R,vX,vY,vS)
        if moonBilliards:
            arc = (2*np.pi-2*intersectAng1+(2*np.pi-normalAng-intesectAng2)*r)
        else:
            arc = intersectAng1 + (normalAng-intersectAng2)*r
        return (vX,vY,vS,velocityAng-normalAng,vSVPP,arc)


def BilliardIte(pX,pY,vX,vY,vS,r,distanceCenters,interctAng1,intersectAng2,isTorus,time,moonBilliards):
    bestTime=1000

    # Check what disperser we are on. Type 0 is the unit disperser (left one)
    # type 1 is the disperser on the right with radius r
    if(abs(pX**2+pY**2-1)<=10**(-13)):
        disp = 0
    else:
        disp = 1

    if(not isTorus):
        # if we pass in a point on the disperser we call reflect
        vX,vY,vS,Phi,vSVPP,arc=reflect(pX,pY,vX,vY,vS,r,distanceCenters,disp,intersectAng1,intersectAng2,moonBilliards)
    else:
        Phi=None
        vSVPP=None
        arc = None

    D1=((2*pY*vY+2*pX*vX)**2-4*(vX**2+vY**2)*(pY**2+pX**2-1))
    if D1>=0:
        # Calculates possible collision points
        t1=(-2*pY*vY-2*pX*vX+D1**0.5)/(2*(vX**2+vY**2))
        t2=(-2*pY*vY-2*pX*vX-D1**0.5)/(2*(vX**2+vY**2))
    else:
        t1=-1
        t2=-1

    D2=((2*pY*vY+2*pX*vX-2*vX*distanceCenters)**2-4*(vX**2+vY**2)*(pY**2+pX**2-2*pX*distanceCenters+distanceCenters**2-r**2))
    if D2>=0:
        # Calculates possible collision points
        t3=(-(2*pY*vY+2*pX*vX-2*vX*distanceCenters)+D2**0.5)/(2*(vX**2+vY**2))
        t4=(-(2*pY*vY+2*pX*vX-2*vX*distanceCenters)-D2**0.5)/(2*(vX**2+vY**2))
    else:
        t3=-1
        t4=-1

    times = [t1,t2,t3,t4]

    for ti in times:
        if ti>10**(-13):
            if ti<bestTime:
                bestPx=pX+vX*ti
                bestPy=pY+vY*ti
                bestTime=ti
    time+=bestTime
    if bestTime == 1000:
        return (0,0,0,0,0,False,0,0,0)
    return (bestPx,bestPy,vX,vY,vS,False,time,Phi,vSVPP,arc)


def torus(pX,pY,wall):
    # Makes the torus effect. Instead of reflecting it starts on the opposite side of the hexagonal bounds.
    if(isTorus):
        if(wall==1 or wall==4):
            pY=-pY
            if wall==1:
                wall=4
            else:
                wall=1
        elif(wall==0 or wall==3):
            rDis=(pX**2+pY**2)**0.5
            if wall==0:
                ang=np.arctan(pY/pX)
                pX=rDis*np.cos(4/3*np.pi-ang)
                pY=rDis*np.sin(4/3*np.pi-ang)
                wall=3
            else:
                ang=np.arctan(pY/pX)
                pX=rDis*np.cos(1/3*np.pi-ang)
                pY=rDis*np.sin(1/3*np.pi-ang)
                wall=0
        elif(wall==2 or wall==5):
            rDis=(pX**2+pY**2)**0.5
            if wall==2:
                ang=np.arctan(pY/pX)
                pX=rDis*np.cos(10/6*np.pi-ang)
                pY=rDis*np.sin(10/6*np.pi-ang)
                wall=5
            else:
                ang=np.arctan(pY/pX)
                pX=rDis*np.cos(2/3*np.pi-ang)
                pY=rDis*np.sin(2/3*np.pi-ang)
                wall=2

        else:
            print("WARNING: VERTEX HIT")
    return (pX,pY,wall)

def box(pX,pY,wall):
    # Makes the torus effect. Instead of reflecting it starts on the opposite side of the hexagonal bounds.
    if(wall==0 or wall==2):
        pX=-pX
        if wall==0:
            wall=2
        else:
            wall=0
    elif(wall==1 or wall==3):
        pY=-pY
        if wall==1:
            wall=3
        else:
            wall=1
    else:
        print("WARNING: VERTEX HIT")
    return (pX,pY,wall)

def getXYAngMoon(r,epsilon,nXY,nAng,xIntersect):
    # Gets initial positions with angles.
    xyPos=[[],[],[]]
    xDomain = np.linspace(-1,xIntersect-epsilon,nAng)
    for x in xDomain:
        for angle in np.linspace(-np.pi/2+epsilon,np.pi/2-epsilon,nAng):
            y=-((1-x**2)**0.5)
            xyPos[0].append(x+epsilon/100)
            xyPos[1].append(y+epsilon/100)
            xyPos[2].append(angle)
    return xyPos

def getXYAngLemon(r,epsilon,nXY,nAng,xIntersect):
    # Gets initial positions with angles.
    xyPos=[[],[],[]]
    xDomain = np.linspace(xIntersect+epsilon,1,nAng)
    for x in xDomain:
        for angle in np.linspace(np.pi/2+epsilon,np.pi*3/2-epsilon,nAng):
            y=-((1-x**2)**0.5)
            xyPos[0].append(x-epsilon/100)
            xyPos[1].append(y+epsilon/100)
            xyPos[2].append(angle)
    return xyPos

def getIntersectAng(xIntersect,yIntersect,distanceCenters):
    if xIntersect==0 and yIntersect>0:
        intersectAng1 = 0
    elif xIntersect==0 and yIntersect<0:
        intersectAng1 = np.pi
    else:
        if (xIntersect>0 and yIntersect>=0):
            intersectAng1=np.arctan(yIntersect/xIntersect)
        elif(xIntersect<0 and yIntersect>=0):
            intersectAng1=np.pi+np.arctan(yIntersect/xIntersect)
        elif (xIntersect<0 and yIntersect<=0):
            intersectAng1=np.pi+np.arctan(yIntersect/xIntersect)
        else:
            intersectAng1=2*np.pi+np.arctan(yIntersect/xIntersect)
    xIntersect=xIntersect-distanceCenters
    if xIntersect==0 and yIntersect>0:
        intersectAng2 = 0
    elif xIntersect==0 and yIntersect<0:
        intersectAng2 = np.pi
    else:
        if (xIntersect>0 and yIntersect>=0):
            intersectAng2=np.arctan(yIntersect/xIntersect)
        elif(xIntersect<0 and yIntersect>=0):
            intersectAng2=np.pi+np.arctan(yIntersect/xIntersect)
        elif (xIntersect<0 and yIntersect<=0):
            intersectAng2=np.pi+np.arctan(yIntersect/xIntersect)
        else:
            intersectAng2=2*np.pi+np.arctan(yIntersect/xIntersect)
    return intersectAng1,intersectAng2


################################################################################
################################ Interact ######################################
################################################################################
r=1 # Radius of the disperser on the "right"
eta= np.arccos(1/3)/np.pi # Mass distribution
N=5 # Number of collistions
nXY=3 # Number of starting positions around half of disperser type 0
nAng = 3 # Number of launch angles between -pi/2 and pi/2 or for lemon pi/2 and 3*pi/2
distanceCenters = 0.5 # Distance between the centers
moonBilliards = False # True = Moon Billiards, False = Lemon Billiards
################################################################################
################################################################################
epsilon=0.0001
########################## TRAJECTORY MAP ######################################
xIntersect = (distanceCenters**2 + 1 - r**2)/(2*distanceCenters)
yIntersect = (r**2 - xIntersect**2)**0.5

intersectAng1,intersectAng2 = getIntersectAng(xIntersect,yIntersect,distanceCenters)

if moonBilliards:
    xyang=getXYAngMoon(r,epsilon,nXY,nAng,xIntersect)
else:
    xyang=getXYAngLemon(r,epsilon,nXY,nAng,xIntersect)

# for pX,pY,startAng in zip(xyang[0],xyang[1],xyang[2]):
startAng = math.radians(27)
pX=0.2
pY=-0.1

fig, ax = plt.subplots()
ax.set_aspect('equal')
vX=np.cos(startAng)
vY=np.sin(startAng)
vS=0
norm = (vX**2 + vY**2 + vS**2)**0.5
vX/=norm
vY/=norm
vS/=norm
trajX=[]
trajY=[]
isTorus=True # Don't change unless we are starting on the disperses
time=0
disp = 1
type = 1

if moonBilliards:
    fname = 'moon_R'+str(round(r,3))+'_B'+str(round(distanceCenters,3))+'_eta'+str(round(eta,3))+'_pX'+str(round(pX,3))+'_pY'+str(round(pY,3))+'_ang'+str(round(startAng,3))+'_vS'+str(round(vS,3))+'_N'+str(N)
else:
    fname = 'lemon_R'+str(round(r,3))+'_B'+str(round(distanceCenters,3))+'_eta'+str(round(eta,3))+'_pX'+str(round(pX,3))+'_pY'+str(round(pY,3))+'_ang'+str(round(startAng,3))+'_vS'+str(round(vS,3))+'_N'+str(N)

for i in range(0,N):
    trajX.append(pX)
    trajY.append(pY)
    (pX,pY,vX,vY,vS,isTorus,time,Phi,vSVPP,arc)=BilliardIte(pX,pY,vX,vY,vS,r,distanceCenters,intersectAng1,intersectAng2,isTorus,time,moonBilliards)
    if time==0:
        print("WARNING- Something went wrong at "+str(i)+"th iteration")
        break
    print(Phi,vSVPP,arc)
    trajX.append(pX)
    trajY.append(pY)
    trajX.append(None)
    trajY.append(None)

ax.plot(trajX, trajY,'k',linewidth=1)
if moonBilliards:
    xDomain1 = np.linspace(-1,xIntersect,500)
    xCirc1=[]
    yCirc1=[]
    for xcirc in xDomain1:
        xCirc1.append(xcirc)
        yCirc1.append((1-xcirc**2)**0.5)
    xCirc1.append(None)
    yCirc1.append(None)
    for xcirc in xDomain1:
        xCirc1.append(xcirc)
        yCirc1.append(-((1-xcirc**2)**0.5))
    xDomain2 = np.linspace(distanceCenters-r,xIntersect,500)
    xCirc2=[]
    yCirc2=[]
    for xcirc in xDomain2:
        xCirc2.append(xcirc)
        yCirc2.append((r**2-(xcirc-distanceCenters)**2)**0.5)
    xCirc2.append(None)
    yCirc2.append(None)
    for xcirc in xDomain2:
        xCirc2.append(xcirc)
        yCirc2.append(-((r**2-(xcirc-distanceCenters)**2)**0.5))
    plt.plot(xCirc1,yCirc1,c='black',linewidth=0.8)
    plt.plot(xCirc2,yCirc2,c='black',linewidth=0.8)
else:
    xDomain1 = np.linspace(xIntersect,1,500)
    xCirc1=[]
    yCirc1=[]
    for xcirc in xDomain1:
        xCirc1.append(xcirc)
        yCirc1.append((1-xcirc**2)**0.5)
    xCirc1.append(None)
    yCirc1.append(None)
    for xcirc in xDomain1:
        xCirc1.append(xcirc)
        yCirc1.append(-((1-xcirc**2)**0.5))
    xDomain2 = np.linspace(distanceCenters-r,xIntersect,500)
    xCirc2=[]
    yCirc2=[]
    for xcirc in xDomain2:
        xCirc2.append(xcirc)
        yCirc2.append((r**2-(xcirc-distanceCenters)**2)**0.5)
    xCirc2.append(None)
    yCirc2.append(None)
    for xcirc in xDomain2:
        xCirc2.append(xcirc)
        yCirc2.append(-((r**2-(xcirc-distanceCenters)**2)**0.5))
    plt.plot(xCirc1,yCirc1,c='black',linewidth=0.8)
    plt.plot(xCirc2,yCirc2,c='black',linewidth=0.8)
############################## Save of Show ###################################
plt.show()
# plt.axis('off')
# plt.savefig(fname+'.eps',transparent=True)
# plt.close('all')
