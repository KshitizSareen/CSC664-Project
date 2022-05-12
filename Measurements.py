import numpy as np
import cv2
import pandas as pd
import xlsxwriter
import math
import cv2
import numpy as np
import copy


def sortBlobs(blob):
    return blob['area']*-1

wb = xlsxwriter.Workbook('NewPositions6,6.xlsx')
Actualx = wb.add_worksheet('Actual Measurements x')
Actualy = wb.add_worksheet('Actual Measurements y')
ActualLeft=wb.add_worksheet('Actual Measurements Left')
ActualTop=wb.add_worksheet('Actual Measurements Top')
ActualWidth=wb.add_worksheet('Actual Measurements Width')
ActualHeight=wb.add_worksheet('Actual Measurements Height')
ActualVelocityX=wb.add_worksheet('velocityX')
ActualVelocityY=wb.add_worksheet('velocityY')

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class Point:
    def __init__(self,x,y):
        self.x=x
        self.y=y


def GetDistance(xOne,xTwo,yOne,yTwo):
    return ((xOne-xTwo)**2)+((yOne-yTwo)**2)

def CalculateArea(l1,r1,l2,r2):
    xDist=min(r1.x,r2.x)-max(l1.x,l2.x)
    yDist=max(r1.y,r2.y)-min(l1.y,l2.y)
    return abs(xDist*yDist)

def CheckIfOverlapping(l1,r1,l2,r2):
    if l1.x>=r2.x or l2.x>=r1.x:
        return False
    if r1.y<=l2.y or r2.y<=l1.y:
        return False
    
    return True

def getArea(blobOne,blobTwo):
    centroidOne=blobOne['centroid']
    centroidTwo=blobTwo['centroid']
    widthOne=blobOne['boxWidth']
    widthTwo=blobTwo['boxWidth']
    heightOne=blobOne['boxHeight']
    heightTwo=blobTwo['boxHeight']
    areaOne=blobOne['boxArea']
    areaTwo=blobTwo['boxArea']
    l1=Point(centroidOne[0]-widthOne/2,centroidOne[1]+heightOne/2)
    r1=Point(centroidOne[0]+widthOne/2,centroidOne[1]-heightOne/2)
    l2=Point(centroidTwo[0]-widthTwo/2,centroidTwo[1]+heightTwo/2)
    r2=Point(centroidTwo[0]+widthTwo/2,centroidTwo[1]-heightTwo/2)
    if CheckIfOverlapping(l1,r1,l2,r2):
        area=CalculateArea(l1,r1,l2,r2)
        return area
    return 0

def ComputeOverlapAreas(blobs,graph,old_gray,frame_gray):
    for i in range(len(blobs)):
        if blobs[i]['frameNumber']==1:
            currentBlob = blobs[i]
            Neighbours=[]
            for j in range(len(graph[i])):
                if graph[i][j]==1:
                    Neighbours.append(blobs[j])
            Neighbours.sort(key=sortBlobs)
            while 1:
                if len(currentBlob['blobIDs'])==0:
                    break
                checkIfFilled=True
                for neighbour in Neighbours:
                    if len(neighbour['blobIDs']) < neighbour['blobCapacity']:
                        checkIfFilled=False
                        break
                if checkIfFilled:
                    break
                for neighbour in Neighbours:
                    if len(neighbour['blobIDs']) < neighbour['blobCapacity']:
                        if len(currentBlob['blobIDs'])>0:
                            blobID=currentBlob['blobIDs'].pop()
                            neighbour['blobIDs'].append(blobID)
    for i in range(len(blobs)):
        if blobs[i]['frameNumber'] == 1:
            currentBlob = blobs[i]
            while len(currentBlob['blobIDs'])>0:
                blobID=currentBlob['blobIDs'].pop()
                currentPoint=[int(currentBlob['centroid'][0]),int(currentBlob['centroid'][1])]
                currentPixelValue = old_gray[currentPoint[1]][currentPoint[0]]
                currentRotation = currentBlob['rotation']
                minCost=float("inf")
                minBlob=None
                for blob in blobs:
                    if blob['frameNumber'] == 2:
                        if len(blob['blobIDs'])<blob['blobCapacity']:
                            newPoint=[blob['centroid'][0],blob['centroid'][1]]
                            currentCost=0
                            sumPixels=0
                            for j in range(math.floor(currentPoint[1]-7),math.ceil(currentPoint[1]+7)):
                                for k in range(math.floor(currentPoint[0]-7),math.ceil(currentPoint[0]+7)):
                                    if j>=0 and k>=0 and j<frame_gray.shape[0] and k<frame_gray.shape[1]:
                                        sumPixels+=((frame_gray[j][k]-currentPixelValue)**2)
                            currentCost+=sumPixels
                            euclideanDistance=GetDistance(currentPoint[0],newPoint[0],currentPoint[1],newPoint[1])
                            currentCost+=euclideanDistance
                            newRotation=blob['rotation']
                            rotationDifference = (newRotation-currentRotation)**2
                            currentCost+=rotationDifference
                            if currentCost<minCost:
                                minCost=currentCost
                                minBlob=blob
                if minBlob!=None:
                    minBlob['blobIDs'].append(blobID)        

    """
    for i in range(len(blobs)):
        if blobs[i]['frameNumber'] == 1:
            currentBlob = blobs[i]
            while len(currentBlob['blobIDs'])>0:
                blobID=currentBlob['blobIDs'].pop()
                currentPoint=[currentBlob['centroid'][0],currentBlob['centroid'][1]]
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, np.array([[currentPoint]],dtype=np.float32), None, **lk_params)
                newPoint = p1[0][0]
                minDistance = float("inf")
                minBlob=None
                for blob in blobs:
                    if blob['frameNumber'] == 2:
                        if len(blob['blobIDs'])<blob['blobCapacity']:
                            currentPoint=[blob['centroid'][0],blob['centroid'][1]]
                            distance = GetDistance(newPoint[0],currentPoint[0],newPoint[1],currentPoint[1])
                            if distance<minDistance:
                                minDistance=distance
                                minBlob=blob
                if minBlob!=None:
                    minBlob['blobIDs'].append(blobID)
    """
   
        

        

def FormGraph(blobsOne,blobsTwo):
    Edges=[]
    blobs=blobsOne+blobsTwo
    Graph=[[0 for i in range(len(blobs))] for j in range(len(blobs))]
    for i in range(len(blobs)):
        for j in range(i+1,len(blobs)):
            blobOne=blobs[i]
            blobTwo=blobs[j]
            if blobOne['frameNumber'] ==1 and blobTwo['frameNumber']==2:
                centroidOne=blobOne['centroid']
                centroidTwo=blobTwo['centroid']
                widthOne=blobOne['boxWidth']
                widthTwo=blobTwo['boxWidth']
                heightOne=blobOne['boxHeight']
                heightTwo=blobTwo['boxHeight']
                areaOne=blobOne['boxArea']
                areaTwo=blobTwo['boxArea']
                l1=Point(centroidOne[0]-widthOne/2,centroidOne[1]-heightOne/2)
                r1=Point(centroidOne[0]+widthOne/2,centroidOne[1]+heightOne/2)
                l2=Point(centroidTwo[0]-widthTwo/2,centroidTwo[1]-heightTwo/2)
                r2=Point(centroidTwo[0]+widthTwo/2,centroidTwo[1]+heightTwo/2)
                if CheckIfOverlapping(l1,r1,l2,r2):
                    l1=Point(centroidOne[0]-widthOne/2,centroidOne[1]+heightOne/2)
                    r1=Point(centroidOne[0]+widthOne/2,centroidOne[1]-heightOne/2)
                    l2=Point(centroidTwo[0]-widthTwo/2,centroidTwo[1]+heightTwo/2)
                    r2=Point(centroidTwo[0]+widthTwo/2,centroidTwo[1]-heightTwo/2)
                    area=CalculateArea(l1,r1,l2,r2)
                    minArea=min(areaOne,areaTwo)
                    if area >= 0.9*minArea:
                        Graph[i][j]=1
                        Graph[j][i]=1
                        Edges.append([i,j])
    
    return Graph,blobs,Edges


class BuildGraph:

    def __init__(self) -> None:
        self.minCost=float("inf")
        self.minGraph=[]

    def DFS(self,nodes,node,visited,Graph):
        visited[node]=True
        nodes.append(node)

        for i in range(len(Graph[node])):
            if Graph[node][i]==1 and visited[i]==False:
                nodes=self.DFS(nodes,i,visited,Graph)

        return nodes


    def ConnectedComponents(self,Graph):

        visited={}
        cc=[]
        
        for node in range(len(Graph)):
            visited[node]=False
        
        for node in range(len(Graph)):
            if visited[node]==False:
                nodes=[]
                cc.append(self.DFS(nodes,node,visited,Graph))
        
        return cc

    def GetDegree(self,node,Graph):
        degree=0
        for i in range(len(Graph[node])):
            if Graph[node][i]==1:
                degree+=1
        return degree

    def CheckValidComponent(self,Component,Graph):
        count=0
        for node in Component:
            degree=self.GetDegree(node,Graph)
            if degree>1:
                if count==1:
                    return False
                count+=1
        return True

    def ComputeGraph(self,Graph,blobs):

        components=self.ConnectedComponents(Graph)

        for component in components:
            if not self.CheckValidComponent(component,Graph):
                return float("inf")
        
        Parents=[]

        for node in range(len(Graph)):
            degree=self.GetDegree(node,Graph)
            if degree>1:
                Parents.append(node)
            elif degree==0:
                Parents.append(node)
            elif degree==1 and blobs[node]['frameNumber']==1:
                Parents.append(node)

        cost=0
        for parent in Parents:
            S=0
            for node in range(len(Graph[parent])):
                if Graph[parent][node]==1:
                    S+=blobs[node]['area']
            area=blobs[parent]['area']
            cost+=((abs(area-S))/(max(area,S)))
        
        return cost


    def RemoveEdges(self,index,Graph,Edges,blobs):
        cost=self.ComputeGraph(Graph,blobs)
        if cost<self.minCost:
            self.minCost=cost
            self.minGraph=copy.deepcopy(Graph)

        for i in range(index,len(Edges)):
            vertexOne=Edges[i][0]
            vertexTwo=Edges[i][1]

            Graph[vertexOne][vertexTwo]=0
            Graph[vertexTwo][vertexOne]=0

            self.RemoveEdges(i+1,Graph,Edges,blobs)

            Graph[vertexOne][vertexTwo]=1
            Graph[vertexTwo][vertexOne]=1


def GetDegree(node,Graph):
    degree=0
    for i in range(len(Graph[node])):
        if Graph[node][i]==1:
            degree+=1
    return degree       

def countEdges(nodes,Graph):
    count=0
    for node in nodes:
        for i in range(len(Graph[node])):
            if Graph[node][i]==1:
                count+=1
    return count//2
    

def GetEdges(Graph):
    Edges=[]
    for i in range(len(Graph)):
        for j in range(len(Graph)):
            if Graph[i][j] == 1:
                if [i,j] not in Edges and [j,i] not in Edges:
                    Edges.append([i,j])
    return Edges

def GetLargestBlob(blobs,Graph,node):
    maxArea=float("-inf")
    largestBlob=None
    for i in range(len(Graph[node])):
        if Graph[node][i] == 1:
            if blobs[i]['area'] > maxArea:
                maxArea=blobs[i]['area']
                largestBlob=blobs[i]
    return largestBlob


def ComputeVelocities(blobs,Graph,Edges,samplingInterval):
    B = 0.5
    for edge in Edges:
        blobOne=edge[0]
        blobTwo=edge[1]
        if GetDegree(blobOne,Graph) > 1:
            blobs[blobTwo]['velocityX'] = blobs[blobOne]['velocityX']
            blobs[blobTwo]['velocityY'] = blobs[blobOne]['velocityY']
        elif GetDegree(blobTwo,Graph) > 1:
            largestBlob=GetLargestBlob(blobs,Graph,blobTwo)
            blobs[blobTwo]['velocityX'] = largestBlob['velocityX']
            blobs[blobTwo]['velocityY'] = largestBlob['velocityY']
        elif GetDegree(blobTwo,Graph) == 1:
            centroidOne = blobs[blobOne]['centroid']
            centroidTwo = blobs[blobTwo]['centroid']
            xDist = abs(centroidOne[0] - centroidTwo[0])
            yDist = abs(centroidOne[1] - centroidTwo[1])
            newVelocityX = (B*(xDist / samplingInterval)) + ((1-B)*blobs[blobOne]['velocityX'])
            newVelocityY = (B*(yDist / samplingInterval)) + ((1-B)*blobs[blobOne]['velocityY'])
            blobs[blobTwo]['velocityX'] = newVelocityX
            blobs[blobTwo]['velocityY'] = newVelocityY

    return blobs

def imadjust(x,a,b,c,d,gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y.astype(int)


cap = cv2.VideoCapture('video.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
samplingInterval= 1 / fps
ok, frame = cap.read()
if not ok:
    exit()

data = pd.read_excel("Coordinates/M6.xlsx",header=None).values
point1= data[0][5]
point2 = data[1][5]
point3=data[2][5]
point4=data[3][5]

oldwell = frame[point3:(point3+point4+1),point1:(point1+point2+1)]
oldWellGray = cv2.cvtColor(oldwell,cv2.COLOR_BGR2GRAY)
background = cv2.cvtColor(cv2.imread('MedianFrames/backgroundWell6,6.png'),cv2.COLOR_BGR2GRAY)
diffImage = cv2.absdiff(oldWellGray,background)

blurImage = cv2.GaussianBlur(diffImage,(5,5),0,0)
contrastImage=imadjust(blurImage,0,0.095,0,1)/255
ret,binaryImage = cv2.threshold(blurImage,20,255,cv2.THRESH_BINARY)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage , 4 , cv2.CV_32S)

blobPropertiesOne=[]


color = np.random.randint(0, 255, (100, 3))


for i in range(1,num_labels):
    area=stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w=stats[i, cv2.CC_STAT_WIDTH]
    h=stats[i, cv2.CC_STAT_HEIGHT]
    boundingBoxArea=w*h
    density=area/boundingBoxArea
    object=binaryImage[y:y+h,x:x+h]
    moments=cv2.moments(object)
    Em00=moments['m00']
    Em10=moments['m10']
    Em01=moments['m01']
    Em11=moments['m11']
    Em02=moments['m02']
    Em20=moments['m20']
    Ex=Em10/Em00
    Ey=Em01/Em00

    a=(Em20/Em00)-(Ex**2)
    b=2*((Em11/Em00)-(Ex*Ey))
    c=(Em02/Em00)-(Ey**2)
    if a==c:
        Etheta=((0.25*math.pi)*180)/math.pi
    else:
        Etheta=(((0.5*math.atan((b/(a-c)))))*180)/math.pi
    if area>=30:
        Property={
        'area' : area,
        'boxArea' : boundingBoxArea,
        'density' : density,
        'velocityX' : 0,
        'velocityY' : 0,
        'boxWidth': w,
        'boxHeight' : h,
        'centroid' : centroids[i],
        'frameNumber' : 1,
        'topLeft': Point(x,y),
        'blobCapacity' : 0,
        'blobIDs' : [],
        'rotation' : Etheta
        }
        blobPropertiesOne.append(Property)
        """
        Actualx.write(k,0,centroids[i][0])
        Actualy.write(k,0,centroids[i][1])
        ActualLeft.write(k,0,x)
        ActualTop.write(k,0,y)
        ActualWidth.write(k,0,w)
        ActualHeight.write(k,0,h)
        ActualVelocityX.write(k,0,0)
        ActualVelocityY.write(k,0,0)
        """

blobPropertiesOne.sort(key=sortBlobs)

i=0
while i<8:
    for blob in blobPropertiesOne:
        if i==8:
            break
        blob['blobCapacity']+=1
        blob['blobIDs'].append(i)
        i+=1


ColorMap={}
for i in range(8):
    ColorMap[i]= color[i].tolist()

for blob in blobPropertiesOne:
    for id in blob['blobIDs']:
        Actualx.write(id,0,blob['centroid'][0])
        Actualy.write(id,0,blob['centroid'][1])
        ActualLeft.write(id,0,blob['topLeft'].x)
        ActualTop.write(id,0,blob['topLeft'].y)
        ActualWidth.write(id,0,blob['boxWidth'])
        ActualHeight.write(id,0,blob['boxHeight'])
        ActualVelocityX.write(id,0,blob['velocityX'])
        ActualVelocityY.write(id,0,blob['velocityY'])


l=1
while l<2000:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    well = frame[point3:(point3+point4+1),point1:(point1+point2+1)]
    wellGray = cv2.cvtColor(well,cv2.COLOR_BGR2GRAY)
    diffImage = cv2.absdiff(wellGray,background)

    blurImage = cv2.GaussianBlur(diffImage,(5,5),0,0)

    ret,binaryImage = cv2.threshold(blurImage,20,255,cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage , 4 , cv2.CV_32S)
    blobPropertiesTwo=[]
    
    for i in range(1,num_labels):
        area=stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w=stats[i, cv2.CC_STAT_WIDTH]
        h=stats[i, cv2.CC_STAT_HEIGHT]
        boundingBoxArea=w*h
        density=area/boundingBoxArea
        velocity=0
        object=binaryImage[y:y+h,x:x+h]
        moments=cv2.moments(object)
        Em00=moments['m00']
        Em10=moments['m10']
        Em01=moments['m01']
        Em11=moments['m11']
        Em02=moments['m02']
        Em20=moments['m20']
        Ex=Em10/Em00
        Ey=Em01/Em00

        a=(Em20/Em00)-(Ex**2)
        b=2*((Em11/Em00)-(Ex*Ey))
        c=(Em02/Em00)-(Ey**2)
        if a==c:
            Etheta=((0.25*math.pi)*180)/math.pi
        else:
            Etheta=(((0.5*math.atan((b/(a-c)))))*180)/math.pi
        if area>=30:
            Property={
                'trackID' : i,
                'area' : area,
                'boxArea' : boundingBoxArea,
                'density' : density,
                'velocityX' : 0,
                'velocityY' : 0,
                'boxWidth': w,
                'boxHeight' : h,
                'centroid' : centroids[i],
                'frameNumber' : 2,
                'topLeft': Point(x,y),
                'blobCapacity' : 0,
                'blobIDs' : [],
                'rotation' : Etheta
            }
            blobPropertiesTwo.append(Property)
    
    blobPropertiesTwo.sort(key=sortBlobs)
    
    
    i=0
    while i<8:
        for blob in blobPropertiesTwo:
            if i==8:
                break
            blob['blobCapacity']+=1
            i+=1
    
    for blob in blobPropertiesTwo:
        print(blob)
        print("\n")
    
    Graph,blobs,Edges=FormGraph(blobPropertiesOne,blobPropertiesTwo)
    buildGraph=BuildGraph()
    buildGraph.RemoveEdges(0,Graph,Edges,blobs) 
    minGraph=buildGraph.minGraph
    newEdges=GetEdges(minGraph)
    newBlobs = ComputeVelocities(blobs,minGraph,newEdges,samplingInterval)
    ComputeOverlapAreas(newBlobs,minGraph,oldWellGray,wellGray)

    for blob in newBlobs:
        if blob['frameNumber'] == 2:
            for id in blob['blobIDs']:
                cv2.rectangle(well,(blob['topLeft'].x,blob['topLeft'].y),(blob['topLeft'].x+blob['boxWidth'],blob['topLeft'].y+blob['boxHeight']),ColorMap[id],2)
                Actualx.write(id,l,blob['centroid'][0])
                Actualy.write(id,l,blob['centroid'][1])
                ActualLeft.write(id,l,blob['topLeft'].x)
                ActualTop.write(id,l,blob['topLeft'].y)
                ActualWidth.write(id,l,blob['boxWidth'])
                ActualHeight.write(id,l,blob['boxHeight'])
                ActualVelocityX.write(id,l,blob['velocityX'])
                ActualVelocityY.write(id,l,blob['velocityY'])
    

    blobPropertiesOne=[]
    for blob in newBlobs:
        if blob['frameNumber'] == 2:
            blob['frameNumber'] = 1
            blobPropertiesOne.append(blob)
    
    l+=1

    cv2.imshow('window',well)
    k=cv2.waitKey(40)

cv2.destroyAllWindows()
wb.close()