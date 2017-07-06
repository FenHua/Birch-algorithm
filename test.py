from birch import*
import time

start=time.clock()
X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]

def GetData(filename):
    fr=open(filename)
    arrayLines=fr.readlines()
    numberofLines=len(arrayLines)
    returnMat=np.zeros((numberofLines,2))
    classLabelVector=[]
    index=0
    for lines in arrayLines:
        line=lines.strip()#去掉两边的换行符（回车符）
        listFromLine=line.split()#以tab分割
        returnMat[index,:]=listFromLine[1:3]
        classLabelVector.append(int(listFromLine[0]))#返回第一个元素作为label
        index+=1
    return returnMat,classLabelVector
#[returnMat,Label]=GetData('/home/yan/Data/Data5.dat')

def readMocks(filename):
    fr=open(filename)
    arraylines=fr.readlines()
    numberoflines=len(arraylines)
    returnMat=np.zeros((numberoflines,3))
    classLabelVector=[]
    index=0
    for line in arraylines:
        line=line.strip()#去掉两边的换行符
        listfromline=line.split(',')
        returnMat[index,:]=listfromline[2:5]
        classLabelVector.append(int(listfromline[1]))
        index+=1
    return returnMat,classLabelVector
#features,labels=readMocks('/home/yan/mocks/MS_mag_test1.txt')

brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True)
brc.fit(X)
print brc.predict(X)

end=time.clock()
print('The program consums %f seconds'%(end-start))
