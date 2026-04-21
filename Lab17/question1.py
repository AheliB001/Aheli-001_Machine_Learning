import numpy as np
import matplotlib.pyplot as plt

def transform(x1,x2):
    return np.array([x1**2,np.sqrt(2)*x1*x2,x2**2])

blue=np.array([
    [0,13],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1]
    ])

red=np.array([
    [3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]
])

plt.scatter(blue[:,0],blue[:,1],color='blue',label='blue')
plt.scatter(red[:,0],red[:,1],color='red',label='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Scatter Plot in 2D')
plt.show()

blue_3d=np.array([transform(x1,x2) for x1,x2 in blue])
red_3d=np.array([transform(x1,x2) for  x1,x2 in red])

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.scatter(blue_3d[:,0],blue_3d[:,1],blue_3d[:,2],color='blue',label='blue')
ax.scatter(red_3d[:,0],red_3d[:,1],red_3d[:,2],color='red',label='red')
ax.set_xlabel('x1**2')
ax.set_ylabel('sqrt x1 x2')
ax.set_zlabel('x2**2')
plt.title('Scatter Plot in 3D')
plt.show()


