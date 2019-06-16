# SophusPy
A python binding using pybind11 for Sophus which is a C++ Lie library.

## installation:
```bash
pip install sophuspy
```

## Examples
### 1. create SO3 and SE3
```py
import numpy as np
import sophus as sp

sp.SO3() # identity by default
'''
SO3([[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])
'''
sp.SO3([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
'''
SO3([[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])
'''

sp.SE3() # identity by default
'''
SE3([[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])
'''
sp.SE3(np.eye(4))
sp.SE3(np.eye(3), np.ones(3)) # R, t
'''
SE3([[1, 0, 0, 1],
     [0, 1, 0, 1],
     [0, 0, 1, 1],
     [0, 0, 0, 1]])
'''
```

### 2. multiplication
```py
R = sp.SO3()
R1 = sp.SO3([[0, 1, 0],
        	 [0, 0, 1],
        	 [1, 0, 0]])
R * R1
'''
SO3([[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]])
'''
R1 *= R # R1 = R1 * R

T = sp.SE3()
T1 = sp.SE3(R1.matrix(), np.ones(3))

T * T1
'''
SE3([[0, 1, 0, 1],
     [0, 0, 1, 1],
     [1, 0, 0, 1],
     [0, 0, 0, 1]])
'''
T1 *= T 	# T1 = T1 * T
```

### 3. rotate and translate points
```py
R = sp.SO3([[0, 1, 0],
        	[0, 0, 1],
        	[1, 0, 0]])
T = sp.SE3(R.matrix(), np.ones(3))

pt = np.array([1, 2, 3])
pts = np.array([[1, 2, 3],
				[4, 5, 6]])

R * pt 	# array([2., 3., 1.])
R * pts # array([[2., 3., 1.],
       	# 		 [5., 6., 4.]])

T * pt 	# array([3., 4., 2.])
T * pts # array([[3., 4., 2.],
        # 		 [6., 7., 5.]])
```

### 4. interfaces
```py
R = sp.SO3([[0, 1, 0],
        	[0, 0, 1],
        	[1, 0, 0]])
T = sp.SE3(R.matrix(), np.ones(3))

R.matrix()
'''
array([[0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]])
'''

R.log() # array([-1.20919958, -1.20919958, -1.20919958])

R.inverse()
'''
SO3([[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]])
'''

R.copy()

T.matrix()
'''
array([[0., 1., 0., 1.],
       [0., 0., 1., 1.],
       [1., 0., 0., 1.],
       [0., 0., 0., 1.]])
'''

T.matrix3x4()
'''
array([[0., 1., 0., 1.],
       [0., 0., 1., 1.],
       [1., 0., 0., 1.]])
'''
T.so3()
'''
SO3([[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]])
'''
T.log() # array([1., 1., 1., -1.20919958, -1.20919958, -1.20919958])
T.inverse()
'''
SE3([[ 0,  0,  1, -1],
     [ 1,  0,  0, -1],
     [ 0,  1,  0, -1],
     [ 0,  0,  0,  1]])
'''
T.copy()

T.translation() # array([1., 1., 1.])
T.rotationMatrix()
'''
array([[0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]])
'''
T.setRotationMatrix(np.eye(3)) 	# set SO3 matrix
T.setTranslation(np.zeros(3))	# set translation
```

### 5. static methods
```py
sp.SO3.hat(np.ones(3))
'''
array([[ 0., -1.,  1.],
       [ 1.,  0., -1.],
       [-1.,  1.,  0.]])
'''
sp.SO3.exp(np.ones(3))
'''
array([[ 0.22629564, -0.18300792,  0.95671228],
       [ 0.95671228,  0.22629564, -0.18300792],
       [-0.18300792,  0.95671228,  0.22629564]])
'''

sp.SE3.hat(np.ones(6))
'''
array([[ 0., -1.,  1.,  1.],
       [ 1.,  0., -1.,  1.],
       [-1.,  1.,  0.,  1.],
       [ 0.,  0.,  0.,  0.]])
'''
sp.SE3.exp(np.ones(6))
'''
array([[ 0.22629564, -0.18300792,  0.95671228,  1.        ],
       [ 0.95671228,  0.22629564, -0.18300792,  1.        ],
       [-0.18300792,  0.95671228,  0.22629564,  1.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
'''
```

### 6. others functions
```py
sp.copyto(R, R1) # copytoSO3(SO3d &dst, const SO3d &src)
sp.copyto(T, T1) # copytoSE3(SE3d &dst, const SE3d &src)

'''
not a strict rotation matrix. Uses Eigen3

Eigen::Quaterniond q(R);
q.normalized().toRotationMatrix();
'''
R_matrix = np.array([[1.   , 0.001, 0.   ],
       				 [0.   , 1.   , 0.   ],
       				 [0.   , 0.   , 1.   ]])
sp.to_orthogonal(R_matrix)
'''
array([[ 9.99999875e-01,  4.99999969e-04,  0.00000000e+00],
       [-4.99999969e-04,  9.99999875e-01, -0.00000000e+00],
       [-0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
'''
pose = T.matrix3x4().ravel() 	# array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.])
sp.invert_poses(pose) 			# array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]) identity matrix returns the same

poses = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
				  [0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1.]])
sp.invert_poses(poses)
'''
array([[ 1.,  0.,  0., -0.,  0.,  1.,  0., -0.,  0.,  0.,  1., -0.],
       [ 0.,  0.,  1., -1.,  1.,  0.,  0., -1.,  0.,  1.,  0., -1.]])
'''
points = np.array([[1., 2., 3.],
       			   [4., 5., 6.],
       			   [7., 8., 9.]])
sp.transform_points_by_poses(poses, points)
'''
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.],
       [ 7.,  8.,  9.],
       [ 3.,  4.,  2.],
       [ 6.,  7.,  5.],
       [ 9., 10.,  8.]])
'''
```