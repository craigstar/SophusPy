# SophusPy
A python binding using pybind11 for Sophus which is a C++ Lie library.(SO3 && SE3)

## Requirements:
1. Pybind11 https://github.com/pybind/pybind11 (Do not use pip to install, download the project and use camke to install manually)

## installation:
```bash
pip install sophuspy
```

## Examples
### 1. create SO3 and SE3
```py
import numpy as np
import sophuspy as sp

# 1. default constructor of SO3
sp.SO3()
'''
SO3([[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])
'''

# 2. constructor of SO3, accepts numpy and list
sp.SO3([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

# 3. default constructor of SE3
sp.SE3()
'''
SE3([[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])
'''

# 4. constructor of SE3, accepts numpy and list
sp.SE3(np.eye(4))

# 5. R, t constructor of SE3
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
# 1. SO3 * SO3
R * R1
'''
SO3([[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]])
'''

# 2.
R1 *= R


T = sp.SE3()
T1 = sp.SE3(R1.matrix(), np.ones(3))

# 3. SE3 * SE3
T * T1
'''
SE3([[0, 1, 0, 1],
     [0, 0, 1, 1],
     [1, 0, 0, 1],
     [0, 0, 0, 1]])
'''

# 4.
T1 *= T
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

# 1. single point
R * pt  # array([2., 3., 1.])

# 2. N points
R * pts # array([[2., 3., 1.],
        #        [5., 6., 4.]])

# 3. single point
T * pt  # array([3., 4., 2.])

# 4. N points
T * pts # array([[3., 4., 2.],
        #        [6., 7., 5.]])
```

### 4. interfaces
```py
R = sp.SO3([[0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]])
T = sp.SE3(R.matrix(), np.ones(3))

# 1. 
R.matrix()
'''
array([[0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]])
'''

# 2.
R.log() # array([-1.20919958, -1.20919958, -1.20919958])

# 3.
R.inverse()
'''
SO3([[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]])
'''

# 4.
R.copy()

# 5.
T.matrix()
'''
array([[0., 1., 0., 1.],
       [0., 0., 1., 1.],
       [1., 0., 0., 1.],
       [0., 0., 0., 1.]])
'''

# 6.
T.matrix3x4()
'''
array([[0., 1., 0., 1.],
       [0., 0., 1., 1.],
       [1., 0., 0., 1.]])
'''

# 7.
T.so3()
'''
SO3([[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]])
'''

# 8.
T.log() # array([1., 1., 1., -1.20919958, -1.20919958, -1.20919958])

# 9.
T.inverse()
'''
SE3([[ 0,  0,  1, -1],
     [ 1,  0,  0, -1],
     [ 0,  1,  0, -1],
     [ 0,  0,  0,  1]])
'''

# 10.
T.copy()

# 11.
T.translation() # array([1., 1., 1.])

# 12.
T.rotationMatrix()
'''
array([[0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]])
'''

# 13.
T.setRotationMatrix(np.eye(3))  # set SO3 matrix

# 14.
T.setTranslation(np.zeros(3))   # set translation
```

### 5. static methods
```py
# 1.
sp.SO3.hat(np.ones(3))
'''
array([[ 0., -1.,  1.],
       [ 1.,  0., -1.],
       [-1.,  1.,  0.]])
'''

# 2.
sp.SO3.exp(np.ones(3))
'''
array([[ 0.22629564, -0.18300792,  0.95671228],
       [ 0.95671228,  0.22629564, -0.18300792],
       [-0.18300792,  0.95671228,  0.22629564]])
'''

# 3.
sp.SE3.hat(np.ones(6))
'''
array([[ 0., -1.,  1.,  1.],
       [ 1.,  0., -1.,  1.],
       [-1.,  1.,  0.,  1.],
       [ 0.,  0.,  0.,  0.]])
'''

# 4.
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
# 1. copy SO3
sp.copyto(R, R1) # copytoSO3(SO3d &dst, const SO3d &src)

# 2. copy SE3
sp.copyto(T, T1) # copytoSE3(SE3d &dst, const SE3d &src)


# 3.if R is not a strict rotation matrix, normalize it. Uses Eigen3 
# Eigen::Quaterniond q(R);
# q.normalized().toRotationMatrix();
R_matrix = np.array([[1.   , 0.001, 0.   ],
                     [0.   , 1.   , 0.   ],
                     [0.   , 0.   , 1.   ]])

sp.to_orthogonal(R_matrix)
'''
array([[ 9.99999875e-01,  4.99999969e-04,  0.00000000e+00],
       [-4.99999969e-04,  9.99999875e-01, -0.00000000e+00],
       [-0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
'''

# 4. invert N poses in a row
pose = T.matrix3x4().ravel()    # array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.])
sp.invert_poses(pose)           # array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]) identity matrix returns the same

poses = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                  [0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1.]])
sp.invert_poses(poses)
'''
array([[ 1.,  0.,  0., -0.,  0.,  1.,  0., -0.,  0.,  0.,  1., -0.],
       [ 0.,  0.,  1., -1.,  1.,  0.,  0., -1.,  0.,  1.,  0., -1.]])
'''

# 6. transform N points by M poses to form N * M points
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
