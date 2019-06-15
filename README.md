# SophusPy
A python binding using pybind11 for Sophus which is a C++ Lie library.

## installation:
```bash
cd SophusPy && mkdir build
cd build
cmake .. && make
```

after sophus\*.so generated, you can copy it to your python site-packages, or add this path to your python path, shown below
<br/>
```bash
export PYTHONPATH="$PYTHONPATH:/Path/to/your/SophusPy"
```

## test:
```py
pytest tests/
```

## Examples
```py
import sophus as sp

# 1. create SO3 and SE3
R = sp.SO3() 			# identity
R1 = sp.SO3(np.eye(3))

T = sp.SE3()			# identity
T1 = sp.SE3(np.eye(4))
T2 = sp.SE3(np.eye(3), np.zeros(3)) # R, t

# 2. multiplication
R2 = R * R1
R2 *= R 				# R2 = R2 * R

T3 = T * T1
T3 *= T 				# T3 = T3 * T

# 3. rotate and translate points
pt = np.ones(3)
pts = np.ones((4, 3))

pt_r = R * pt
pts_r = R * pts

pt_t = T * pt
pts_t = T * pts

# 4. interfaces
R.matrix()
R.log()
R.inverse()
R.copy()

T.matrix()
T.matrix()
T.matrix3x4()
T.so3()
T.log()
T.inverse()
T.copy()
T.translation()
T.rotationMatrix()
T.setRotationMatrix()
T.setTranslation()

# 5. static methods
sp.SO3.hat(np.ones(3))
sp.SO3.exp(np.ones(3))

sp.SE3.hat(np.ones(6))
sp.SE3.exp(np.ones(6))
```