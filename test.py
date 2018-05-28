import numpy as np
import sophus as sp

# print(sp.__file__)

Tnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787, -1103.428193030075],
                [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596, -35.93298768047541],
                [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105,  798.7780129474496],
                [                   0,                   0,                    0,                  1]])

T = sp.SE3(Tnp)
T.setTranslation(np.ones(3, dtype=np.float64))
T.setRotationMatrix(np.eye(3))
print(T * np.ones((3,1)))
print(T)