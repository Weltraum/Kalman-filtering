import numpy as np
import matplotlib.pyplot as plt


def kalman(z):
    sz = (len(z), )

    Q = 1e-5  # Ковариационная матрица входящих шумов
    R = 1e-2  # Ковариационная матрица измерительных шумов

    xhat = np.zeros(sz)
    xhatminus = np.zeros(sz)
    P = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)

    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, len(z)):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k]*(z[k] - xhatminus[k])
        P[k] = (1 - K[k])*Pminus[k]

    return xhat


def kalman_1_adapt(z):
    sz = (len(z), )

    Q = 1e-5  # Ковариационная матрица входящих шумов
    #R = 1e-2  # Ковариационная матрица измерительных шумов

    xhat = np.zeros(sz)
    xhatminus = np.zeros(sz)
    P = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)
    V = np.zeros(sz)
    C = np.zeros(sz)

    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, len(z)):
        # time update
        xhatminus[k] = xhat[k-1]
        V[k] = z[k] - xhatminus[k]
        C[k] =(C[k-1]*(k-1) + V[k]**2)/k # np.sum(V[:k]**2)/k#
        Pminus[k] = P[k-1] + Q
        R = C[k] - Pminus[k]
        if R < 0: R = 0

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k]*V[k]
        P[k] = (1 - K[k])*Pminus[k]

    return xhat


def kalman_2_adapt(z):
    sz = (len(z), )

    R = 1e-2  # Ковариационная матрица измерительных шумов

    xhat = np.zeros(sz)
    xhatminus = np.zeros(sz)
    P = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)
    V = np.zeros(sz)
    C = np.zeros(sz)

    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, len(z)):
        # time update
        xhatminus[k] = xhat[k-1]
        V[k] = z[k] - xhatminus[k]
        C[k] = (C[k-1]*(k-1) + V[k]**2)/k
        Q = K[k-1] * C[k] * K[k-1]
        Pminus[k] = P[k-1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k]*V[k]
        P[k] = (1 - K[k])*Pminus[k]

    return xhat


def mean_filter(z):
    sz = (len(z), )
    xhat = np.zeros(sz)
    xhat[0] = 0.0
    for k in range(1, len(z)):
        xhat[k] = xhat[k-1] + (z[k] - xhat[k-1])/k

    return xhat


def f_sys(x):
    return -0.3727
    #return x/n_iter
    #return np.sin(x/(0.15*n_iter))

# initial parameters
n_iter = 500

x = [f_sys(x) for x in range(0, n_iter)]
z = [np.random.normal(z, 0.1) for z in x]

# plot
plt.figure()
plt.plot(z, 'y-', label='Измерение с шумами', alpha=0.5)
plt.plot(kalman(z), 'b-', label='Фильтр Калмана')
plt.plot(kalman_1_adapt(z), 'r.', label='Адаптивный фильтр Калмана 1-го рода', alpha=0.5)
plt.plot(kalman_2_adapt(z), 'r--', label='Адаптивный фильтр Калмана 2-го рода', alpha=0.5)
plt.plot(mean_filter(z), 'g-', label='Простое осреднение', alpha=0.5)
plt.plot(x, color='y', label='Истинные значения', alpha=0.3)
plt.legend()
plt.title('Фильтр')
plt.xlabel('Шаг')

plt.show()

