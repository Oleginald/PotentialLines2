import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate, misc
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs, fmin, fmin_l_bfgs_b, fmin_slsqp, fminbound, fmin_cobyla
import pygame

def parse(filename):
    """Функция для преобразования точек, экспортированных из NX в двумерный массив.
     0 столбец - координаты х, а 1-ый - координаты y точек."""
    with open(filename) as file:
        data = file.readlines()

    arr = np.empty(0)
    rows = 0
    data[0] = data[0][3:]
    data = data[:len(data)][:]
    for row in data:
        # print(row)
        i1 = row.index(',')
        i2 = row[i1+1:].index(',') + i1 + 1
        arr = np.append(arr, float(row[:i1-1]))
        arr = np.append(arr, float(row[i2+1:-1]))
    arr = np.reshape(arr, (len(data), 2))
    # arr = arr[arr[:, 0].argsort()]
    # # Меняем столбцы по надобности
    # arr[:,[1,0]] = arr[:,[0,1]]
    return arr

class Line(object):
    """Класс определяющий параметрически заданную прямую на плоскости.\n
            x = x0 + a*t.
            y = y0 + b*t.
    Также используется форма: y = k*x + c."""

    def __init__(self, a: float, b: float, p: np.ndarray):
        """p - точка np.array([x0,y0])"""
        self.__a = a
        self.__b = b
        self.__x0 = p[0]
        self.__y0 = p[1]

    def __repr__(self):
        return f"Line:\nx = {self.__x0} + {self.__a}*x\ny = {self.__y0} + {self.__b}*y\n"

    @classmethod
    def line2Pts(cls, p0, p1):
        """p0, p1 - точки типа np.ndarray.
        a = x1 - x0, b = y1 - y0, self.p0 = p0"""
        return cls(p1[0] - p0[0], p1[1] - p0[1], p0)

    @classmethod
    def lineKPt(cls, k: float, p: np.ndarray):
        """Определяет прямую через коэффициент наколна k и свободный член c."""
        return cls(1, k, p)

    def get_k(self):
        return self.__b / self.__a

    def get_c(self):
        return (self.__b/self.__a) * self.__x0 + self.__y0

    def eval(self, t):
        """Возвращает декартовы координаты точки [x;y], соответствующей занчению параметра t"""
        return np.array([self.__x0 + self.__a * t, self.__y0 + self.__b * t])

    def evalX(self, x):
        '''Возрващает величину y, соответсвующую абсциссе х'''
        return self.__y0 + self.__a * ((x - self.__x0) / self.__b)

    def point_distance(self, p : np.ndarray):
        '''Возвращает дистанцию между точкой p и прямой.'''
        A = self.__a # коэффициент при Х
        B = -self.__b # коэффициент при Y
        C = -self.__a*self.__x0 + self.__b*self.__y0
        return abs(A*p[0] + B*p[1] + C) / (A ** 2 + B ** 2) ** 0.5

    def pltDraw(self, t):
        plt.plot([self.__x0, self.eval(t)[0]], [self.__y0, self.eval(t)[1]])


class Spline(object):
    '''Сплайн задается параметрически. tck - массив контрольных узлов. u - параметр определяющий положений точки на сплайне от его длины.'''
    @staticmethod
    def find_tck(points):
        tck, u = scipy.interpolate.splprep(points.transpose(), s=0)
        return tck

    def __init__(self, tck):
        self.__tck = tck

    def eval(self, q1):
        '''Возвращает массив [x,y] - координаты точки на сплайне, соответствующие u.'''
        res = scipy.interpolate.splev(q1, self.__tck)
        res = np.array([float(res[0]), float(res[1])])
        return res

    def der(self, q1, dq1 = 1e-08):
        dx = self.eval(q1+dq1)[0] - self.eval(q1-dq1)[0]
        print('dx=',dx)
        dy = self.eval(q1+dq1)[1] - self.eval(q1-dq1)[1]
        print('dy=',dy)
        return dy/dx

    def perp(self, q1):
        return math.tan(math.atan(self.der(q1)) + math.pi / 2)

    def pltDraw(self, num = 1000):
        qs = np.linspace(0, 1, num)
        res = scipy.interpolate.splev(qs, self.__tck)
        plt.plot(res[0], res[1])


def distance2p(p1 : np.ndarray, p2 : np.ndarray) -> float:
    '''Дистанция между двумя точками.'''
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def intersection(s : Spline, l : Line) -> float:
    """Функция находит точку пересечения линии и сплайна."""
    # assert obj1.get_limits()[0] > obj2.get_limits()[1] or obj1.get_limits()[1] < obj2.get_limits()[0] , f"Указанные объекты на пересекаются в пределах их лимитов.\n"
    def MAE(q1):
        p = s.eval(q1)
        distance = l.point_distance(p)
        return distance ** 2
    result = fminbound(MAE, 0, 1)
    return float(result)

def createPerp(s : Spline, p : np.ndarray) -> Line:
    """Функция возвращает прямую проходяющую через точку и перпендикулярную сплайну.
    Будем находить нормаль не через направляющие коэффиценты, как в ранних версиях, а через минимизацию расстояния между точками."""

    def MAE(q1):
        return distance2p(p, s.eval(q1))**2
    res = fminbound(MAE, 0, 1)
    res = Line.line2Pts(p, s.eval(res))
    return res

def eqAreaSolver(s1 : Spline, s2 : Spline, q1 : float, am : float) -> float:
    """s1 - сплайн, описывающий ведущий диск. s2 - сплайн, описывающий покрывной диск.
    q1 - продольная координата, на которой строится точка, между s1 и s2.
    am - коэффициент определяющий соотношение площадей."""
    print(f'Starting area solving.')
    p1 = s1.eval(q1)
    p2 = s2.eval(q1)
    l1 = Line.line2Pts(p1,p2)

    def MAE(t):
        '''area1 и area2 - это полуплощади усеченных конусов.'''
        p0 = l1.eval(t)
        l01 = createPerp(s1, p0)
        l02 = createPerp(s2, p0)
        p01 = s1.eval(intersection(s1,l01))
        p02 = s2.eval(intersection(s2,l02))
        length1 = distance2p(p0, p01)
        length2 = distance2p(p0, p02)
        area1 = math.pi * abs(p0[0]+p01[0])/2 * length1
        area2 = math.pi * abs(p0[0]+p02[0])/2 * length2
        return (am*area1 - area2) ** 2

    res = fmin_slsqp(MAE, 0.5)
    print(f'Area solving have done successfully.')
    return l1.eval(res)

def createPotential(s1 : Spline, s2 : Spline, am : float, num = 100) -> np.ndarray:
    '''am - коэффициент определяющий соотношение площадей. num - количество точек для вычисления.'''
    h = (s1.get_limits()[1] - s1.get_limits()[0]) / num
    xs = np.arange(s1.get_limits()[0], s1.get_limits()[1], h)
    ys = np.empty(0)
    # Непосредственно само вычисление точек
    for i in xs:
        ys = np.append(ys, eqAreaSolver(s1, s2, i, am))
    ys = np.reshape(ys, (num,2))

    # Чистка точек, от паразитно сгенерированных
    idel = np.empty(0, dtype='int')
    for i in range(num):
        c1 = abs(s1.eval(ys[i,0]) - ys[i,1]) <= 0.1
        c2 = abs(s2.eval(ys[i,0]) - ys[i,1]) <= 0.1
        c3 = ys[i,1] > s1.eval(ys[i,0])
        c4 = ys[i,1] < s2.eval(ys[i,0])
        if c1 or c2  or c3 or c4:
            idel = np.append(idel, i)

    ys = np.delete(ys, idel, 0)
    return ys
