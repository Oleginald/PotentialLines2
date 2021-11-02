from Lib import *

def main():

    filename1 = b"D:\Users\CHISTYAKOV\Detander_Methods\imports\Ch3.txt"
    arr = parse(filename1)
    # plt.plot(arr[:,0],arr[:,1],'+')
    # print(arr)
    tck = Spline.find_tck(arr)
    s_hub = Spline(tck, arr[0,0], arr[np.shape(arr)[0]-1,0])

    filename2 = b"D:\Users\CHISTYAKOV\Detander_Methods\imports\Cs3.txt"
    arr = parse(filename2)
    # plt.plot(arr[:,0],arr[:,1],'+')
    tck = Spline.find_tck(arr)
    s_shroud = Spline(tck, arr[0,0], arr[np.shape(arr)[0] - 1, 0])


    # находим среднюю линию
    # num = 20
    # am = 3
    # h = (s_hub.get_limits()[1] - s_hub.get_limits()[0]) / num
    # xs = np.arange(s_hub.get_limits()[0], s_hub.get_limits()[1], h)
    # ys = np.empty(0)
    # ys2 = np.empty(0)
    # for i in xs:
    #     ys2 = np.append(ys2, s_hub.eval(i))
    #     ys = np.append(ys, eqAreaSolver(s_hub, s_shroud, i, am))
    # ys = np.reshape(ys, (np.shape(xs)[0],2))


    A1 = createPotential(s_hub, s_shroud, 0.333333, 50)
    A2 = createPotential(s_hub, s_shroud, 1, 50)
    A3 = createPotential(s_hub, s_shroud, 3, 50)

    plt.plot(A1[:,0], A1[:,1], '+')
    plt.plot(A2[:,0], A2[:,1], '+')
    plt.plot(A3[:,0], A3[:,1], '+')


    # l1.pltDraw()
    # l2.pltDraw()
    # s_mid.pltDraw()

    s_hub.pltDraw()
    s_shroud.pltDraw()
    # plt.xlim(-5,s_hub.get_limits()[1]+10)
    # plt.ylim(-5,s_hub.eval(s_hub.get_limits()[1])+10)
    plt.xticks(np.arange(-5, s_hub.get_limits()[1] + 10, 10))
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()