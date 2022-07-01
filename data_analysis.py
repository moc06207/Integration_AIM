#!/usr/bin/env python
#-*- coding:utf-8 -*-	# 한글 주석을 달기 위해 사용한다.

import csv

import math

import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import copy
def RMSE(a,b):

    c = (float(a) - float(b)) ** 2

    d = np.sqrt(c)
    if d >= 100:
        c = (-float(a) - float(b)) ** 2
        d = np.sqrt(c)
    return d

def main():

    # ------------------------------
    f = open('data_kalman.csv', 'r', encoding="UTF-8")
    rdr = csv.reader(f)
    kalman = []
    for line in rdr:
        row_list = []
        for i in range(len(line)):
            num = float(line[i])
            row_list.append(num)
        kalman.append(row_list)
    kalman = np.array(kalman)
    f.close()
    print(len(kalman[0]))
    # ------------------------------
    f = open('vn300.csv', 'r', encoding="UTF-8")
    rdr = csv.reader(f)
    vn300 = []
    for line in rdr:
        row_list = []
        for i in range(len(line)):
            num = float(line[i])
            row_list.append(num)
        vn300.append(row_list)
    vn300 = np.array(vn300)
    f.close()

    # ------------------------------
    f = open('vn100.csv', 'r', encoding="UTF-8")
    rdr = csv.reader(f)
    vn100 = []
    for line in rdr:
        row_list = []
        for i in range(len(line)):
            num = float(line[i])
            row_list.append(num)
        vn100.append(row_list)
    vn100 = np.array(vn100)
    f.close()

    # ------------------------------
    f = open('NED_data.csv', 'r', encoding="UTF-8")
    rdr = csv.reader(f)
    NED_data = []
    for line in rdr:
        row_list = []
        for i in range(len(line)):
            num = float(line[i])
            row_list.append(num)
        NED_data.append(row_list)
    NED_data = np.array(NED_data)
    f.close()

    print(len(kalman))
    print(len(vn300))
    plt_kalman = []
    for i in range(len(kalman)):
        for k in range(1):
            plt_kalman.append(kalman[i])

    plt_kalman = np.array(plt_kalman)

    plt_vn100 = []
    for i in range(len(vn100)):
        for k in range(1):
            plt_vn100.append(vn100[i])

    plt_vn100 = np.array(plt_vn100)

    error =[]
    for i in range(len(plt_vn100)-200):

        vn300_vs_kalman_MSE = RMSE(plt_kalman[i, 8], vn300[140 + i, 7])
        GPSyaw_vs_kalman_MSE = RMSE(plt_kalman[i, 8], plt_vn100[i, 14])

        pitch_MSE = RMSE(plt_kalman[i, 7], vn300[140 + i, 8])
        roll_MSE = RMSE(plt_kalman[i, 6], vn300[140 + i, 9])

        vn_MSE = RMSE(plt_kalman[i, 3], vn300[140 + i, 13])
        ve_MSE = RMSE(plt_kalman[i, 4], vn300[140 + i, 14])
        vd_MSE = RMSE(plt_kalman[i, 5], vn300[140 + i, 15])

        error.append([vn300_vs_kalman_MSE,
                      GPSyaw_vs_kalman_MSE,
                      pitch_MSE,
                      roll_MSE,
                      vn_MSE,
                      ve_MSE,
                      vd_MSE])


    error = np.array(error)

    error_2 =[]
    for i in range(len(plt_vn100)-200):
        vn300_vs_kalman_MSE = vn300[149 + i, 7] - plt_kalman[i, 8]
        if vn300_vs_kalman_MSE >= 100 or vn300_vs_kalman_MSE <= -100:
            vn300_vs_kalman_MSE = - vn300[149 + i, 7] - plt_kalman[i, 8]

        pitch_MSE = vn300[149 + i, 8] - plt_kalman[i, 7]
        roll_MSE =  vn300[149 + i, 9] - plt_kalman[i, 6]

        vn_MSE =  vn300[149 + i, 13] - plt_kalman[i, 3]
        ve_MSE =  vn300[149 + i, 14] - plt_kalman[i, 4]
        vd_MSE = vn300[149 + i, 15] - plt_kalman[i, 5]

        error_2.append([vn300_vs_kalman_MSE,pitch_MSE,roll_MSE,
                        vn_MSE, ve_MSE, vd_MSE])

    error_2 = np.array(error_2)

    print("vn300_vs_kalman_MSE std : ",np.std(error[:,0]))
    print("vn300_vs_kalman_MSE mean : ",np.mean(error[:,0]))

    print("GPSyaw_vs_kalman_MSE std : ", np.std(error[:, 1]))
    print("GPSyaw_vs_kalman_MSE mean : ", np.mean(error[:, 1]))

    print("pitch_MSE std : ", np.std(error[:, 2]))
    print("pitch_MSE mean : ", np.mean(error[:, 2]))

    print("roll_MSE std : ", np.std(error[:, 3]))
    print("roll_MSE mean : ", np.mean(error[:, 3]))

    print("vn_MSE std : ", np.std(error[:, 4]))
    print("vn_MSE mean : ", np.mean(error[:, 4]))

    print("ve_MSE std : ", np.std(error[:, 5]))
    print("ve_MSE mean : ", np.mean(error[:, 5]))

    print("vd_MSE std : ", np.std(error[:, 6]))
    print("vd_MSE mean : ", np.mean(error[:, 6]))

    x = np.arange(-0, len(plt_kalman))

    y = 0 * x + 0

    plt.figure(1)
    #141
    plt.subplot(311)
    plt.plot(plt_kalman[:, 8], 'r-',label='INS(Kalman Filter)')
    plt.plot(vn300[149:, 7], 'g-', label='vn300')
    plt.plot(plt_vn100[:, 14], 'b-', label='GPS_yaw')
    plt.ylabel('yaw(deg)')
    plt.title('yaw')

    plt.legend()

    plt.subplot(312)
    plt.plot(plt_kalman[:, 7], 'r-', )
    plt.plot(-vn300[149:, 8], 'g-', label='Truth')
    plt.ylabel('pitch(deg)')
    plt.title('pitch')
    print(-vn300[151, 8])
    print(-vn300[151, 9])
    plt.subplot(313)
    plt.plot(plt_kalman[:, 6], 'r-')
    plt.plot(-vn300[149:, 9], 'g-', label='vn300')
    plt.ylabel('roll(deg)')
    plt.title('roll')

    plt.figure(2)

    plt.subplot(311)
    plt.plot(plt_kalman[:, 3], 'r-', label='INS_Velocity(Kalman Filter)')
    plt.plot(plt_vn100[:, 3], 'b-', label='RTK_code_Velocity')
    plt.plot(vn300[149:, 13], 'g-', label='INS_Velocity')
    plt.plot(plt_vn100[:, 15], 'k-', label='RTK_real_Velocity')
    plt.ylabel('v_n(m/s)')
    plt.title('v_n')
    plt.legend()

    plt.subplot(312)
    plt.plot(plt_kalman[:, 4], 'r-')
    plt.plot(plt_vn100[:, 4], 'b-')
    plt.plot(vn300[149:, 14], 'g-')
    plt.plot(plt_vn100[:, 16], 'k-')
    plt.ylabel('v_e(m/s)')
    plt.title('v_e')

    plt.subplot(313)
    plt.plot(plt_kalman[:, 5], 'r-')
    plt.plot(plt_vn100[:, 5], 'b-')
    plt.plot(vn300[149:, 15], 'g-')
    plt.plot(plt_vn100[:, 17], 'k-')
    plt.ylabel('v_d(m/s)')
    plt.title('v_d')

    plt.figure(3)

    plt.subplot(311)
    #plt.plot(plt_kalman[:, 9], 'r-', label='INS_accel(Kalman Filter)')
    plt.plot(vn100[:, 6], 'b-', label='vn100_accel')
    plt.plot(x,y,'k-')
    plt.title('accel_x')
    plt.legend()

    plt.subplot(312)
    #plt.plot(plt_kalman[:, 10], 'r-')
    plt.plot(vn100[:, 7], 'b-')
    plt.plot(x,y,'k-')
    plt.title('accel_y')

    plt.subplot(313)
    #plt.plot(plt_kalman[:, 11], 'r-')
    plt.plot(vn100[:, 8], 'b-')
    plt.plot(x,y-9.8,'k-')
    plt.title('accel_z')

    plt.figure(4)

    plt.subplot(311)
    #plt.plot(plt_kalman[:, 12], 'r-', label='INS_accel(Kalman Filter)')
    plt.plot(vn100[:, 9], 'b-', label='vn100_accel')
    plt.plot(x,y,'k-')
    plt.title('angular_x')
    plt.legend()

    plt.subplot(312)
    #plt.plot(plt_kalman[:, 13], 'r-')
    plt.plot(vn100[:, 10], 'b-')
    plt.plot(x,y,'k-')
    plt.title('angular_y')

    plt.subplot(313)
    #plt.plot(plt_kalman[:, 14], 'r-')
    plt.plot(vn100[:, 11], 'b-')
    plt.plot(x,y,'k-')
    plt.title('angular_z')

    plt.figure(5)

    plt.subplot(111)
    plt.plot(vn100[:, 12], 'r-', label='dt')
    plt.plot(vn100[:, 13], 'b-', label='GPS_dt')
    plt.title('dt')
    plt.legend()

    plt.figure(6)

    plt.subplot(111)
    plt.plot(NED_data[:, 2], NED_data[:, 3], "or", markersize=4,label='RTK')
    plt.plot(NED_data[:, 0], NED_data[:, 1], "og", markersize=3,label='Kalman(vn100)')
    plt.xlim(sum(NED_data[:, 0]) / len(NED_data[:, 0]) - 22,sum(NED_data[:, 0]) / len(NED_data[:, 0]) + 22)
    plt.ylim(sum(NED_data[:, 1]) / len(NED_data[:, 1]) - 22, sum(NED_data[:, 1]) / len(NED_data[:, 1]) + 22)
    plt.title('RTK vs Kalman(vn100)')
    plt.legend()

    #error = np.array(error)

    plt.figure(7)
    plt.subplot(411)
    plt.plot(error[:, 0], 'r-', label='yaw error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('yaw error(deg)')
    plt.title('yaw error')

    plt.legend()
    plt.subplot(412)
    plt.plot(error[:, 1], 'r-', label='yaw error(GPS_yaw vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylim(-3,30)
    plt.ylabel('GPS_yaw error(deg)')
    plt.title('GPS_yaw error')

    plt.legend()
    plt.subplot(413)
    plt.plot(error[:, 2], 'r-', label='pitch error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('pitch error(deg)')
    plt.title('pitch error')
    plt.legend()
    plt.subplot(414)
    plt.plot(error[:, 3], 'r-', label='roll error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('roll error(deg)')
    plt.title('roll error')
    plt.legend()

    plt.figure(8)

    plt.subplot(311)
    plt.plot(error[:, 4], 'r-', label='vn error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('vn error(m/s)')
    plt.title('vn error')

    plt.legend()
    plt.subplot(312)
    plt.plot(error[:, 5], 'r-', label='ve error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('ve error(m/s)')
    plt.title('ve error')
    plt.legend()
    plt.subplot(313)
    plt.plot(error[:, 6], 'r-', label='vd error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('vd error(m/s)')
    plt.title('vd error')
    plt.legend()

    plt.figure(9)

    plt.subplot(311)
    plt.plot(error_2[:, 0], 'r-', label='yaw error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('yaw error(deg)')
    plt.title('yaw error')

    plt.legend()
    plt.subplot(312)
    plt.plot(error_2[:, 1], 'r-', label='pitch error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('pitch error(deg)')
    plt.title('pitch error')

    plt.legend()
    plt.subplot(313)
    plt.plot(error_2[:, 2], 'r-', label='roll error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('roll error(deg)')
    plt.title('roll error')
    plt.legend()

    plt.figure(10)

    plt.subplot(311)
    plt.plot(error_2[:, 3], 'r-', label='vn error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('vn error(m/s)')
    plt.title('vn error')
    plt.legend()
    plt.subplot(312)
    plt.plot(error_2[:, 4], 'r-', label='ve error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('ve error(m/s)')
    plt.title('ve error')
    plt.legend()
    plt.subplot(313)
    plt.plot(error_2[:, 5], 'r-', label='vd error(vn300 vs Kalman)')
    plt.plot(x,y,'k-')
    plt.ylabel('vd error(m/s)')
    plt.title('vd error')
    plt.legend()


    plt.show()





if __name__ == '__main__':
    main()
