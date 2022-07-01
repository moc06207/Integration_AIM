#!/usr/bin/env python
#-*- coding:utf-8 -*-	# 한글 주석을 달기 위해 사용한다.
import rospy

from sensor_msgs.msg import Imu, MagneticField, NavSatFix, PointCloud2

import time
from multiprocessing import Value, Process, Manager,Queue

from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import copy
import numpy as np
from scipy import linalg
import pymap3d as pm
from JU_matrix import Kalman_Filter, m_n_inverse, n_n_inverse,DCM2eul_bn,eul2DCM_bn,product_matrix_dx,product_matrix,skew,plus_matrix,minus_matrix,cross_product,n_1_inverse
import matplotlib.pyplot as plt
show_animation = True
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension, Float32MultiArray
import csv

# roll 과 pitch를 최초 값으로 고정하였음

def chatterCallback(data):



    current_accel_x.value, current_accel_y.value, current_accel_z.value = round(data.linear_acceleration.x,2), round(data.linear_acceleration.y,2), round(data.linear_acceleration.z,2)

    current_vel_x.value, current_vel_y.value,current_vel_z.value = round(data.angular_velocity.x,5), round(data.angular_velocity.y,5), round(data.angular_velocity.z,5)

    current_quat_x.value, current_quat_y.value, current_quat_z.value, current_quat_w.value = round(data.orientation.x,2), round(data.orientation.y,2), round(data.orientation.z,2), round(data.orientation.w,2)

    quat_list = [current_quat_x.value, current_quat_y.value,current_quat_z.value, current_quat_w.value]

    roll, pitch, yaw = euler_from_quaternion(quat_list)

    yaw = math.degrees(yaw)

    current_yaw.value = round(yaw, 4)
    current_roll.value = round(roll, 4)
    current_pitch.value = round(pitch, 4)

    IMU_CTC.value = 1
    IMU_CNT.value += 1
    if IMU_CNT.value > 255:
        IMU_CNT.value = 0
    
    # 이유는 모르겠으나 def GPSINS에서 yaw값을 산출하면 속도는 엄청나게 빠르나 중간에 계속 렉 걸리면서 컴퓨터가 멈춰버림
    # 안정성을 위해서 callback에서 처리하는게 나을까?



def GPSCallback(data):

    current_lat.value, current_lon.value, current_alt.value = data.latitude, data.longitude, data.altitude

    GPS_CTC.value = 1
    GPS_CNT.value += 1
    if GPS_CNT.value > 255:
        GPS_CNT.value = 0




def vn100_Kalman_filter(current_lat, current_lon, current_alt, current_accel_x,
    current_accel_y, current_accel_z,current_vel_x, current_vel_y,current_vel_z,current_yaw,
                  current_roll,current_pitch,GPS_CNT,kalman_yaw,GPS_CTC,
                        kalman_pitch, kalman_roll, kalman_lat,
                        kalman_lon, kalman_alt, kalman_NED_N, kalman_NED_E, kalman_NED_D,
                        kalman_ENU_E, kalman_ENU_N, kalman_ENU_U,
                        GPS_NED_N, GPS_NED_E, GPS_NED_D, GPS_ENU_E, GPS_ENU_N, GPS_ENU_U,IMU_CNT):

    GPS_comp_plot = np.empty((0, 2), float)

    Kalman_comp_plot= np.empty((0, 2), float)

    degree = math.pi / 180
    radian = 180 / math.pi


    while True:

        if GPS_CTC.value != 0:
            print("GPS/INS Integration Start!")

            Q = [[0 for j in range(15)] for i in range(15)]

            Q[0][0] = 1000.0  # LLH
            Q[1][1] = 1000.0
            Q[2][2] = 1000.0
            Q[3][3] = 100000 # v_ned
            Q[4][4] = 100000
            Q[5][5] = 100000.0
            Q[6][6] = 0.0001  # rpy
            Q[7][7] = 0.0001
            Q[8][8] = 0.0001
            Q[9][9] = 0.1  # accel
            Q[10][10] = 0.1
            Q[11][11] = 0.1
            Q[12][12] = 0.1  # gyro
            Q[13][13] = 0.1
            Q[14][14] = 0.1

            Q = np.array(Q) * 10000

            P = [[0 for j in range(15)] for i in range(15)]

            for i in range(15):
                P[i][i] = 0.01

            P = np.array(P)

            R = [[0 for j in range(6)] for i in range(6)]

            for i in range(6):
                R[i][i] = 1.0

            R = np.array(R)

            H = [[0 for j in range(15)] for i in range(6)]

            for i in range(6):
                H[i][i] = 1

            H = np.array(H)

            start_time = time.time()  # dt산출을 위해서! # 기존 로그기반 필터는 i ~ i + 1 로 시간을 산출했으나,
            # 실시간은 i-1 ~ i로 산출해야함!


            start_lat,start_lon,start_alt = current_lat.value, current_lon.value, current_alt.value
            time0 = time.time()
            print('Try Attitude Initialization')
            while True:
                if current_lat.value != start_lat and  current_lon.value != start_lon:
                    break
                else:
                    continue
            init_time = time.time() - time0
            print('Success Attitude Initialization')

            N, E, D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value, start_lat,
                                      start_lon, start_alt)

            GPS_v_n,GPS_v_e,GPS_v_d = (N - 0) / init_time,(E - 0) / init_time,(D - 0) / init_time # 1초 멈췄으니까 1로 처

            init_N, init_E, init_D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value,
                                                     start_lat, start_lon, start_alt)

            init_yaw = math.atan2(E, N) * radian #atan2 덕분에 사분면 상관없이 yaw 나옴
            # init_yaw = math.atan2(-1, 1) * radian
            # print(init_yaw) #이거 해보면 됨 그럼 이해

            X0 = np.array([current_lat.value,current_lon.value, current_alt.value,
                           GPS_v_n, GPS_v_e, GPS_v_d,
                           current_roll.value, current_pitch.value,
                           init_yaw])  # set initial state  RTK 와 IMU 를 이용했음
            constant_roll,constant_pitch =  current_roll.value, current_pitch.value
            dX0 = [
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1],
                [1],
                [2],
                [0.14],
                [0.14],
                [0.14],
                [0.0035],
                [0.0035],
                [0.0035],
            ]
            dX0 = np.array(dX0)

            ecc = 0.0818192  # earth's eccentricity
            R0 = 6378137.0  # earth's mean radius (m)
            Ome = 7.2921151467e-5  # earth's rotational rate (rad/s)
            g = 9.81  # gravity
            ecc_2 = ecc ** 2

            X = X0
            dX = copy.deepcopy(dX0)


            prev_N, prev_E, prev_D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value,
                                                     start_lat, start_lon, start_alt)

            # TODo interval[i]는 i부터 i+1까지의 시간이다.

            G_CNT = GPS_CNT.value
            I_CNT = IMU_CNT.value
            print("Start Kalman!")

            f = open("data_kalman.csv", "w")
            writer = csv.writer(f)

            f = open("vn100.csv", "w")
            vn100_writer = csv.writer(f)

            #GPS_dt = time.time() - GPS_start_time  # i-1 ~ i 까지의 시간차
            GPS_start_time = time.time()  # i-1 시간 다시 갱신
            GPS_dt = 0

            while True:


                dt = time.time() - start_time # i-1 ~ i 까지의 시간차
                start_time = time.time() # i-1 시간 다시 갱신

                # state update with X
                lat, lon, h = X[0], X[1], X[2]
                v_n, v_e, v_d = X[3], X[4], X[5]
                roll, pitch, yaw = X[6], X[7], X[8]


                # -- ros communication --
                kalman_roll.value , kalman_pitch.value, kalman_yaw.value =  float(X[6]), float(X[7]), float(X[8])
                kalman_lat.value  ,kalman_lon.value   , kalman_alt.value =  float(X[0]), float(X[1]), float(X[2])

                # earth model
                # matlab sind = sin(lat * pi / 180)
                Rm = R0 * (1 - ecc_2) / (1 - ecc_2 * (math.sin(lat * degree)) ** 2) ** 1.5
                Rt = R0 / (1 - ecc_2 * (math.sin(lat * degree)) ** 2) ** 0.5
                Rmm = (3 * R0 * (1 - ecc_2) * ecc_2 * math.sin(lat * degree)) * math.cos(lat * degree) / (
                            (1 - ecc_2 * (math.sin(lat * degree)) ** 2) ** 2.5)

                Rtt = R0 * ecc_2 * math.sin(lat * degree) * math.cos(lat * degree) / ((1 - ecc_2
                                                                                       * (math.sin(lat * degree)) ** 2) ** 1.5)

                # gyro measurement
                w_ibb = np.array([current_vel_x.value, current_vel_y.value, current_vel_z.value]) # gyro 값
                w_enn = np.array([v_e / (Rt + h), -v_n / (Rm + h), -v_e * math.tan(lat * degree) / (Rt + h)])
                rho_n, rho_e, rho_d = w_enn[0], w_enn[1], w_enn[2]
                w_ien = np.array([Ome * math.cos(lat * degree), 0, - Ome * math.sin(lat * degree)])
                w_inn = np.array([w_ien[0] + w_enn[0], w_ien[1] + w_enn[1], w_ien[2] + w_enn[2]])
                Cbn = np.array(eul2DCM_bn(roll, pitch,yaw))  # body to ned DCM

                # accel measurement
                f_ned = Cbn @ np.array([[current_accel_x.value],[current_accel_y.value], [current_accel_z.value]])  # body to ned accel

                f_n, f_e, f_d = f_ned[0][0], f_ned[1][0], f_ned[2][0]

                # mechanization
                Cbn = Cbn + (Cbn @ np.array(skew(w_ibb[0] - dX[12][0], w_ibb[1] - dX[13][0], w_ibb[2] - dX[14][0])) -
                             np.array(skew(w_inn[0], w_inn[1], w_inn[2])) @ Cbn) * dt

                roll, pitch,yaw = DCM2eul_bn(Cbn)  # attitude update

                kalman_yaw.value = yaw # 우리가 쓸 yaw 값 !!

                csv_save = np.array([yaw, pitch, roll])



                # -------------velocity update

                V_ned = np.array([v_n, v_e, v_d])

                m_2 = Cbn @ np.array([[current_accel_x.value - dX[9][0]],
                                      [current_accel_y.value - dX[10][0]],
                                      [current_accel_z.value - dX[11][0]]])

                m_5 = np.array([w_ien[0] * 2, w_ien[1] * 2, w_ien[2] * 2]) + np.array([w_enn[0], w_enn[1], w_enn[2]])

                m_7 = m_2 - np.array(np.cross(m_5, V_ned)).reshape((-1, 1))
                m_8 = (m_7 + np.array([[0], [0], [g]])) * dt
                V = V_ned.reshape((-1, 1)) + m_8

                # velocity update
                # -------------velocity update

                # -------------Position update

                lat = lat + (180 / math.pi) * (0.5 * (V[0][0] + v_n) / (Rm + h)) * dt
                lon = lon + (180 / math.pi) * (0.5 * (V[1][0] + v_e) / ((Rt + h) * math.cos(lat * degree))) * dt
                h = h - (180 / math.pi) * (0.5 * (V[2][0] + v_d)) * dt

                # -----------------constant_roll,constant_pitch
                X = np.array([lat, lon, h, V[0][0], V[1][0], V[2][0], roll, pitch, yaw])  # next state

                # Kalman filter

                F = np.array(
                    Kalman_Filter(lat, h, Rm, Rt, Rmm, Rtt, rho_n, rho_e, rho_d, v_n, v_e, v_d, f_n, f_e, f_d, w_ien, Cbn))
                A = dt * F
                A = linalg.expm(A)  # discretization F matrix

                # -------------Prediction
                dX = A @ dX  # error state prediction
                P = A @ P @ np.transpose(A) + Q  # P prediction

                X_copy = X.tolist()
                X_copy = copy.deepcopy(X_copy)


                if GPS_CNT.value > G_CNT: # GPS 정보가 들어옴
                    G_CNT = GPS_CNT.value

                    # 255 인 경우는 적용안됨
                    N, E, D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value, start_lat,
                                                             start_lon, start_alt)

                    # init_N, init_E, init_D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value,
                    #                                          start_lat, start_lon, start_alt)
                    GPS_yaw = math.atan2(E - prev_E, N - prev_N) * radian
                    # print("GPS : ", GPS_yaw)
                    # print("Kalman : ",yaw)
                    error = GPS_yaw - yaw
                    print("error : ",error)
                    # print('...')
                    GPS_dt = time.time() - GPS_start_time  # i-1 ~ i 까지의 시간차
                    GPS_start_time = time.time()  # i-1 시간 다시 갱신
                    GPS_v_n,GPS_v_e,GPS_v_d = (N - prev_N) / GPS_dt,(E - prev_E) / GPS_dt,(D - prev_D) / GPS_dt

                    K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)  # Kalman gain

                    z = np.array([
                        [X_copy[0] - current_lat.value],
                        [X_copy[1] - current_lon.value],
                        [X_copy[2] - current_alt.value],
                        [X_copy[3] - GPS_v_n],
                        [X_copy[4] - GPS_v_e],
                        [X_copy[5] - GPS_v_d]
                    ])

                    dX = dX + (K @ (z - (H @ dX)))

                    P = P - K @ H @ P

                    X[0] = X_copy[0] - dX[0]
                    X[1] = X_copy[1] - dX[1]
                    X[2] = X_copy[2] - dX[2]
                    X[3] = X_copy[3] - dX[3]
                    X[4] = X_copy[4] - dX[4]
                    X[5] = X_copy[5] - dX[5]

                    dX = copy.deepcopy(dX0)
                    prev_N, prev_E, prev_D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value, start_lat,
                                                         start_lon, start_alt)

                elif G_CNT == 255 and GPS_CNT.value == 0: # 255일때 대응 가능
                    G_CNT = GPS_CNT.value

                    N, E, D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value, start_lat,
                                              start_lon, start_alt)
                    GPS_dt = time.time() - GPS_start_time  # i-1 ~ i 까지의 시간차
                    GPS_start_time = time.time()  # i-1 시간 다시 갱신
                    GPS_v_n,GPS_v_e,GPS_v_d = (N - prev_N) / GPS_dt,(E - prev_E) / GPS_dt,(D - prev_D) / GPS_dt

                    K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)  # Kalman gain

                    z = np.array([
                        [X_copy[0] - current_lat.value],
                        [X_copy[1] - current_lon.value],
                        [X_copy[2] - current_alt.value],
                        [X_copy[3] - GPS_v_n],
                        [X_copy[4] - GPS_v_e],
                        [X_copy[5] - GPS_v_d]
                    ]) # error measurment update

                    dX = dX + (K @ (z - (H @ dX)))

                    P = P - K @ H @ P

                    X[0] = X_copy[0] - dX[0]
                    X[1] = X_copy[1] - dX[1]
                    X[2] = X_copy[2] - dX[2]
                    X[3] = X_copy[3] - dX[3]
                    X[4] = X_copy[4] - dX[4]
                    X[5] = X_copy[5] - dX[5]

                    dX = copy.deepcopy(dX0)
                    prev_N, prev_E, prev_D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value, start_lat,
                                                             start_lon, start_alt)

                else:
                    if I_CNT == IMU_CNT.value and G_CNT == GPS_CNT.value:  # 통신 문제 발생
                        time.sleep(0.1)
                        if I_CNT == IMU_CNT.value and G_CNT == GPS_CNT.value:
                            print("Emergency!! Stop Integration")
                            break
                    else:
                        I_CNT = IMU_CNT.value

                writer.writerow(X)
                vn100_writer.writerow(np.array([current_lat.value, current_lon.value, current_alt.value,
                                                GPS_v_n,GPS_v_e,GPS_v_d,
                                                current_accel_x.value, current_accel_y.value, current_accel_z.value,
                                                current_vel_x.value, current_vel_y.value, current_vel_z.value,
                                                dt, GPS_dt
                                                ]))

                #
                # kalman_ENU_E.value, kalman_ENU_N.value, kalman_ENU_U.value = pm.geodetic2enu(X[0], X[1], X[2], start_lat, start_lon, start_alt)
                #
                # Kalman_N, Kalman_E, Kalman_D = pm.geodetic2ned(X[0], X[1], X[2], start_lat, start_lon, start_alt)
                #
                # kalman_NED_N.value, kalman_NED_E.value, kalman_NED_D.value = Kalman_N, Kalman_E, Kalman_D
                #
                # GPS_NED_N_N, GPS_NED_E_E, GPS_NED_D_D = pm.geodetic2ned(float(current_lat.value), float(current_lon.value),
                #                                                   float(current_alt.value), start_lat, start_lon,
                #                                                   start_alt)
                #
                # GPS_ENU_E_E, GPS_ENU_N_N, GPS_ENU_U_U = pm.geodetic2enu(float(current_lat.value), float(current_lon.value),
                #                                                   float(current_alt.value), start_lat, start_lon,
                #                                                   start_alt)
                #
                # GPS_NED_N.value, GPS_NED_E.value, GPS_NED_D.value = GPS_NED_N_N, GPS_NED_E_E, GPS_NED_D_D
                #
                # GPS_ENU_E.value, GPS_ENU_N.value, GPS_ENU_U.value = GPS_ENU_E_E, GPS_ENU_N_N, GPS_ENU_U_U
                #
                #
                # Kalman_comp_plot = np.append(Kalman_comp_plot, np.array([[Kalman_E, Kalman_N]]), axis=0)
                # GPS_comp_plot = np.append(GPS_comp_plot, np.array([[GPS_NED_E_E, GPS_NED_N_N]]), axis=0)
                # if show_animation:
                #     plt.cla()
                #     # for stopping simulation with the esc key.
                #     plt.gcf().canvas.mpl_connect(
                #         'key_release_event',
                #         lambda event: [exit(0) if event.key == 'escape' else None])
                #
                #     try:
                #         # ENU 확인
                #
                #         plt.plot(Kalman_comp_plot[:, 0], Kalman_comp_plot[:, 1], "og", markersize=3)
                #         plt.plot(GPS_comp_plot[:, 0], GPS_comp_plot[:, 1], "or", markersize=3) # 현재 위치
                #
                #         # plt.xlim(Kalman_E - 3, Kalman_E + 3)
                #         # plt.ylim(Kalman_N - 3, Kalman_N + 3)
                #
                #         # plt.xlim(GPS_NED_E_E - 5, GPS_NED_E_E + 5)
                #         # plt.ylim(GPS_NED_N_N - 5, GPS_NED_N_N + 5)
                #
                #     except:
                #         pass
                #
                #     plt.grid(True)
                #     plt.pause(0.00001)




def GNSS_Subscribe():

    rospy.init_node('GNSS_Subscribe', anonymous=True)

    rospy.Subscriber("vectornav/IMU", Imu, chatterCallback)

    rospy.Subscriber("raw_data/fix", NavSatFix, GPSCallback)

    rospy.spin()


if __name__ == '__main__':
    current_lat = Value('d', 0.0)
    current_lon = Value('d', 0.0)
    current_alt = Value('d', 0.0)
    current_accel_x = Value('d', 0.0)
    current_accel_y = Value('d', 0.0)
    current_accel_z = Value('d', 0.0)
    current_vel_x = Value('d', 0.0)
    current_vel_y = Value('d', 0.0)
    current_vel_z = Value('d', 0.0)
    current_quat_x = Value('d', 0.0)
    current_quat_y = Value('d', 0.0)
    current_quat_z = Value('d', 0.0)
    current_quat_w = Value('d', 0.0)
    current_yaw = Value('d', 0.0)
    

    #obj = [dist,x_cent, y_cent, x_min,x_max, y_min, y_max]
    obj1_dist = Value('d', 0.0)
    obj2_dist = Value('d', 0.0)
    obj3_dist = Value('d', 0.0)
    obj4_dist = Value('d', 0.0)
    
    obj1_x_cent = Value('d', 0.0)
    obj2_x_cent = Value('d', 0.0)
    obj3_x_cent = Value('d', 0.0)
    obj4_x_cent = Value('d', 0.0)

    obj1_y_cent = Value('d', 0.0)
    obj2_y_cent = Value('d', 0.0)
    obj3_y_cent = Value('d', 0.0)
    obj4_y_cent = Value('d', 0.0)

    obj1_x_min = Value('d', 0.0)
    obj2_x_min = Value('d', 0.0)
    obj3_x_min = Value('d', 0.0)
    obj4_x_min = Value('d', 0.0)

    obj1_x_max = Value('d', 0.0)
    obj2_x_max = Value('d', 0.0)
    obj3_x_max = Value('d', 0.0)
    obj4_x_max = Value('d', 0.0)

    obj1_y_min = Value('d', 0.0)
    obj2_y_min = Value('d', 0.0)
    obj3_y_min = Value('d', 0.0)
    obj4_y_min = Value('d', 0.0)

    obj1_y_max = Value('d', 0.0)
    obj2_y_max = Value('d', 0.0)
    obj3_y_max = Value('d', 0.0)
    obj4_y_max = Value('d', 0.0)

    IMU_CTC    = Value('d', 0.0)
    IMU_CNT    = Value('d', 0.0)
    GPS_CTC    = Value('d', 0.0)
    GPS_CNT    = Value('d', 0.0)
    LIDAR_CTC  = Value('d', 0.0)
    LIDAR_CNT  = Value('d', 0.0)
    
    LIDAR_obj_1 = Value('d', 0.0)
    LIDAR_obj_2 = Value('d', 0.0)
    LIDAR_obj_3 = Value('d', 0.0)

    LIDAR_obj_4 = Value('d', 0.0)

    current_roll = Value('d', 0.0)
    current_pitch = Value('d', 0.0)

    kalman_pitch = Value('d', 0.0)
    kalman_roll = Value('d', 0.0)
    kalman_yaw = Value('d', 0.0)
    kalman_lat = Value('d', 0.0)
    kalman_lon = Value('d', 0.0)
    kalman_alt = Value('d', 0.0)

    kalman_NED_N = Value('d', 0.0)
    kalman_NED_E = Value('d', 0.0)
    kalman_NED_D = Value('d', 0.0)

    kalman_ENU_E = Value('d', 0.0)
    kalman_ENU_N = Value('d', 0.0)
    kalman_ENU_U = Value('d', 0.0)



    GPS_NED_N = Value('d', 0.0)
    GPS_NED_E = Value('d', 0.0)
    GPS_NED_D = Value('d', 0.0)

    GPS_ENU_E = Value('d', 0.0)
    GPS_ENU_N = Value('d', 0.0)
    GPS_ENU_U = Value('d', 0.0)




    processedQ = Queue()

    th3 = Process(target=vn100_Kalman_filter, args = (current_lat, current_lon, current_alt, current_accel_x,
    current_accel_y, current_accel_z,current_vel_x, current_vel_y,current_vel_z,current_yaw,current_roll,
                                                      current_pitch,GPS_CNT,kalman_yaw,GPS_CTC,
                                                      kalman_pitch, kalman_roll, kalman_lat,
                                                      kalman_lon, kalman_alt,kalman_NED_N, kalman_NED_E, kalman_NED_D,
                                                      kalman_ENU_E, kalman_ENU_N, kalman_ENU_U,
                                                      GPS_NED_N, GPS_NED_E, GPS_NED_D,
                                                      GPS_ENU_E, GPS_ENU_N, GPS_ENU_U,IMU_CNT))

    th3.start()

    th5 = Process(target=GNSS_Subscribe, args = ())
    th5.start()

    rospy.init_node('INS_Integration', anonymous=True)

    pub_p = rospy.Publisher('INS', Float32MultiArray, queue_size=100)
    pub_q = rospy.Publisher('ENU', Float32MultiArray, queue_size=30)
    pub_t = rospy.Publisher('NED', Float32MultiArray, queue_size=30)

    while True:

        INS_array = Float32MultiArray()
        ENU_array = Float32MultiArray()
        NED_array = Float32MultiArray()

        if kalman_lat.value != 0: # 칼만필터가 적용 되었을 때


            INS_array.data = [kalman_lat.value, kalman_lon.value, kalman_alt.value,
                              current_lat.value, current_lon.value, current_alt.value,
                              kalman_roll.value, kalman_pitch.value, kalman_yaw.value,
                              current_accel_x.value, current_accel_y.value, current_accel_z.value,
                              current_vel_x.value, current_vel_y.value, current_vel_z.value,
                              current_quat_x.value, current_quat_y.value, current_quat_z.value, current_quat_w.value,
                              GPS_NED_N.value, GPS_NED_E.value, GPS_NED_D.value,
                              GPS_ENU_E.value, GPS_ENU_N.value,GPS_ENU_U.value
                              ]

            ENU_array.data = [kalman_ENU_E.value, kalman_ENU_N.value, kalman_ENU_U.value]

            NED_array.data = [kalman_NED_N.value, kalman_NED_E.value, kalman_NED_D.value]



            pub_p.publish(INS_array)
            pub_q.publish(ENU_array)
            pub_t.publish(NED_array)

            time.sleep(0.01)

