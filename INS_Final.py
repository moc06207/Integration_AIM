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
from geometry_msgs.msg import TwistWithCovarianceStamped

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

def GPSCallback(data):

    current_lat.value, current_lon.value, current_alt.value = data.latitude, data.longitude, data.altitude

    GPS_CTC.value = 1
    GPS_CNT.value += 1
    if GPS_CNT.value > 255:
        GPS_CNT.value = 0

def RMSE(a,b):

    c = (float(a) - float(b)) ** 2

    d = np.sqrt(c)
    if d >= 100:
        c = (-float(a) - float(b)) ** 2
        d = np.sqrt(c)
    return d

def vn100_Kalman_filter(current_lat, current_lon, current_alt, current_accel_x,
    current_accel_y, current_accel_z,current_vel_x, current_vel_y,current_vel_z,current_yaw,
                  current_roll,current_pitch,GPS_CNT,kalman_yaw,GPS_CTC,
                        kalman_pitch, kalman_roll, kalman_lat,
                        kalman_lon, kalman_alt, kalman_NED_N, kalman_NED_E, kalman_NED_D,
                        kalman_ENU_E, kalman_ENU_N, kalman_ENU_U,
                        GPS_NED_N, GPS_NED_E, GPS_NED_D, GPS_ENU_E, GPS_ENU_N, GPS_ENU_U,IMU_CNT,GPS_fix_velocity_x,GPS_fix_velocity_y,GPS_fix_velocity_z):


    GPS_comp_plot = np.empty((0, 2), float)

    Kalman_comp_plot= np.empty((0, 2), float)

    degree = math.pi / 180
    radian = 180 / math.pi

    while True:

        if GPS_CTC.value != 0:
            print("GPS/INS Integration Start!")

            Q = [[0 for j in range(15)] for i in range(15)]

            Q[0][0] = 10000.0  # LLH
            Q[1][1] = 10000.0
            Q[2][2] = 10000.0
            Q[3][3] = 10  # v_ned
            Q[4][4] = 10
            Q[5][5] = 1000
            Q[6][6] = 0.01 # rpy
            Q[7][7] = 0.01
            Q[8][8] = 1
            Q[9][9] = 10  # accel
            Q[10][10] = 10
            Q[11][11] = 10
            Q[12][12] = 10  # gyro
            Q[13][13] = 10
            Q[14][14] = 10

            Q = np.array(Q) * 10

            P = [[0 for j in range(15)] for i in range(15)]

            for i in range(15):
                P[i][i] = 0.01

            P = np.array(P)

            R = [[0 for j in range(6)] for i in range(6)]

            for i in range(6):
                R[i][i] = 2.0

            R = np.array(R)

            H = [[0 for j in range(15)] for i in range(6)]

            for i in range(6):
                H[i][i] = 1

            H = np.array(H)

            start_lat,start_lon,start_alt = current_lat.value, current_lon.value, current_alt.value
            time0 = time.time()
            print('Try Attitude Initialization')
            imu_cnt = IMU_CNT.value
            cal_bias = []

            # wait until change LLH(moving vehicle)
            while True:

                if current_lat.value != start_lat and  current_lon.value != start_lon:
                    break
                else:
                    if IMU_CNT.value > imu_cnt:
                        imu_cnt = IMU_CNT.value
                        cal_bias.append([current_lat.value, current_lon.value, current_alt.value,
                                         GPS_fix_velocity_y.value, GPS_fix_velocity_x.value, -GPS_fix_velocity_z.value,
                                         current_roll.value, current_pitch.value,current_yaw.value,
                                         current_accel_x.value,current_accel_y.value, current_accel_z.value,
                                         current_vel_x.value, current_vel_y.value, current_vel_z.value
                        ])

                    elif imu_cnt == 255 and IMU_CNT.value == 0:
                        imu_cnt = IMU_CNT.value
                        cal_bias.append([current_lat.value, current_lon.value, current_alt.value,
                                         GPS_fix_velocity_y.value, GPS_fix_velocity_x.value, -GPS_fix_velocity_z.value,
                                         current_roll.value, current_pitch.value, current_yaw.value,
                                         current_accel_x.value, current_accel_y.value, current_accel_z.value,
                                         current_vel_x.value, current_vel_y.value, current_vel_z.value])

            cal_bias = np.array(cal_bias)
            init_time = time.time() - time0
            print('Success Attitude Initialization')
            print('start lat : ',current_lat.value)
            print('start lon : ',current_lon.value)

            N, E, D = pm.geodetic2ned(current_lat.value, current_lon.value, current_alt.value, start_lat,
                                      start_lon, start_alt)

            init_yaw = math.atan2(E, N) * radian #atan2 덕분에 사분면 상관없이 yaw 나옴
            start_time = time.time()  # dt산출을 위해서!
            X0 = np.array([current_lat.value, current_lon.value, current_alt.value,
                              GPS_fix_velocity_y.value, GPS_fix_velocity_x.value, GPS_fix_velocity_z.value,
                              current_roll.value, current_pitch.value , init_yaw])  # set initial state  RTK 와 IMU 를 이용했음

            dX0 = np.array([[3.1249998144744495e-09], [2.4999999848063226e-08], [0.006000000000000449],
                            [0.0125625], [0.011875000000000002], [0.01665625],
                            [0.0], [0.0], [0.0], [0.0194921875],
                            [0.3706640625], [0.02261718750000026], [0.0005827734375], [0.0006302734375],
                            [0.00054640625]])
            dX0 = np.array(dX0)

            ecc = 0.0818192  # earth's eccentricity
            R0 = 6378137.0  # earth's mean radius (m)
            Ome = 7.2921151467e-5  # earth's rotational rate (rad/s)
            g = 9.81  # gravity
            ecc_2 = ecc ** 2

            X = X0
            dX = copy.deepcopy(dX0)

            G_CNT = GPS_CNT.value
            I_CNT = IMU_CNT.value
            print("Start Kalman!")

            imu_cnt = IMU_CNT.value
            cur_lat, cur_lon, cur_alt = current_lat.value, current_lon.value, current_alt.value

            I_CNT = IMU_CNT.value
            G_CNT_emergency = GPS_CNT.value

            prev_N, prev_E, prev_D = N, E, D

            while True:

                while True:
                    dt = time.time() - start_time  # i-1 ~ i 까지의 시간차
                    if dt >= 0.05:
                        break
                    else:
                        continue

                start_time = time.time()  # i-1 시간 다시 갱신

                while True:
                    GPS_Integration = 0
                    if GPS_CNT.value != G_CNT:

                        cur_lat, cur_lon, cur_alt = current_lat.value, current_lon.value, current_alt.value
                        GPS_vn, GPS_ve, GPS_vd = GPS_fix_velocity_y.value, GPS_fix_velocity_x.value, -GPS_fix_velocity_z.value
                        cur_accel_x, cur_accel_y, cur_accel_z = current_accel_x.value, current_accel_y.value, current_accel_z.value
                        cur_angluar_x, cur_angluar_y, cur_angluar_z = current_vel_x.value, current_vel_y.value, current_vel_z.value
                        G_CNT = GPS_CNT.value
                        imu_cnt = IMU_CNT.value
                        GPS_Integration = 1
                        break
                    else:
                        pass
                    if IMU_CNT.value != imu_cnt:

                        GPS_vn, GPS_ve, GPS_vd = GPS_fix_velocity_y.value, GPS_fix_velocity_x.value, -GPS_fix_velocity_z.value
                        cur_accel_x,cur_accel_y,cur_accel_z = current_accel_x.value, current_accel_y.value, current_accel_z.value
                        cur_angluar_x,cur_angluar_y,cur_angluar_z =  current_vel_x.value, current_vel_y.value, current_vel_z.value
                        imu_cnt = IMU_CNT.value
                        break
                    else:
                        pass

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

                # only gyro measurement(sensor measurment) # 순수 센서 angular vel
                w_ibb = np.array([cur_angluar_x, cur_angluar_y, cur_angluar_z]) # 순수 gyro 값

                # navigation component of angulat rate with respect to ECEF(ECEF에 대한 nav frame의 angular rate)
                w_enn = np.array([v_e / (Rt + h), -v_n / (Rm + h), -v_e * math.tan(lat * degree) / (Rt + h)])
                rho_n, rho_e, rho_d = w_enn[0], w_enn[1], w_enn[2]

                # earth rotation angular vel(지구 자전 각속도)
                w_ien = np.array([Ome * math.cos(lat * degree), 0, - Ome * math.sin(lat * degree)])

                # using poistion and velocity, calculate gyro(angular vel)
                w_inn = np.array([w_ien[0] + w_enn[0], w_ien[1] + w_enn[1], w_ien[2] + w_enn[2]])
                Cbn = np.array(eul2DCM_bn(roll, pitch,yaw))  # body to ned DCM #검수완료

                # accel measurement #boty to nav #검수완료
                f_ned = Cbn @ np.array([[cur_accel_x],[cur_accel_y], [cur_accel_z]])  # body to ned accel

                f_n, f_e, f_d = f_ned[0][0], f_ned[1][0], f_ned[2][0]

                # mechanization # DCM differential equation #검수완료
                Cbn = Cbn + (Cbn @ np.array(skew(w_ibb[0] - dX[12][0], w_ibb[1] - dX[13][0], w_ibb[2] - dX[14][0])) -
                             np.array( skew(w_inn[0], w_inn[1], w_inn[2])) @ Cbn) * dt

                roll, pitch,yaw = DCM2eul_bn(Cbn)  # attitude update

                # -------------velocity update

                V_ned = np.array([v_n, v_e, v_d])

                m_7 = Cbn @ np.array([[cur_accel_x - dX[9][0]],
                                      [cur_accel_y - dX[10][0]],
                                      [cur_accel_z- dX[11][0]]]) - \
                      np.array(np.cross(np.array([w_ien[0] * 2, w_ien[1] * 2, w_ien[2] * 2]) +
                                        np.array([w_enn[0], w_enn[1], w_enn[2]]), V_ned)).reshape((-1, 1))

                V = V_ned.reshape((-1, 1)) + (m_7 + np.array([[0], [0], [g]])) * dt

                # velocity update
                # -------------velocity update

                # -------------Position update

                lat = lat + (180 / math.pi) * (0.5 * (V[0][0] + v_n) / (Rm + h)) * dt
                lon = lon + (180 / math.pi) * (0.5 * (V[1][0] + v_e) / ((Rt + h) * math.cos(lat * degree))) * dt
                h = h - (180 / math.pi) * (0.5 * (V[2][0] + v_d)) * dt

                # -----------------constant_roll,constant_pitch
                #X = np.array([lat, lon, h, V[0][0], V[1][0], V[2][0], roll, pitch, yaw])  # next state
                X = np.array([lat, lon, h, 0.5 * (V[0][0] + v_n), 0.5 * (V[1][0] + v_e), 0.5 * (V[2][0] + v_d), roll, pitch, yaw])  # next state

                # Kalman filter

                F = np.array(
                    Kalman_Filter(lat, h,
                                  Rm, Rt, Rmm, Rtt,
                                  rho_n, rho_e, rho_d,
                                  v_n, v_e, v_d,
                                  f_n, f_e, f_d,
                                  w_ien,
                                  Cbn))

                A = linalg.expm(dt * F)  # discretization F matrix

                X_copy = X.tolist()

                X_copy = copy.deepcopy(X_copy)

                if GPS_Integration == 1: # GPS 정보가 들어옴

                    # 255 인 경우는 적용안됨
                    N, E, D = pm.geodetic2ned(cur_lat, cur_lon, cur_alt, start_lat,start_lon, start_alt)

                    GPS_v_n,GPS_v_e,GPS_v_d = GPS_vn, GPS_ve, GPS_vd

                    K = P @ np.transpose(H) @ np.linalg.inv(H @ P @ np.transpose(H) + R)  # Kalman gain

                    z = np.array([
                        [X_copy[0] - cur_lat],
                        [X_copy[1] - cur_lon],
                        [X_copy[2] - cur_alt],
                        [X_copy[3] - GPS_v_n],
                        [X_copy[4] - GPS_v_e],
                        [X_copy[5] - GPS_v_d]
                    ])

                    dX = dX + (K @ (z - (H @ dX)))

                    P = P - K @ H @ P

                    X[0] = X_copy[0] - dX[0][0]
                    X[1] = X_copy[1] - dX[1][0]
                    X[2] = X_copy[2] - dX[2][0]
                    X[3] = X_copy[3] - dX[3][0]
                    X[4] = X_copy[4] - dX[4][0]
                    X[5] = X_copy[5] - dX[5][0]

                    dX = copy.deepcopy(dX0)

                    GPS_yaw = math.atan2(E - prev_E, N - prev_N) * radian

                    prev_N, prev_E, prev_D = N, E, D

                    print(kalman_yaw.value - GPS_yaw)



                dX = A @ dX  # error state prediction
                P = A @ P @ np.transpose(A) + Q  # P prediction

                if I_CNT == IMU_CNT.value and G_CNT_emergency == GPS_CNT.value:  # 통신 문제 발생

                    time.sleep(0.1)
                    if I_CNT == IMU_CNT.value and G_CNT_emergency == GPS_CNT.value:
                        print("Emergency!! Stop Integration")
                        break
                else:
                    I_CNT = IMU_CNT.value
                    G_CNT_emergency = GPS_CNT.value

                kalman_ENU_E.value, kalman_ENU_N.value, kalman_ENU_U.value = pm.geodetic2enu(X[0], X[1], X[2], start_lat, start_lon, start_alt)

                Kalman_N, Kalman_E, Kalman_D = pm.geodetic2ned(X[0], X[1], X[2], start_lat, start_lon, start_alt)

                kalman_NED_N.value, kalman_NED_E.value, kalman_NED_D.value = Kalman_N, Kalman_E, Kalman_D

                GPS_NED_N_N, GPS_NED_E_E, GPS_NED_D_D = pm.geodetic2ned(float(cur_lat), float(cur_lon),
                                                                  float(cur_alt), start_lat, start_lon,
                                                                  start_alt)

                GPS_ENU_E_E, GPS_ENU_N_N, GPS_ENU_U_U = pm.geodetic2enu(float(cur_lat), float(cur_lon),
                                                                  float(cur_alt), start_lat, start_lon,
                                                                  start_alt)

                GPS_NED_N.value, GPS_NED_E.value, GPS_NED_D.value = GPS_NED_N_N, GPS_NED_E_E, GPS_NED_D_D

                GPS_ENU_E.value, GPS_ENU_N.value, GPS_ENU_U.value = GPS_ENU_E_E, GPS_ENU_N_N, GPS_ENU_U_U

                Kalman_comp_plot = np.append(Kalman_comp_plot, np.array([[Kalman_E, Kalman_N]]), axis=0)
                GPS_comp_plot = np.append(GPS_comp_plot, np.array([[GPS_NED_E_E, GPS_NED_N_N]]), axis=0)

                if show_animation:
                    plt.cla()
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect(
                        'key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])

                    try:
                        # ENU 확인

                        plt.plot(Kalman_comp_plot[:, 0], Kalman_comp_plot[:, 1], "og", markersize=3)
                        plt.plot(GPS_comp_plot[:, 0], GPS_comp_plot[:, 1], "or", markersize=3) # 현재 위치

                        plt.xlim(Kalman_E - 5, Kalman_E + 5)
                        plt.ylim(Kalman_N - 5, Kalman_N + 5)

                        # plt.xlim(GPS_NED_E_E - 3, GPS_NED_E_E + 3)
                        # plt.ylim(GPS_NED_N_N - 3, GPS_NED_N_N + 3)

                    except:
                        pass

                    plt.grid(True)
                    plt.pause(0.00001)

def GPS_vel_Callback(data):
    GPS_fix_velocity_x.value = data.twist.twist.linear.x
    GPS_fix_velocity_y.value = data.twist.twist.linear.y
    GPS_fix_velocity_z.value = data.twist.twist.linear.z


def GNSS_Subscribe():

    rospy.init_node('GNSS_Subscribe', anonymous=True)

    rospy.Subscriber("vectornav/IMU", Imu, chatterCallback)

    rospy.Subscriber("raw_data/fix", NavSatFix, GPSCallback)

    rospy.Subscriber("raw_data/fix_velocity", TwistWithCovarianceStamped, GPS_vel_Callback)

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

    GPS_fix_velocity_x = Value('d', 0.0)
    GPS_fix_velocity_y = Value('d', 0.0)
    GPS_fix_velocity_z = Value('d', 0.0)



    processedQ = Queue()

    th3 = Process(target=vn100_Kalman_filter, args = (current_lat, current_lon, current_alt, current_accel_x,
    current_accel_y, current_accel_z,current_vel_x, current_vel_y,current_vel_z,current_yaw,current_roll,
                                                      current_pitch,GPS_CNT,kalman_yaw,GPS_CTC,
                                                      kalman_pitch, kalman_roll, kalman_lat,
                                                      kalman_lon, kalman_alt,kalman_NED_N, kalman_NED_E, kalman_NED_D,
                                                      kalman_ENU_E, kalman_ENU_N, kalman_ENU_U,
                                                      GPS_NED_N, GPS_NED_E, GPS_NED_D,
                                                      GPS_ENU_E, GPS_ENU_N, GPS_ENU_U,IMU_CNT,
                                                      GPS_fix_velocity_x,GPS_fix_velocity_y,GPS_fix_velocity_z))

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

