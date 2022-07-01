import math


def Kalman_Filter(Lat, h, Rm, Rt, Rmm, Rtt, rho_n, rho_e, rho_d, v_n, v_e, v_d, f_n, f_e, f_d, w_ien, Cbn):
    degree = math.pi / 180

    cos_lat = math.cos(Lat * degree)
    tan_lat = math.tan(Lat * degree)
    sin_lat = math.sin(Lat * degree)

    F_pp = [
        [Rmm * rho_e / (Rm + h), 0, rho_e / (Rm + h)],
        [rho_n * (tan_lat - Rtt / (Rt + h)) / cos_lat, 0, -rho_n /(cos_lat * (Rt + h)) ],
        [0, 0, 0]
    ]

    F_pv = [
        [1 / (Rm + h), 0, 0],
        [0, 1 / (cos_lat * (Rt + h)), 0],
        [0, 0, -1]
    ]

    F_vp = [
        [Rmm * rho_e * v_d / (Rm + h) - (rho_n / ((cos_lat) ** 2) + 2 * w_ien[0]) * v_e - rho_n * rho_d * Rtt, 0,
         rho_e * v_d / (Rm + h) - rho_n * rho_d],
        [(2 * w_ien[0] + rho_n / ((cos_lat) ** 2) + rho_d * Rtt / (Rt + h)) * v_n - (
                    rho_n * Rtt / (Rt + h) - 2 * w_ien[2]) * v_d, 0, rho_d * v_n / (Rt + h) - rho_n * v_d / (Rt + h)],
        [(rho_n ** 2) * Rtt + (rho_e ** 2) * Rmm - 2 * w_ien[2] * v_e, 0, rho_n ** 2 + rho_e ** 2]

    ]

    F_vv = [[v_d / (Rm + h), 2 * rho_d + 2 * w_ien[2], -rho_e],
            [-2 * w_ien[2] - rho_d, (v_n * tan_lat + v_d) / (Rt + h), 2 * w_ien[0] + rho_n],
            [2 * rho_e, -2 * w_ien[0] - 2 * rho_n, 0]
            ]

    F_vphi = [
        [0, -f_d, f_e],
        [f_d, 0, -f_n],
        [-f_e, f_n, 0]
    ]

    F_phip = [
        [w_ien[2] - rho_n * Rtt / (Rt + h), 0, -rho_n / (Rt + h)],
        [-rho_e * Rmm / (Rm + h), 0, -rho_e / (Rm + h)],
        [-w_ien[0] - rho_n / (cos_lat) ** 2 - rho_d * Rtt / (Rt + h), 0, -rho_d / (Rt + h)]
    ]

    F_phiv = [
        [0, 1 / (Rt + h), 0],
        [-1 / (Rm + h), 0, 0],
        [0, -tan_lat / (Rt + h), 0]
    ]

    F_phiphi = [
        [0, w_ien[2] + rho_d, -rho_e],
        [-w_ien[2] - rho_d, 0, w_ien[0] + rho_n],
        [rho_e, -w_ien[0] - rho_n, 0]
    ]

    Cbn_minus = (-1) * Cbn

    F = [[0 for j in range(15)] for i in range(15)]

    for i in range(3):
        for k in range(3):
            F[i][k] = F_pp[i][k]

    for i in range(3):
        for k in range(3, 6):
            F[i][k] = F_pv[i][k - 3]

    for i in range(3, 6):
        for k in range(3):
            F[i][k] = F_vp[i - 3][k]

    for i in range(3, 6):
        for k in range(3, 6):
            F[i][k] = F_vv[i - 3][k - 3]

    for i in range(3, 6):
        for k in range(6, 9):
            F[i][k] = F_vphi[i - 3][k - 6]

    for i in range(3, 6):
        for k in range(9, 12):
            F[i][k] = Cbn[i - 3][k - 9]

    for i in range(6, 9):
        for k in range(0, 3):
            F[i][k] = F_phip[i - 6][k]

    for i in range(6, 9):
        for k in range(3, 6):
            F[i][k] = F_phiv[i - 6][k - 3]

    for i in range(6, 9):
        for k in range(6, 9):
            F[i][k] = F_phiphi[i - 6][k - 6]

    for i in range(6, 9):
        for k in range(12, 15):
            F[i][k] = Cbn_minus[i - 6][k - 12]

    return F


def m_n_inverse(matrix):
    m = len(matrix)

    n = len(matrix[0])

    after = [[0 for i in range(m)] for k in range(n)]

    for i in range(m):
        for k in range(n):
            after[k][i] = matrix[i][k]

    return after


def n_n_inverse(matrix):
    before = matrix
    after = [[0 for i in range(15)] for k in range(15)]

    for i in range(len(before)):
        for k in range(len(before[0])):
            after[k][i] = before[i][k]

    return after


def DCM2eul_bn(matrix):

    roll = math.atan2(matrix[2][1], matrix[2][2]) * 180 /math.pi #* math.pi / 180 이거 아니다.... 개고생함 ㅅㅂ
    #pitch = math.atan2(- matrix[2][0],(1 - (matrix[2][0] ** 2)) ** ( 1 / 2 )) * 180 / math.pi
    pitch = math.asin(-matrix[2][0]) * 180 / math.pi  # * math.pi / 180
    yaw = math.atan2(matrix[1][0], matrix[0][0]) * 180 /math.pi

    return roll, pitch, yaw

def eul2DCM_bn(roll, pitch, yaw): # C ( b to n)
    degree = math.pi / 180
    matrix = [
        [math.cos(pitch * degree) * math.cos(yaw * degree),
         math.cos(yaw * degree) * math.sin(pitch * degree) * math.sin(roll * degree) -
         math.cos(roll * degree) * math.sin(yaw * degree),
         math.cos(roll * degree) * math.cos(yaw * degree) * math.sin(pitch * degree) +
         math.sin(roll * degree) * math.sin(yaw * degree)
         ],
        [math.cos(pitch * degree) * math.sin(yaw * degree),
         math.cos(roll * degree) * math.cos(yaw * degree) +
         math.sin(pitch * degree) * math.sin(roll * degree) * math.sin(yaw * degree),
         math.cos(roll * degree) * math.sin(pitch * degree) * math.sin(yaw * degree) -
         math.cos(yaw * degree) * math.sin(roll * degree)
         ],
        [-math.sin(pitch * degree),
         math.cos(pitch * degree) * math.sin(roll * degree),
         math.cos(pitch * degree) * math.cos(roll * degree)
         ]

    ]
    return matrix


def product_matrix_dx(A, B):
    if len(B) == 15:

        matrix = [0 for i in range(len(A))]
        for i in range(len(A)):

            for j in range(len(A[0])):

                for k in range(len(B)):
                    matrix[i] = A[i][j] * B[k]

    return matrix


def product_matrix(A, B):
    if len(A[0]) == len(B):  # 열과 행이 같아 연산 가능

        matrix = [[0 for j in range(len(B[0]))] for i in range(len(A))]

        for i in range(len(A)):

            for j in range(len(A[0])):

                for k in range(len(B)):

                    for t in range(len(B[0])):
                        matrix[i][t] = A[i][j] * B[k][t]

    else:
        pass

    return matrix


def skew(x, y, z):
    M = [[0, -z, y],
         [z, 0, -x],
         [-y, x, 0]]

    return M


def plus_matrix(A, B):
    matrix = [[0 for j in range(len(B[0]))] for i in range(len(A))]

    if len(A) == len(B):
        if len(A[0]) == len(B[0]):

            for i in range(len(A)):

                for j in range(len(A[0])):

                    for k in range(len(B)):

                        for t in range(len(B[0])):
                            matrix[i][t] = A[i][j] + B[k][t]

    else:
        pass

    return matrix


def minus_matrix(A, B):
    matrix = [[0 for j in range(len(B[0]))] for i in range(len(A))]

    if len(A) == len(B):
        if len(A[0]) == len(B[0]):

            for i in range(len(A)):

                for j in range(len(A[0])):

                    for k in range(len(B)):

                        for t in range(len(B[0])):
                            matrix[i][t] = A[i][j] - B[k][t]

    else:
        pass

    return matrix



def cross_product(A, B):
    matrix = [[A[1][0] * B[2][0] - A[2][0] * B[1][0]],
              [A[2][0] * B[0][0] - A[0][0] * B[2][0]],
              [A[0][0] * B[1][0] - A[1][0] * B[0][0]]]

    return matrix


def n_1_inverse(A):
    matrix = []

    for i in range(len(A)):
        matrix.append([A[i]])

    return matrix