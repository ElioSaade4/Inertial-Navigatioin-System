import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from INS import INS
from utils import plotMAT, llaToNEU, plotsAppendixA, plotHeading


if __name__ == "__main__":

    #TODO: Create classes: IMU, GPS

    data = loadmat( 'data.mat' )

    N = 10_000  # max is 10000

    # True Trajectory
    trajAcc = data[ 'trajAcc' ][:N, :]                # True trajectory acceleration: a_N, a_E, a_U (N x 3) (NEU)
    trajAngVel = data[ 'trajAngVel' ][:N, :]           # True trajectory angular velocity: p, q, r (N x 3) 
    trajOrientDeg = data[ 'trajOrientDeg' ][:N, :]     # True trajectory orientation: roll, pitch, yaw (N x 3)
    trajPos = data[ 'trajPos' ][:N, :]                 # True trajectory position: x_N, x_E, x_U (N x 3)
    trajVel = data[ 'trajVel' ][:N, :]                 # True trajectory velocity: v_N, v_E, v_U (N x 3)


    # Sensor Measurements
    accel = data[ 'accel' ][:N, :]        # Accelerometer data: ax, ay, az (N x 3) (XYZ)
    gyro = data[ 'gyro' ][:N, :]          # Gyroscope data: p, q, r (N x 3) (XYZ)
    mag = data[ 'mag' ][:N, :]            # Magnetometer data: mx, my, mz (N x 3) (XYZ)
    v_gps = data[ 'gpsvel' ][:N, :]       # GPS velocity: v_N, v_E, v_U (N x 3) (NEU)
    lla = data[ 'lla' ][:N, :]            # GPS position: latitude, longitude, altitude (N x 3) (NEU)

    Ts = 0.01                     # Sampling time (s)
    time = np.arange( 0, accel.shape[ 0 ] * Ts, Ts )     # Time vector (s)

    # Add bias to accelerometer measurements
    # accel[:, 0] += 0.02  # Add bias to ax
    # accel[:, 1] += -0.02  # Add bias to ay

    # Data preprocessing
    # 1- Shift the true trajectory so that it starts at the origin (0,0,0)
    trajPos -= trajPos[ 0, : ]  # Shift trajectory position to start at origin

    # 2- Fix ax, ay direction in body frame (positive ax is forward, positive ay is left) because the data uses opposite
    accel[:, 0] *= -1  # Flip ax direction
    accel[:, 1] *= -1  # Flip ay direction

    # 2- Remove yaw rate bias
    # gyro[:,2] -= 0.122

    # 3- Convert GPS position from LLA to NEU
    gps_pos = llaToNEU( lla )

    # plotsAppendixA( time, accel, gyro, mag, v_gps, lla, trajAcc, trajAngVel, trajOrientDeg, trajPos, trajVel )

    # Allocate memory for results 
    # 8 states: heading angle, gyro bias, xN, vN, aN, xE, vE, aE
    est_states = np.zeros( ( N, 8 ) )
    
    # Initialize INS
    ins = INS( Ts, mag[ 0, : ] )
    est_states[ 0, : ] = np.hstack( ( ins.s1.flatten(), ins.s2.flatten(), ins.s3.flatten() ) )

    for k in range( 0, N-1 ):
        s1 = ins.HeadingUpdateKF( gyro[ k, : ], mag[ k, : ] )
        s2, s3 = ins.IMUUpdateKF( accel[ k, : ] )
        est_states[ k+1, : ] = np.hstack( ( s1.flatten(), s2.flatten(), s3.flatten() ) )

    plotMAT( time, accel, gyro, mag, gps_pos, trajAcc, trajVel, trajPos, trajAngVel, trajOrientDeg, est_states )

    plotHeading( time, gyro, mag, trajOrientDeg, trajAngVel, est_states )


    

    # plt.show()