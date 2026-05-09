import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path

from INS import INS
from utils import llaToNEU, plotsAppendixA, plotEstimation, CalculateRMSE, plotAblation


if __name__ == "__main__":

    # Create directories for plots
    base_dir = Path("plots")
    subdirs = ["appendixA", "INS", "INS-NoGPS"]

    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    data = loadmat( 'data.mat' )

    N = 10_000  # max is 10000

    # True Trajectory
    traj_accel = data[ 'trajAcc' ][:N, :]                # True trajectory acceleration: a_N, a_E, a_U (N x 3) (NEU)
    traj_ang_vel = data[ 'trajAngVel' ][:N, :]           # True trajectory angular velocity: p, q, r (N x 3) 
    traj_orient = data[ 'trajOrientDeg' ][:N, :]     # True trajectory orientation: roll, pitch, yaw (N x 3)
    traj_pos = data[ 'trajPos' ][:N, :]                 # True trajectory position: x_N, x_E, x_U (N x 3)
    traj_vel = data[ 'trajVel' ][:N, :]                 # True trajectory velocity: v_N, v_E, v_U (N x 3)


    # Sensor Measurements
    accel = data[ 'accel' ][:N, :]        # Accelerometer data: ax, ay, az (N x 3) (XYZ)
    gyro = data[ 'gyro' ][:N, :]          # Gyroscope data: p, q, r (N x 3) (XYZ)
    mag = data[ 'mag' ][:N, :]            # Magnetometer data: mx, my, mz (N x 3) (XYZ)
    gps_v = data[ 'gpsvel' ][:N, :]       # GPS velocity: v_N, v_E, v_U (N x 3) (NEU)
    lla = data[ 'lla' ][:N, :]            # GPS position: latitude, longitude, altitude (N x 3) (NEU)

    Ts = 0.01                     # Sampling time (s)
    time = np.arange( 0, accel.shape[ 0 ] * Ts, Ts )     # Time vector (s)

    # Data preprocessing
    # 1- Shift the true trajectory so that it starts at the origin (0,0,0)
    traj_pos -= traj_pos[ 0, : ]  # Shift trajectory position to start at origin

    # 2- Fix ax, ay direction in body frame (positive ax is forward, positive ay is left) because the data uses opposite
    accel[:, 0] *= -1  # Flip ax direction
    accel[:, 1] *= -1  # Flip ay direction

    # 3- Convert GPS position from LLA to NEU
    gps_pos = llaToNEU( lla )

    plotsAppendixA( time, accel, gyro, mag, gps_v, lla, traj_pos, traj_vel, traj_ang_vel, traj_orient )

    # Allocate memory for results 
    # 8 states: heading angle, gyro bias, xN, vN, aN, xE, vE, aE
    est_states = np.zeros( ( N, 8 ) )
    
    # Initialize INS
    ins = INS( Ts, mag[ 0, : ] )
    est_states[ 0, : ] = np.hstack( ( ins.s1.flatten(), ins.s2.flatten(), ins.s3.flatten() ) )

    # Asynchronous state estimation loop
    # IMU rate: 100 Hz
    # GPS rate: 5 Hz
    for k in range( 0, N-1 ):       # Loop rate: 100 Hz (IMU rate)
        # State prediction for all states
        ins.StatePrediction( gyro[ k, : ] )

        # When gyro and magnetometer are available, update heading and bias estimates
        ins.HeadingUpdateKF( gyro[ k, : ], mag[ k, : ] )

        # When accelerometer is available, update position, velocity, and acceleration estimates
        ins.IMUUpdateKF( accel[ k, : ] )

        # when GPS is available, update position and velocity estimates
        if k % 20 == 0:  # GPS update every 10 IMU samples (10 Hz)
            ins.GPSUpdateKF( gps_pos[ k, : ], gps_v[ k, : ] )

        # Store estimates
        est_states[ k+1, : ] = np.hstack( ( ins.s1.flatten(), ins.s2.flatten(), ins.s3.flatten() ) )


    print( "RMSE for full INS with GPS updates" )
    print( "-----------------------------" )
    CalculateRMSE( traj_pos, traj_vel, traj_accel, traj_orient, est_states )
    plotEstimation( time, gyro, gps_pos, gps_v, traj_pos, traj_vel, traj_accel, traj_orient, est_states )
    

    # Ablation Study: Dead-Reckoning only (no GPS updates)
    est_states2 = np.zeros( ( N, 8 ) )
    
    # Initialize INS
    ins2 = INS( Ts, mag[ 0, : ] )
    est_states2[ 0, : ] = np.hstack( ( ins2.s1.flatten(), ins2.s2.flatten(), ins2.s3.flatten() ) )

    for k in range( 0, N-1 ):       # Loop rate: 100 Hz (IMU rate)
        # State prediction for all states
        ins2.StatePrediction( gyro[ k, : ] )

        # When gyro and magnetometer are available, update heading and bias estimates
        ins2.HeadingUpdateKF( gyro[ k, : ], mag[ k, : ] )

        # When accelerometer is available, update position, velocity, and acceleration estimates
        ins2.IMUUpdateKF( accel[ k, : ] )

        # Store estimates
        est_states2[ k+1, : ] = np.hstack( ( ins2.s1.flatten(), ins2.s2.flatten(), ins2.s3.flatten() ) )


    print( "\n\nRMSE for INS without GPS updates" )
    print( "-----------------------------" )
    CalculateRMSE( traj_pos, traj_vel, traj_accel, traj_orient, est_states2 )
    plotAblation( time, traj_pos, traj_vel, est_states, est_states2 )

    # plt.show()