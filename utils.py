import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pymap3d as pm


def GPSProcessing( v_gps: np.ndarray[ float ], lon: np.ndarray[ float ], lat: np.ndarray[ float ] ) -> np.ndarray[ float ]:
    """
    Removes large outliers in GPS velocity data by replacing them by the previous value

    Args:
        v_gps ( np.ndarray[ float ] ): array of GPS velocity (m/s)
        lon ( np.ndarray[ float ] ): array of GPS longitude (degrees)
        lat ( np.ndarray[ float ] ): array of GPS latitude (degrees)

    Returns:
        v_gps_proc ( np.ndarray[ float ] ): array of processed GPS velocity (m/s)
    """
    # TODO: Can make it as a linear interpolation of the previous and next valid values instead of just the previous value

    # Remove large outliers from GPS data
    v_gps_proc = np.zeros_like( v_gps )
    lon_proc = np.zeros_like( lon )
    lat_proc = np.zeros_like( lat )

    for i in range( v_gps.size ):
        if( v_gps[i] > 50 ):
            v_gps_proc[ i ] = v_gps_proc[ i - 1 ]
        else:
            v_gps_proc[ i ] = v_gps[ i ]

        if lon[ i ] < 315 or lon[ i ] > 318:
            lon_proc[ i ] = lon_proc[ i - 1 ]
        else:
            lon_proc[ i ] = lon[ i ]

        if lat[ i ] > 40 or lat[ i ] < 30:
            lat_proc[ i ] = lat_proc[ i - 1 ]
        else:
            lat_proc[ i ] = lat[ i ]
    
    return v_gps_proc, lon_proc, lat_proc


def gpsToXY( lon, lat ):

    # Earth radius in meters (this is the mean radius, but it can vary based on location)
    R = 6_371_000 

    # Convert longitude and latitude from degrees to radians
    lon = np.radians( lon )
    lat = np.radians( lat )

    lon0 = lon[ 0 ] 
    lat0 = lat[ 0 ]

    x = R * ( lon - lon0 ) * np.cos( lat0 )
    y = R * ( lat - lat0 )

    return x, y


def llaToNEU( lla ):
    # lla is Nx3: [lat, lon, alt] in degrees, meters
    lat = lla[:, 0]
    lon = lla[:, 1]
    alt = lla[:, 2]

    # reference point (first sample or given)
    lat0 = lat[0]
    lon0 = lon[0]
    alt0 = alt[0]

    # convert to NED
    n, e, d = pm.geodetic2ned( lat, lon, alt, lat0, lon0, alt0 )

    return np.column_stack( (n, e, -d) )  # return as NEU


def plotHeading( time, gyro, mag, trajOrientDeg, trajAngVel, est_states ):
    # Plot limits
    t_min = time[ 0 ]
    t_max = time[ -1 ]

    fig, ax = plt.subplots( figsize=(12, 6.75))

    mag_heading = np.degrees(np.arctan2(-mag[:, 1], mag[:, 0])) - 13.5  # both in degrees now
    mag_heading = (mag_heading + 180) % 360 - 180

    angle_0 = trajOrientDeg[ 0, 0 ]  # initial heading angle from true trajectory
    yaw_int = np.cumsum( gyro[:, 2] ) * 0.01  # integrate yaw rate to get yaw angle (in radians)
    yaw_int = ( np.degrees( yaw_int ) + angle_0 + 180 ) % 360 - 180  # convert to degrees

    ins_heading = ( np.degrees( est_states[:, 0] )  + 180 ) % 360 - 180  # INS heading in degrees

    ax.plot( time, trajOrientDeg[:, 0], color='C0', label='True' )
    ax.plot( time, mag_heading, color='C1', label='Mag' )
    ax.plot( time, yaw_int, color='C2', label='IMU' )
    ax.plot( time, ins_heading, color='C3', label='INS' )

    ax.set_xlabel( 'Time (s)' )
    ax.set_ylabel( 'Heading (deg)' )
    ax.set_xlim( t_min, t_max )
    ax.grid()
    ax.legend()
    fig.savefig( 'plots/heading_estimation.png', bbox_inches='tight' )

    # fig1, ax1 = plt.subplots( figsize=(12, 6.75))
    # ax1.plot( time, trajAngVel[:, 2], label='True' )
    # ax1.plot( time, gyro[:, 2], label='Gyro' )
    # ax1.set_xlabel( 'Time (s)' )
    # ax1.set_ylabel( 'Yaw Rate (rad/s)' )
    # ax1.set_xlim( t_min, t_max )
    # ax1.grid()
    # ax1.legend()

    fig2, ax2 = plt.subplots( figsize=(12, 6.75))
    ax2.axhline( y=0.122, color='C0', label='True')
    ax2.plot( time, est_states[:, 1], color='C3', label='INS' )
    ax2.set_xlabel( 'Time (s)' )
    ax2.set_ylabel( 'Gyro Bias (rad/s)' )
    ax2.set_xlim( t_min, t_max )
    ax2.grid()
    ax2.legend()
    fig2.savefig( 'plots/gyro_bias_estimation.png', bbox_inches='tight' )


def plotsAppendixA( time, accel, gyro, mag, v_gps, lla, trajAcc, trajAngVel, trajOrientDeg, trajPos, trajVel ):
    """Generates plots for Appendix A of the report

    Args:
        time (_type_): _description_
        accel (_type_): _description_
        gyro (_type_): _description_
        mag (_type_): _description_
        v_gps (_type_): _description_
        lla (_type_): _description_
        trajAcc (_type_): _description_
        trajAngVel (_type_): _description_
        trajOrientDeg (_type_): _description_
        trajPos (_type_): _description_
        trajVel (_type_): _description_
    """

    # Plot limits
    t_min = time[ 0 ]
    t_max = time[ -1 ]

    # 1- IMU measurements
    _, ax1 = plt.subplots( 5, 1, sharex=True, figsize=(6, 10) )
    ax1.flatten()
    ax1[ 0 ].plot( time, accel[:, 0], label='ax' )
    ax1[ 0 ].set_ylabel( 'ax (m/s^2)' )
    ax1[ 0 ].set_xlim( t_min, t_max )
    ax1[ 0 ].grid()

    ax1[ 1 ].plot( time, accel[:, 1], label='ay' )
    # ax1[ 1 ].set_xlabel( 'Time (s)' )
    ax1[ 1 ].set_ylabel( 'ay (m/s^2)' )
    ax1[ 1 ].set_xlim( t_min, t_max )
    ax1[ 1 ].grid()

    ax1[ 2 ].plot( time, gyro[:, 2], label='yaw rate' )
    # ax1[ 2 ].set_xlabel( 'Time (s)' )
    ax1[ 2 ].set_ylabel( 'r (rad/s)' )
    ax1[ 2 ].set_xlim( t_min, t_max )
    ax1[ 2 ].grid()

    ax1[ 3 ].plot( time, mag[:, 0], label='mx' )
    ax1[ 3 ].set_ylabel( 'mx (uT)' )
    ax1[ 3 ].set_xlim( t_min, t_max )
    ax1[ 3 ].grid()

    ax1[ 4 ].plot( time, mag[:, 1], label='my' )
    ax1[ 4 ].set_ylabel( 'my (uT)' )
    ax1[ 4 ].set_xlim( t_min, t_max )
    ax1[ 4 ].grid()
    ax1[ 4 ].set_xlabel( 'Time (s)' )

    plt.savefig( 'plots/dataset/IMU_measurements.png', bbox_inches='tight' )

    # 2- GPS Measurements
    _, ax2 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax2.flatten()
    ax2[ 0 ].plot( time, np.sqrt( v_gps[:, 0]**2 + v_gps[:, 1]**2 ), label='v_gps' )
    ax2[ 0 ].set_ylabel( 'Ground Speed (m/s)' )
    ax2[ 0 ].set_xlim( t_min, t_max )
    ax2[ 0 ].grid()

    ax2[ 1 ].plot( time, lla[:, 0], label='Latitude' )
    ax2[ 1 ].set_ylabel( 'Latitude (deg)' )
    ax2[ 1 ].set_xlim( t_min, t_max )
    ax2[ 1 ].grid()

    ax2[ 2 ].plot( time, lla[:, 1], label='Longitude' )
    ax2[ 2 ].set_ylabel( 'Longitude (deg)' )
    ax2[ 2 ].set_xlim( t_min, t_max )
    ax2[ 2 ].grid()
    ax2[ 2 ].set_xlabel( 'Time (s)' )

    plt.savefig( 'plots/dataset/GPS_measurements.png', bbox_inches='tight' )

    # 3- True trajectory
    # Trajectory
    x = trajPos[:, 0]
    y = trajPos[:, 1]
    _, ax3 = plt.subplots()
    ax3.plot(x[:-1], y[:-1], '--', linewidth=1.5, label='True')
    ax3.plot(x[-1], y[-1], marker='D', markersize=8)
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.set_xlabel( 'North (m)' )
    ax3.set_ylabel( 'East (m)' )
    plt.savefig( 'plots/dataset/true_trajectory.png', bbox_inches='tight' )

    # 4- Some true states
    _, ax4 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax4.flatten()
    ax4[ 0 ].plot( time, np.sqrt( trajVel[:, 0]**2 + trajVel[:, 1]**2 ), label='True Speed' )
    ax4[ 0 ].set_ylabel( 'Speed (m/s)' )
    ax4[ 0 ].set_xlim( t_min, t_max )
    ax4[ 0 ].grid()
    ax4[ 1 ].plot( time, np.radians(trajOrientDeg[:, 0]), label='Heading angle' )
    ax4[ 1 ].set_ylabel( 'Heading (deg)' )
    ax4[ 1 ].set_xlim( t_min, t_max )
    ax4[ 1 ].grid()
    ax4[ 2 ].plot( time, trajAngVel[:, 2], label='Yaw rate' )
    ax4[ 2 ].set_ylabel( 'Yaw Rate (rad/s)' )
    ax4[ 2 ].set_xlim( t_min, t_max )
    ax4[ 2 ].grid()
    ax4[ 2 ].set_xlabel( 'Time (s)' )
    plt.savefig( 'plots/dataset/true_states.png', bbox_inches='tight' )

    # 5- True velocities
    _, ax5 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax5.flatten()
    ax5[ 0 ].plot( time, trajVel[:, 0], label='v_N' )
    ax5[ 0 ].set_ylabel( 'North Speed (m/s)' )
    ax5[ 0 ].set_xlim( t_min, t_max )
    ax5[ 0 ].grid()

    ax5[ 1 ].plot( time, trajVel[:, 1], label='v_E' )
    ax5[ 1 ].set_ylabel( 'East Speed (m/s)' )
    ax5[ 1 ].set_xlim( t_min, t_max )
    ax5[ 1 ].grid()

    ax5[ 2 ].plot( time, trajVel[:, 2], label='v_D' )
    ax5[ 2 ].set_ylabel( 'Up Speed (m/s)' )
    ax5[ 2 ].set_xlim( t_min, t_max )
    ax5[ 2 ].grid()
    ax5[ 2 ].set_xlabel( 'Time (s)' )
    plt.savefig( 'plots/dataset/true_velocities.png', bbox_inches='tight' )

    # 6- GPS velocities
    _, ax6 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax6.flatten()
    ax6[ 0 ].plot( time, v_gps[:, 0], label='v_N' )
    ax6[ 0 ].set_ylabel( 'North Speed (m/s)' )
    ax6[ 0 ].set_xlim( t_min, t_max )
    ax6[ 0 ].grid()

    ax6[ 1 ].plot( time, v_gps[:, 1], label='v_E' )
    ax6[ 1 ].set_ylabel( 'East Speed (m/s)' )
    ax6[ 1 ].set_xlim( t_min, t_max )
    ax6[ 1 ].grid()

    ax6[ 2 ].plot( time, v_gps[:, 2], label='v_D' )
    ax6[ 2 ].set_ylabel( 'Up Speed (m/s)' )
    ax6[ 2 ].set_xlim( t_min, t_max )
    ax6[ 2 ].grid()
    ax6[ 2 ].set_xlabel( 'Time (s)' )
    plt.savefig( 'plots/dataset/gps_velocities.png', bbox_inches='tight' )

    # 7- GPS LLA
    _, ax7 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax7.flatten()
    ax7[ 0 ].plot( time, lla[:, 0], label='Latitude' )
    ax7[ 0 ].set_ylabel( 'Latitude (deg)' )
    ax7[ 0 ].set_xlim( t_min, t_max )
    ax7[ 0 ].grid()
    ax7[ 1 ].plot( time, lla[:, 1], label='Longitude' )
    ax7[ 1 ].set_ylabel( 'Longitude (deg)' )
    ax7[ 1 ].set_xlim( t_min, t_max )
    ax7[ 1 ].grid()
    ax7[ 2 ].plot( time, lla[:, 2], label='Altitude' )
    ax7[ 2 ].set_ylabel( 'Altitude (m)' )
    ax7[ 2 ].set_xlim( t_min, t_max )
    ax7[ 2 ].grid()
    ax7[ 2 ].set_xlabel( 'Time (s)' )
    plt.savefig( 'plots/dataset/gps_lla.png', bbox_inches='tight' )


def plotMAT( time, accel, gyro, mag, gps_pos, trajAcc, trajVel, trajPos, trajAngVel, trajOrientDeg, est_states ):
    # Plot limits
    t_min = time[ 0 ]
    t_max = time[ -1 ]

    # Just checking data consistency and accelerometer bias
    a_N = trajAcc[:, 0]
    a_E = trajAcc[:, 1]

    traj_ax = a_N * np.cos( np.radians( trajOrientDeg[:, 0] ) ) + a_E * np.sin( np.radians( trajOrientDeg[:, 0] ) )
    traj_ay = -a_N * np.sin( np.radians( trajOrientDeg[:, 0] ) ) + a_E * np.cos( np.radians( trajOrientDeg[:, 0] ) )

    ins_ax = est_states[:, 4] * np.cos( est_states[:, 0] ) + est_states[:, 7] * np.sin( est_states[:, 0] )
    ins_ay = -est_states[:, 4] * np.sin( est_states[:, 0] ) + est_states[:, 7] * np.cos( est_states[:, 0] )

    meas_a_N = accel[:, 0] * np.cos( np.radians( trajOrientDeg[:, 0] ) ) - accel[:, 1] * np.sin( np.radians( trajOrientDeg[:, 0] ) )
    meas_a_E = accel[:, 0] * np.sin( np.radians( trajOrientDeg[:, 0] ) ) + accel[:, 1] * np.cos( np.radians( trajOrientDeg[:, 0] ) )

    # # IMU ax
    # _, ax1 = plt.subplots()
    # ax1.plot( time, traj_ax, label="Calculated from true a_N and a_E" )
    # ax1.plot( time, accel[:, 0], label="IMU" )
    # ax1.plot( time, ins_ax, label="INS" )
    
    # ax1.set_xlabel( 'Time (s)' )
    # ax1.set_ylabel( 'Measured Ax (m/s^2)' )
    # ax1.set_title( 'Ax in Body Frame(m/s^2)' )
    # ax1.set_xlim( t_min, t_max )
    # ax1.grid()
    # ax1.legend()
    # plt.savefig( 'plots/ax_body.png' )

    # # IMU ay
    # _, ax2 = plt.subplots()
    # ax2.plot( time, accel[:, 1], label="IMU" )
    # ax2.plot( time, traj_ay, label="Calculated from true a_N and a_E" )
    # ax2.set_xlabel( 'Time (s)' )
    # ax2.set_ylabel( 'Measured Ay (m/s^2)' )
    # ax2.set_title( 'Ay in Body Frame (m/s^2)' )
    # ax2.set_xlim( t_min, t_max )
    # ax2.grid()
    # ax2.legend()
    # plt.savefig( 'plots/ay_body.png' )



    # Trajectory
    x = trajPos[:, 0]
    y = trajPos[:, 1]
    fig5, ax5 = plt.subplots()
    # trajectory (all but last point)
    ax5.plot( x[:-1], y[:-1], color="C0", linestyle='--', linewidth=1.5, label='True')
    ax5.plot( est_states[:, 2], est_states[:, 5], color="C3", label='INS' )
    # current position (last point)
    ax5.plot( x[-1], y[-1], marker='D', markersize=8 )
    # ax5.scatter( gps_pos[:, 0], gps_pos[:, 1], c='red', s=10, label='GPS' )
    ax5.set_aspect('equal')
    ax5.grid(True)
    ax5.set_xlabel( 'North (meters)' )
    ax5.set_ylabel( 'East (meters)' )
    ax5.legend()
    fig5.savefig( 'plots/trajectory.png', bbox_inches='tight' )

    # a_N in NEU frame
    fig6, ax6 = plt.subplots()
    ax6.plot( time, trajAcc[:, 0], color="C0", label='True' )
    ax6.plot( time, meas_a_N, color="C1", label='Calculated from IMU' )
    ax6.plot( time, est_states[:, 4], color="C3", label='INS' )
    ax6.set_xlabel( 'Time (s)' )
    ax6.set_ylabel( 'Trajectory Acceleration (m/s^2)' )
    ax6.set_title( 'A_N in NEU frame' )
    ax6.set_xlim( t_min, t_max )
    ax6.grid()
    ax6.legend()
    fig6.savefig( 'plots/aN_neu.png', bbox_inches='tight' )

    # a_E in NEU frame
    fig7, ax7 = plt.subplots()
    ax7.plot( time, trajAcc[:, 1], color="C0", label='True' )
    ax7.plot( time, meas_a_E, color="C1", label='Calculated from IMU' )
    ax7.plot( time, est_states[:, 7], color="C3", label='INS' )
    ax7.set_xlabel( 'Time (s)' )
    ax7.set_ylabel( 'Trajectory Acceleration (m/s^2)' )
    ax7.set_title( 'A_E in NEU frame' )
    ax7.set_xlim( t_min, t_max )
    ax7.grid()
    ax7.legend()
    fig7.savefig( 'plots/aE_neu.png', bbox_inches='tight' )

    # v_N in NEU frame
    fig8, ax8 = plt.subplots()
    ax8.plot( time, trajVel[:, 0], color="C0", label='True' )
    ax8.plot( time, est_states[:, 3], color="C3", label='INS' )
    ax8.set_xlabel( 'Time (s)' )
    ax8.set_ylabel( 'Trajectory Velocity (m/s)' )
    ax8.set_title( 'V_N in NEU frame' )
    ax8.set_xlim( t_min, t_max )
    ax8.grid()
    ax8.legend()
    fig8.savefig( 'plots/vN_neu.png', bbox_inches='tight' )

    # v_E in NEU frame
    fig9, ax9 = plt.subplots()
    ax9.plot( time, trajVel[:, 1], color="C0", label='True' )
    ax9.plot( time, est_states[:, 6], color="C3", label='INS' )
    ax9.set_xlabel( 'Time (s)' )
    ax9.set_ylabel( 'Trajectory Velocity (m/s)' )
    ax9.set_title( 'V_E in NEU frame' )
    ax9.set_xlim( t_min, t_max )
    ax9.grid()
    ax9.legend()
    fig9.savefig( 'plots/vE_neu.png', bbox_inches='tight' )

    # Ground velocity
    fig10, ax10 = plt.subplots()
    ax10.plot( time, np.sqrt( trajVel[:, 0]**2 + trajVel[:, 1]**2 ), color="C0", label='True' )
    ax10.plot( time, np.sqrt( est_states[:, 3]**2 + est_states[:, 6]**2 ), color="C3", label='INS' )
    ax10.set_xlabel( 'Time (s)' )
    ax10.set_ylabel( 'Ground Speed (m/s)' )
    ax10.set_title( 'Ground Speed' )
    ax10.set_xlim( t_min, t_max )
    ax10.grid()
    fig10.savefig( 'plots/ground_speed.png', bbox_inches='tight' )




def dataPlotting( time, ax_m, ay_m, az_m, r_m, v_gps_m, lon_m, lat_m, gps_x, gps_y, x_EKF ):
    """
    Generates all the plots

    Args:
        time (np.array): time vector
        ax_m (np.array): measured IMU longitudinal acceleration (mG)
        ay_m (np.array): measured IMU lateral acceleration (mG)
        az_m (np.array): measured IMU gravity (mG)
        r_m (np.array): measured IMU yaw rate (dps)
        v_gps_m (np.array): measured GPS velocity (m/s)
        lon_m (np.array): measured GPS longitude (degrees)
        lat_m (np.array): measured GPS latitude (degrees)
    """
    t_min = time[ 0 ]
    t_max = time[ -1 ]     # there is a stationary phase in the data at the end (no need to plot it)

    # Measured IMU longitudinal acceleration (ax)
    _, ax1 = plt.subplots()
    ax1.plot( time, ax_m )
    ax1.set_xlabel( 'Time (s)' )
    ax1.set_ylabel( 'Measured Ax (m/s^2)' )
    ax1.set_title( 'Measured Longitudinal Acceleration Ax (m/s^2)' )
    ax1.set_xlim( t_min, t_max )
    ax1.grid()
    plt.savefig( 'plots/measured_ax.png' )

    # Measured IMU lateral acceleration (ay)
    _, ax2 = plt.subplots()
    ax2.plot( time, ay_m )
    ax2.set_xlabel( 'Time (s)' )
    ax2.set_ylabel( 'Measured Ay (m/s^2)' )
    ax2.set_title( 'Measured Lateral Acceleration Ay (m/s^2)' )
    ax2.set_xlim( t_min, t_max )
    ax2.grid()
    plt.savefig( 'plots/measured_ay.png' )

    # Measured IMU gravity (az)
    _, ax3 = plt.subplots()
    ax3.plot( time, az_m )
    ax3.set_xlabel( 'Time (s)' )
    ax3.set_ylabel( 'Measured Az (m/s^2)' )
    ax3.set_title( 'Measured Gravity Az (m/s^2)' )
    ax3.set_xlim( t_min, t_max )
    ax3.grid()
    plt.savefig( 'plots/measured_az.png' )

    # Measured IMU yaw rate (r_m)
    _, ax4 = plt.subplots()
    ax4.plot( time, r_m )
    ax4.set_xlabel( 'Time (s)' )
    ax4.set_ylabel( 'Measured Yaw Rate (dps)' )
    ax4.set_title( 'Measured Yaw Rate (dps)' )
    ax4.set_xlim( t_min, t_max )
    ax4.grid()
    plt.savefig( 'plots/measured_yaw_rate.png' )

    # Measured GPS velocity
    _, ax5 = plt.subplots()
    ax5.plot( time, v_gps_m, label='GPS' )
    ax5.plot( time, np.sqrt( x_EKF[2, :]**2 + x_EKF[3, :]**2 ), label='EKF' )
    ax5.set_xlabel( 'Time (s)' )
    ax5.set_ylabel( 'Speed (m/s)' )
    ax5.set_title( 'Speed vs. Time' )
    ax5.set_xlim( t_min, t_max )
    ax5.grid()
    ax5.legend()
    plt.savefig( 'plots/estimated_velocity.png' )

    # Measured GPS longitude
    _, ax6 = plt.subplots()
    ax6.plot( time, lon_m )
    ax6.set_xlabel( 'Time (s)' )
    ax6.set_ylabel( 'Measured GPS longitude (degrees)' )
    ax6.set_title( 'Measured GPS longitude (degrees)' )
    ax6.set_xlim( t_min, t_max )
    ax6.grid()
    plt.savefig( 'plots/measured_gps_longitude.png' )

    # Measured GPS latitude
    _, ax7 = plt.subplots()
    ax7.plot( time, lat_m )
    ax7.set_xlabel( 'Time (s)' )
    ax7.set_ylabel( 'Measured GPS latitude (degrees)' )
    ax7.set_title( 'Measured GPS latitude (degrees)' )
    ax7.set_xlim( t_min, t_max )
    ax7.grid()
    plt.savefig( 'plots/measured_gps_latitude.png' )

    # Plot GPS trajectory
    fig8, ax8 = plt.subplots()
    t = np.arange(len(gps_x))
    sc = ax8.scatter( gps_x, gps_y, c=t, cmap='viridis', s=2 )
    fig8.colorbar(sc, ax=ax8, label='Time / Index')
    ax8.set_xlabel( 'x (meters)' )
    ax8.set_ylabel( 'y (meters)' )
    ax8.set_title( 'GPS Trajectory (x,y)' )
    ax8.grid()
    ax8.axis('equal')  # Set equal aspect ratio for x and y axes
    plt.savefig( 'plots/gps_trajectory_scatter.png' )

    fig9, ax9 = plt.subplots()
    # Create segments
    points = np.array([gps_x, gps_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Time/index for coloring
    t = np.arange(len(gps_x))

    # Create colored line
    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(t)

    ax9.add_collection(lc)

    # Important: rescale axes
    ax9.autoscale()
    ax9.set_aspect('equal')
    ax9.axis('equal')  # Set equal aspect ratio for x and y axes

    # Colorbar
    cbar = fig9.colorbar(lc, ax=ax9)
    cbar.set_label('Time / Index')

    ax9.set_xlabel('x (meters)')
    ax9.set_ylabel('y (meters)')
    ax9.set_title('GPS Trajectory (Gradient Progression)')
    ax9.grid()
    plt.savefig( 'plots/gps_trajectory_line.png' )

    # x GPS vs time
    _, ax10 = plt.subplots()
    ax10.plot( time, gps_x, label='GPS' )
    ax10.plot( time, x_EKF[0, :], label='EKF' )
    ax10.set_xlabel( 'Time (s)' )
    ax10.set_ylabel( 'x (meters)' )
    ax10.set_title( 'x position vs Time' )
    ax10.set_xlim( t_min, t_max )
    ax10.grid()
    ax10.legend()
    plt.savefig( 'plots/estimated_x_position.png' )

    # y GPS vs time
    _, ax11 = plt.subplots()
    ax11.plot( time, gps_y, label='GPS' )
    ax11.plot( time, x_EKF[1, :], label='EKF' )
    ax11.set_xlabel( 'Time (s)' )
    ax11.set_ylabel( 'y (meters)' )
    ax11.set_title( 'y position vs Time' )
    ax11.set_xlim( t_min, t_max )
    ax11.grid()
    ax11.legend()
    plt.savefig( 'plots/estimated_y_position.png' )

    # Filtered Heading angle
    _, ax12 = plt.subplots()
    ax12.plot( time, np.rad2deg( np.arctan2( np.sin( x_EKF[4, :] ), np.cos( x_EKF[4, :] ) ) ), label='EKF' )
    ax12.set_xlabel( 'Time (s)' )
    ax12.set_ylabel( 'Heading Angle (degrees)' )
    ax12.set_title( 'Estimated Heading Angle vs Time' )
    ax12.set_xlim( t_min, t_max )
    ax12.grid()
    ax12.legend()
    plt.savefig( 'plots/estimated_heading_angle.png' )

    # Estimated trajectory Scatter
    fig13, ax13 = plt.subplots()
    t = np.arange(len(gps_x))
    sc = ax13.scatter( x_EKF[ 0, :], x_EKF[ 1, :], c=t, cmap='viridis', s=2 )
    fig13.colorbar(sc, ax=ax13, label='Time / Index')
    ax13.set_xlabel( 'x (meters)' )
    ax13.set_ylabel( 'y (meters)' )
    ax13.set_title( 'Estimated Trajectory (x,y)' )
    ax13.grid()
    ax13.axis('equal')  # Set equal aspect ratio for x and y axes
    plt.savefig( 'plots/estimated_trajectory_scatter.png' )

    # # Estimated ax bias
    # _, ax14 = plt.subplots()
    # ax14.plot( time, x_EKF[5, :], label='EKF' )
    # ax14.set_xlabel( 'Time (s)' )
    # ax14.set_ylabel( 'Estimated Ax Bias (m/s^2)' )
    # ax14.set_title( 'Estimated Ax Bias vs Time' )
    # ax14.set_xlim( t_min, t_max )
    # ax14.grid()
    # ax14.legend()
    # plt.savefig( 'plots/estimated_ax_bias.png' )

    # # Estimated ay bias
    # _, ax15 = plt.subplots()
    # ax15.plot( time, x_EKF[6, :], label='EKF' )
    # ax15.set_xlabel( 'Time (s)' )
    # ax15.set_ylabel( 'Estimated Ay Bias (m/s^2)' )
    # ax15.set_title( 'Estimated Ay Bias vs Time' )
    # ax15.set_xlim( t_min, t_max )
    # ax15.grid()
    # ax15.legend()
    # plt.savefig( 'plots/estimated_ay_bias.png' )

    # # Estimated yaw rate bias 
    # _, ax16 = plt.subplots()
    # ax16.plot( time, x_EKF[7, :], label='EKF' )
    # ax16.set_xlabel( 'Time (s)' )
    # ax16.set_ylabel( 'Estimated Yaw Rate Bias (dps)' )
    # ax16.set_title( 'Estimated Yaw Rate Bias vs Time' )
    # ax16.set_xlim( t_min, t_max )
    # ax16.grid()
    # ax16.legend()
    # plt.savefig( 'plots/estimated_yaw_rate_bias.png' )


    # plt.show()