import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm


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

    return np.column_stack( ( n, e, -d ) )  # return as NEU


def CalculateRMSE( traj_pos, traj_vel, traj_accel, traj_orient, est_states ):

    # Heading angle
    ins_heading = ( np.degrees( est_states[:, 0] )  + 180 ) % 360 - 180  # INS heading in degrees
    heading_error = traj_orient[:, 0] - ins_heading
    heading_error_wrapped = (heading_error + 180) % 360 - 180
    heading_rmse = np.sqrt( np.mean( heading_error_wrapped**2 ) )

    print( f'Heading RMSE: {heading_rmse:.2f} degrees' )

    # North position
    xN_rsme = np.sqrt( np.mean( ( traj_pos[:, 0] - est_states[:, 2] )**2 ) )
    print( f'North Position RMSE: {xN_rsme:.2f} meters' )

    # East position
    xE_rsme = np.sqrt( np.mean( ( traj_pos[:, 1] - est_states[:, 5] )**2 ) )
    print( f'East Position RMSE: {xE_rsme:.2f} meters' )

    # North velocity
    vN_rsme = np.sqrt( np.mean( ( traj_vel[:, 0] - est_states[:, 3] )**2 ) )
    print( f'North Velocity RMSE: {vN_rsme:.2f} m/s' )

    # East velocity
    vE_rsme = np.sqrt( np.mean( ( traj_vel[:, 1] - est_states[:, 6] )**2 ) )
    print( f'East Velocity RMSE: {vE_rsme:.2f} m/s' )

    # Ground speed
    speed_traj = np.sqrt( traj_vel[:, 0]**2 + traj_vel[:, 1]**2 )
    speed_est = np.sqrt( est_states[:, 3]**2 + est_states[:, 6]**2 )
    speed_rmse = np.sqrt( np.mean( ( speed_traj - speed_est )**2 ) )
    print( f'Ground Speed RMSE: {speed_rmse:.2f} m/s' )

    # North acceleration
    aN_rsme = np.sqrt( np.mean( ( traj_accel[:, 0] - est_states[:, 4] )**2 ) )
    print( f'North Acceleration RMSE: {aN_rsme:.2f} m/s^2' )

    # East acceleration
    aE_rsme = np.sqrt( np.mean( ( traj_accel[:, 1] - est_states[:, 7] )**2 ) )
    print( f'East Acceleration RMSE: {aE_rsme:.2f} m/s^2' )


def plotsAppendixA( time, accel, gyro, mag, v_gps, lla, traj_pos, traj_vel, traj_ang_vel, traj_orient ):

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
    ax1[ 1 ].set_ylabel( 'ay (m/s^2)' )
    ax1[ 1 ].set_xlim( t_min, t_max )
    ax1[ 1 ].grid()

    ax1[ 2 ].plot( time, gyro[:, 2], label='yaw rate' )
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

    plt.savefig( 'plots/appendixA/IMU_measurements.png', bbox_inches='tight', dpi=600 )

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

    plt.savefig( 'plots/appendixA/GPS_measurements.png', bbox_inches='tight', dpi=600 )

    # 3- True trajectory
    # Trajectory
    x = traj_pos[:, 0]
    y = traj_pos[:, 1]
    _, ax3 = plt.subplots()
    ax3.plot(x[:-1], y[:-1], '--', linewidth=1.5, label='True')
    ax3.plot(x[-1], y[-1], marker='D', markersize=8)
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.set_xlabel( 'North (m)' )
    ax3.set_ylabel( 'East (m)' )
    plt.savefig( 'plots/appendixA/true_trajectory.png', bbox_inches='tight', dpi=600 )

    # 4- Some true states
    _, ax4 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax4.flatten()
    ax4[ 0 ].plot( time, np.sqrt( traj_vel[:, 0]**2 + traj_vel[:, 1]**2 ), label='True Speed' )
    ax4[ 0 ].set_ylabel( 'Speed (m/s)' )
    ax4[ 0 ].set_xlim( t_min, t_max )
    ax4[ 0 ].grid()
    ax4[ 1 ].plot( time, np.radians(traj_orient[:, 0]), label='Heading angle' )
    ax4[ 1 ].set_ylabel( 'Heading (deg)' )
    ax4[ 1 ].set_xlim( t_min, t_max )
    ax4[ 1 ].grid()
    ax4[ 2 ].plot( time, traj_ang_vel[:, 2], label='Yaw rate' )
    ax4[ 2 ].set_ylabel( 'Yaw Rate (rad/s)' )
    ax4[ 2 ].set_xlim( t_min, t_max )
    ax4[ 2 ].grid()
    ax4[ 2 ].set_xlabel( 'Time (s)' )
    plt.savefig( 'plots/appendixA/true_states.png', bbox_inches='tight', dpi=600 )

    # 5- True velocities
    _, ax5 = plt.subplots( 3, 1, sharex=True, figsize=(6, 6) )
    ax5.flatten()
    ax5[ 0 ].plot( time, traj_vel[:, 0], label='v_N' )
    ax5[ 0 ].set_ylabel( 'North Speed (m/s)' )
    ax5[ 0 ].set_xlim( t_min, t_max )
    ax5[ 0 ].grid()

    ax5[ 1 ].plot( time, traj_vel[:, 1], label='v_E' )
    ax5[ 1 ].set_ylabel( 'East Speed (m/s)' )
    ax5[ 1 ].set_xlim( t_min, t_max )
    ax5[ 1 ].grid()

    ax5[ 2 ].plot( time, traj_vel[:, 2], label='v_D' )
    ax5[ 2 ].set_ylabel( 'Up Speed (m/s)' )
    ax5[ 2 ].set_xlim( t_min, t_max )
    ax5[ 2 ].grid()
    ax5[ 2 ].set_xlabel( 'Time (s)' )
    plt.savefig( 'plots/appendixA/true_velocities.png', bbox_inches='tight', dpi=600 )

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
    plt.savefig( 'plots/appendixA/gps_velocities.png', bbox_inches='tight', dpi=600 )

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
    plt.savefig( 'plots/appendixA/gps_lla.png', bbox_inches='tight', dpi=600 )


def plotEstimation( time, gyro, gps_pos, gps_vel, traj_pos, traj_vel, traj_accel, traj_orient, est_states ):
    # Plot limits
    t_min = time[ 0 ]
    t_max = time[ -1 ]

    fig1, ax1 = plt.subplots( 2, 1, sharex=True, figsize=(3.5,3.5) )

    angle_0 = traj_orient[ 0, 0 ]  # initial heading angle from true trajectory
    yaw_int = np.cumsum( gyro[:, 2] ) * 0.01  # integrate yaw rate to get yaw angle (in radians)
    yaw_int = ( np.degrees( yaw_int ) + angle_0 + 180 ) % 360 - 180  # convert to degrees

    ins_heading = ( np.degrees( est_states[:, 0] )  + 180 ) % 360 - 180  # INS heading in degrees

    ax1[0].plot( time, traj_orient[:, 0], color='C0', linewidth=1.25, label='True' )
    ax1[0].plot( time, yaw_int, color='C2', linewidth=1.25, label='IMU' )
    ax1[0].plot( time, ins_heading, color='C1', linewidth=1.25, label='INS' )

    ax1[0].set_ylabel( 'Heading (deg)' )
    ax1[0].set_xlim( t_min, t_max )
    ax1[0].grid()

    ax1[1].plot( time, np.rad2deg(est_states[:, 1]), color='C1', linewidth=1.25, label='INS' )
    ax1[1].axhline( y=np.rad2deg(0.122), color='C0', linewidth=1.25, label='True')
    ax1[1].set_xlabel( 'Time (s)' )
    ax1[1].set_ylabel( 'Gyro Bias (deg/s)' )
    ax1[1].set_xlim( t_min, t_max )
    ax1[1].grid()

    # Single horizontal legend above plots
    handles, labels = ax1[0].get_legend_handles_labels()

    fig1.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig1.subplots_adjust(left=0.18)

    fig1.savefig( 'plots/INS/heading_est.png', bbox_inches='tight', dpi=600 )


    # Trajectory
    x = traj_pos[:, 0]
    y = traj_pos[:, 1]
    fig2, ax2 = plt.subplots( figsize=(3.5, 3.5) )
    ax2.plot( x[:-1], y[:-1], color="C0", linewidth=1.25, label='True')
    ax2.plot( x[-1], y[-1], marker='D', markersize=6 )
    ax2.scatter( gps_pos[:, 0], gps_pos[:, 1], color="C2", s=1, label='GPS' )
    ax2.plot( est_states[:, 2], est_states[:, 5], color="C1", linewidth=1.25, label='INS' )
    ax2.plot( est_states[-1, 2], est_states[-1, 5], marker='D', markersize=6 )
  
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel( 'xN (m)' )
    ax2.set_ylabel( 'xE (m)' )

    # Single horizontal legend above plots
    handles, labels = ax2.get_legend_handles_labels()

    fig2.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig2.savefig( 'plots/INS/trajectory_est.png', bbox_inches='tight', dpi=600 )

    # xN vs. time
    fig3, ax3 = plt.subplots( 2, 1, sharex=True, figsize=(3.5,3.5) )
    ax3[0].plot( time, traj_pos[:, 0], color="C0", linewidth=1.25, label='True' )
    ax3[0].scatter( time, gps_pos[:, 0], color="C2", s=1, label='GPS' )
    ax3[0].plot( time, est_states[:, 2], color="C1", linewidth=1.2, label='INS' )
    ax3[0].set_ylabel( 'xN (m)' )
    ax3[0].set_xlim( t_min, t_max )
    ax3[0].grid()

    # xE vs. time
    ax3[1].plot( time, traj_pos[:, 1], color="C0", linewidth=1.25, label='True' )
    ax3[1].scatter( time, gps_pos[:, 1], color="C2", s=1, label='GPS' )
    ax3[1].plot( time, est_states[:, 5], color="C1", linewidth=1.25, label='INS' )
    ax3[1].set_xlabel( 'Time (s)' )
    ax3[1].set_ylabel( 'xE (m)' )
    ax3[1].set_xlim( t_min, t_max )
    ax3[1].grid()

    # Single horizontal legend above plots
    handles, labels = ax3[0].get_legend_handles_labels()

    fig3.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig3.subplots_adjust(left=0.18)

    fig3.savefig( 'plots/INS/position_est.png', bbox_inches='tight', dpi=600 )

    # v_N in NEU frame
    fig4, ax4 = plt.subplots( 2, 1, sharex=True, figsize=(3.5,3.5) )
    ax4[0].plot( time, traj_vel[:, 0], color="C0", label='True' )
    # ax4[0].scatter( time, gps_vel[:, 0], color="C2", s=2, label='GPS' )
    ax4[0].plot( time, est_states[:, 3], color="C1", label='INS' )
    ax4[0].set_ylabel( 'vN (m/s)' )
    ax4[0].set_xlim( t_min, t_max )
    ax4[0].grid()

    # v_E in NEU frame
    ax4[1].plot( time, traj_vel[:, 1], color="C0", label='True' )
    # ax4[1].scatter( time, gps_vel[:, 1], color="C2", s=2, label='GPS' )
    ax4[1].plot( time, est_states[:, 6], color="C1", label='INS' )
    ax4[1].set_xlabel( 'Time (s)' )
    ax4[1].set_ylabel( 'vE (m/s)' )
    ax4[1].set_xlim( t_min, t_max )
    ax4[1].grid()

    # Single horizontal legend above plots
    handles, labels = ax4[0].get_legend_handles_labels()

    fig4.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig4.subplots_adjust(left=0.18)

    fig4.savefig( 'plots/INS/velocity_est.png', bbox_inches='tight', dpi=600 )


    # Ground velocity
    fig5, ax5 = plt.subplots( figsize=(3.5,2.5))
    ax5.plot( time, np.sqrt( traj_vel[:, 0]**2 + traj_vel[:, 1]**2 ), color="C0", linewidth=1.25, label='True' )
    ax5.scatter( time, np.sqrt( gps_vel[:, 0]**2 + gps_vel[:, 1]**2 ), color="C2", s=1, label='GPS' )
    ax5.plot( time, np.sqrt( est_states[:, 3]**2 + est_states[:, 6]**2 ), color="C1", linewidth=1.25, label='INS' )
    ax5.set_xlabel( 'Time (s)' )
    ax5.set_ylabel( 'Ground Speed (m/s)' )
    ax5.set_xlim( t_min, t_max )
    ax5.grid()
    # Single horizontal legend above plots
    handles, labels = ax5.get_legend_handles_labels()

    fig5.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False
    )
    
    fig5.savefig( 'plots/INS/ground_speed_est.png', bbox_inches='tight', dpi=600 )

    # a_N in NEU frame
    fig6, ax6 = plt.subplots( 2, 1, sharex=True, figsize=(3.5,3.5) )
    ax6[0].plot( time, traj_accel[:, 0], color="C0", label='True' )
    # ax6[0].plot( time, meas_a_N, color="C2", label='Calculated from IMU' )
    ax6[0].plot( time, est_states[:, 4], color="C1", label='INS' )
    ax6[0].set_ylabel( 'aN (m/s^2)' )
    ax6[0].set_xlim( t_min, t_max )
    ax6[0].grid()

    # a_E in NEU frame
    ax6[1].plot( time, traj_accel[:, 1], color="C0", label='True' )
    # ax6[1].plot( time, meas_a_E, color="C2", label='Calculated from IMU' )
    ax6[1].plot( time, est_states[:, 7], color="C1", label='INS' )
    ax6[1].set_xlabel( 'Time (s)' )
    ax6[1].set_ylabel( 'aE (m/s^2)' )
    ax6[1].set_xlim( t_min, t_max )
    ax6[1].grid()

    # Single horizontal legend above plots
    handles, labels = ax6[0].get_legend_handles_labels()

    fig6.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig6.subplots_adjust(left=0.18)

    fig6.savefig( 'plots/INS/accel_est.png', bbox_inches='tight', dpi=600 )


def plotAblation( time, traj_pos, traj_vel, est_gps, est_no_gps ):
    # Plot limits
    t_min = time[ 0 ]
    t_max = time[ -1 ]

    # xN vs. time
    fig1, ax1 = plt.subplots( 2, 1, sharex=True, figsize=(3.5,3.5) )
    ax1[0].plot( time, traj_pos[:, 0], color="C0", linewidth=1.25, label='True' )
    ax1[0].plot( time, est_gps[:, 2], color="C1", linewidth=1.2, label='INS' )
    ax1[0].plot( time, est_no_gps[:, 2], color="C2", linewidth=1.2, label='INS - no GPS' )
    ax1[0].set_ylabel( 'xN (m)' )
    ax1[0].set_xlim( t_min, t_max )
    ax1[0].grid()

    # xE vs. time
    ax1[1].plot( time, traj_pos[:, 1], color="C0", linewidth=1.25, label='True' )
    ax1[1].plot( time, est_gps[:, 5], color="C1", linewidth=1.25, label='INS' )
    ax1[1].plot( time, est_no_gps[:, 5], color="C2", linewidth=1.25, label='INS - no GPS' )
    ax1[1].set_xlabel( 'Time (s)' )
    ax1[1].set_ylabel( 'xE (m)' )
    ax1[1].set_xlim( t_min, t_max )
    ax1[1].grid()

    # Single horizontal legend above plots
    handles, labels = ax1[0].get_legend_handles_labels()

    fig1.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig1.subplots_adjust(left=0.18)

    fig1.savefig( 'plots/INS-NoGPS/position_ablation.png', bbox_inches='tight', dpi=600 )

    # Trajectory
    x = traj_pos[:, 0]
    y = traj_pos[:, 1]
    fig2, ax2 = plt.subplots( figsize=(3.5, 3.5) )
    ax2.plot( x[:-1], y[:-1], color="C0", linewidth=1.25, label='True')
    ax2.plot( x[-1], y[-1], marker='D', markersize=6 )
    ax2.plot( est_gps[:, 2], est_gps[:, 5], color="C1", linewidth=1.25, label='INS' )
    ax2.plot( est_gps[-1, 2], est_gps[-1, 5], marker='D', markersize=6 )
    ax2.plot( est_no_gps[:, 2], est_no_gps[:, 5], color="C2", linewidth=1.25, label='INS - no GPS' )
    ax2.plot( est_no_gps[-1, 2], est_no_gps[-1, 5], marker='D', markersize=6 )
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel( 'xN (m)' )
    ax2.set_ylabel( 'xE (m)' )

    # Single horizontal legend above plots
    handles, labels = ax2.get_legend_handles_labels()

    fig2.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False
    )

    fig2.savefig( 'plots/INS-NoGPS/trajectory_ablation.png', bbox_inches='tight', dpi=600 )

    # Ground Speed
    # Ground velocity
    fig3, ax3 = plt.subplots( figsize=(3.5,2.5))
    ax3.plot( time, np.sqrt( traj_vel[:, 0]**2 + traj_vel[:, 1]**2 ), color="C0", linewidth=1.25, label='True' )
    ax3.plot( time, np.sqrt( est_gps[:, 3]**2 + est_gps[:, 6]**2 ), color="C1", linewidth=1.25, label='INS' )
    ax3.plot( time, np.sqrt( est_no_gps[:, 3]**2 + est_no_gps[:, 6]**2 ), color="C2", linewidth=1.25, label='INS - no GPS' )
    ax3.set_xlabel( 'Time (s)' )
    ax3.set_ylabel( 'Ground Speed (m/s)' )
    ax3.set_xlim( t_min, t_max )
    ax3.grid()
    # Single horizontal legend above plots
    handles, labels = ax3.get_legend_handles_labels()

    fig3.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False
    )
    
    fig3.savefig( 'plots/INS-NoGPS/ground_speed_ablation.png', bbox_inches='tight', dpi=600 )