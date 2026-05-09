import numpy as np

class INS():

    def __init__( self, Ts, mag: np.ndarray ):
        
        # Assumption: vehicle starts at rest at the origin (0,0)

        self.Ts = Ts  # Sampling time (seconds)

        # Magnetic declination at the location (radians) - Found from 1st lat, lon and https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
        self.mag_declination = 0.23562  

        # Initialize heading angle from first magnetometer reading
        m = np.sqrt( mag[0]**2 + mag[1]**2 )  
        mx = mag[0] / m
        my = mag[1] / m

        # 1st Filter: Heading angle and yaw rate estimation
        self.s1 = np.array( [
            [ np.arctan2( -my, mx ) - self.mag_declination ],          # State: heading angle (rad)
            [ 0 ]                               # State: yaw rate (rad/s)
        ] )  

        # State: [heading, bias]
        self.A1 = np.array([
            [ 1,   -Ts ],   # heading += -bias * Ts
            [ 0,   1   ]
        ])  # bias is modeled as constant

        self.B1 = np.array([
            [ Ts ],       # gyro input drives heading
            [ 0 ]
        ])      # gyro doesn't directly update bias

        self.C1 = np.array([
            [ 1, 0 ]
        ])    # magnetometer only observes heading

        self.P1 = np.array([
            [ 0.1,   0  ],
            [ 0,     100 ]
        ])  # high initial bias uncertainty

        self.Q1 = np.array([
            [ 1,   0    ],
            [ 0,   1e-2 ]
        ])  # bias changes very slowly

        self.R1 = np.array([
            [ 0.1 ]
        ])         
        
        # Filter 2: North direction position, velocity, and acceleration estimation
        self.s2 = np.array( [
            [ 0 ],                          # State: position in North direction (xN)       
            [ -1.04 ],                      # State: velocity in North direction (vN)
            [ 0 ]                           # State: acceleration in North direction (aN)
        ])

        # Filter 2 state transition matrix
        self.A2 = np.array( [
            [ 1,   self.Ts,   1/2 * self.Ts ** 2 ],
            [ 0,   1,         self.Ts            ],
            [ 0,   0,         1                  ] 
        ])

        # Filter 2 measurement matrix
        self.C2 = np.array([
            [ 0, 0, 1 ]
        ])

        # Filter 2 error covariance matrix
        self.P2 = np.array([
            [ 0.1,   0,     0   ],
            [ 0,     0.1,   0   ],
            [ 0,     0,     0.1 ]
        ] )  

        # Filter 2 process noise covariance matrix
        self.Q2 = np.array( [
            [ 0.01,   0,      0    ],
            [ 0,      0.01,   0    ],
            [ 0,      0,      0.01 ]
        ] )

        # Filter 2 measurement noise covariance matrix
        self.R2 = np.array( [
            [ 0.5 ]
        ] )

        # Filter 3: East direction position, velocity, and acceleration estimation
        self.s3 = np.array( [
            [ 0    ],                      # State: position in East direction (xN)       
            [ 1.97 ],                      # State: velocity in East direction (vN)
            [ 0    ]                       # State: acceleration in East direction (aN)
        ])

        # Filter 3 state transition matrix
        self.A3 = np.array( [
            [ 1,   self.Ts,   1/2 * self.Ts ** 2 ],
            [ 0,   1,         self.Ts            ],
            [ 0,   0,         1                  ] 
        ])

        # Filter 3 measurement matrix
        self.C3 = np.array([
            [ 0, 0, 1 ]
        ])

        # Filter 3 error covariance matrix
        self.P3 = np.array([
            [ 0.1,   0,     0   ],
            [ 0,     0.1,   0   ],
            [ 0,     0,     0.1 ]
        ] )  

        # Filter 3 process noise covariance matrix
        self.Q3 = np.array( [
            [ 0.01, 0, 0 ],
            [ 0, 0.01, 0 ],
            [ 0, 0, 0.01 ]
        ] )

        # Filter 3 measurement noise covariance matrix
        self.R3 = np.array( [
            [ 0.1 ]
        ] )


        self.A4 = np.array( [
            [ 1,   self.Ts,   1/2 * self.Ts ** 2,   0,   0,         0                  ],
            [ 0,   1,         self.Ts           ,   0,   0,         0                  ],
            [ 0,   0,         1                 ,   0,   0,         0                  ],
            [ 0,   0,         0,                    1,   self.Ts,   1/2 * self.Ts ** 2 ],
            [ 0,   0,         0,                    0,   1,         self.Ts            ],
            [ 0,   0,         0,                    0,   0,         1                  ]
        ])

        self.P4 = np.eye( 6 ) * 0.1

        self.Q4 = np.array( [
            [ 0.01, 0, 0, 0, 0, 0 ],
            [ 0, 0.01, 0, 0, 0, 0 ],
            [ 0, 0, 0.01, 0, 0, 0 ],
            [ 0, 0, 0, 0.01, 0, 0 ],
            [ 0, 0, 0, 0, 0.01, 0 ],
            [ 0, 0, 0, 0, 0, 0.01 ]
        ])
        
        self.R4 = np.array( [
            [ 100, 0, 0 ],
            [ 0, 100, 0 ],
            [ 0, 0, 0.5 ]
        ] )

        
    def StatePrediction( self, gyro: np.ndarray ):
        u = np.array( [
            [ gyro[ 2 ] ]
        ])   # Yaw rate measurement (rad/s)

        self.s1 = self.A1 @ self.s1 + self.B1 @ u
        self.s2 = self.A2 @ self.s2
        self.s3 = self.A3 @ self.s3

    
    def HeadingUpdateKF( self, gyro: np.ndarray, mag: np.ndarray ):

        s1_p = self.s1.copy()
        
        # Predict error covariance
        P1_p = self.A1 @ self.P1 @ self.A1.T + self.Q1

        # Predict the output
        y_p = self.C1 @ s1_p
        
        # Measurement vector
        m = np.sqrt( mag[ 0 ]**2 + mag[ 1 ]**2 )  
        mx = mag[ 0 ] / m
        my = mag[ 1 ] / m

        y1 = np.array( [
            [ np.arctan2( -my, mx ) - self.mag_declination ],          # Measured heading angle (rad)
        ] )

        # Calculate the Kalman gain
        K1 = P1_p @ self.C1.T @ np.linalg.inv( self.C1 @ P1_p @ self.C1.T + self.R1 )

        # State correction
        wrapped_error = ( y1 - y_p + np.pi ) % ( 2 * np.pi ) - np.pi  # Wrap angle error to [-pi, pi]
        self.s1 = s1_p +  K1 @ wrapped_error

        # Error covariance correction
        self.P1 = ( np.eye( 2 ) - K1 @ self.C1 ) @ P1_p

    
    def  IMUUpdateKF( self, accel: np.ndarray ):
        # Part 1: Update the North components
        # Copy the predicted state
        s2_p = self.s2.copy()

        # Predict error covariance
        P2_p = self.A2 @ self.P2 @ self.A2.T + self.Q2

        # Predict the output
        y_p = self.C2 @ s2_p

        # Measurement vector
        aN_m = accel[ 0 ] * np.cos( self.s1[ 0, 0 ] ) - accel[ 1 ] * np.sin( self.s1[ 0, 0 ] ) 
        aE_m = accel[ 0 ] * np.sin( self.s1[ 0, 0 ] ) + accel[ 1 ] * np.cos( self.s1[ 0, 0 ] )

        y2 = np.array( [
            [ aN_m ]                       # Measured acceleration in North direction (m/s^2)
        ] )

        # Calculate the Kalman gain
        K2 = P2_p @ self.C2.T @ np.linalg.inv( self.C2 @ P2_p @ self.C2.T + self.R2 )

        # State correction
        self.s2 = s2_p +  K2 @ ( y2 - y_p )

        # Error covariance correction
        self.P2 = ( np.eye( 3 ) - K2 @ self.C2 ) @ P2_p

        # Part 2: Update the East components
        # Copy the predicted state
        s3_p = self.s3.copy()

        # Predict error covariance
        P3_p = self.A3 @ self.P3 @ self.A3.T + self.Q3

        # Predict the output
        y_p = self.C3 @ s3_p

        # Measurement vector
        y3 = np.array( [
            [ aE_m ]                       # Measured acceleration in East direction (m/s^2)
        ] )

        # Calculate the Kalman gain
        K3 = P3_p @ self.C3.T @ np.linalg.inv( self.C3 @ P3_p @ self.C3.T + self.R3 )

        # State correction
        self.s3 = s3_p +  K3 @ ( y3 - y_p )

        # Error covariance correction
        self.P3 = ( np.eye( 3 ) - K3 @ self.C3 ) @ P3_p


    def GPSUpdateKF( self, gps_pos: np.ndarray, gps_vel: np.ndarray ):

        # Copy the predicted state
        s4_p = np.vstack( ( self.s2, self.s3 ) )  # Combined state vector for position and velocity

        # Predict error covariance
        P4_p = self.A4 @ self.P4 @ self.A4.T + self.Q4

        # Predicted output
        y_p = np.array( [
            [ s4_p[ 0, 0 ] ],  # Predicted position in North direction (xN)
            [ s4_p[ 3, 0 ] ],  # Predicted position in East direction (xE)
            [ np.sqrt( s4_p[ 1, 0 ]**2 + s4_p[ 4, 0 ]**2 ) ]  # Predicted ground speed
        ] )
        
        # Calculate Jacobian of the measurement function
        H = np.zeros( ( 3, 6 ) )
        vN = s4_p[ 1, 0 ]  # Predicted velocity in North direction
        vE = s4_p[ 4, 0 ]  # Predicted velocity in East direction
        H[ 0, 0 ] = 1  # xN measurement
        H[ 1, 3 ] = 1  # xE measurement
        H[ 2, 1 ] = vN / ( np.sqrt( vN**2 + vE**2 ) + 1e-6 ) # vN measurement
        H[ 2, 4 ] = vE / ( np.sqrt( vN**2 + vE**2 ) + 1e-6 ) # vE measurement

        # Calculate the Kalman gain
        K4 = P4_p @ H.T @ np.linalg.inv( H @ P4_p @ H.T + self.R4 )

        # Measurement vector
        y4 = np.array( [
            [ gps_pos[ 0 ] ],  # Measured position in North direction (xN)
            [ gps_pos[ 1 ] ],  # Measured position in East direction (xE)
            [ np.sqrt( gps_vel[ 0 ]**2 + gps_vel[ 1 ]**2 ) ]  # Measured ground speed
         ] )
        
        # State correction
        s4_corrected = s4_p + K4 @ ( y4 - y_p )

        # Error covariance correction
        self.P4 = ( np.eye( 6 ) - K4 @ H ) @ P4_p

        # Update the individual filter states with the corrected combined state
        self.s2[ 0, 0 ] = s4_corrected[ 0, 0 ]  # Updated position in North direction (xN)
        self.s2[ 1, 0 ] = s4_corrected[ 1, 0 ]  # Updated velocity in North direction (vN)
        self.s2[ 2, 0 ] = s4_corrected[ 2, 0 ]  # Updated acceleration in North direction (aN)
        self.s3[ 0, 0 ] = s4_corrected[ 3, 0 ]  # Updated position in East direction (xE)
        self.s3[ 1, 0 ] = s4_corrected[ 4, 0 ]  # Updated velocity in East direction (vE)
        self.s3[ 2, 0 ] = s4_corrected[ 5, 0 ]  # Updated acceleration in East direction (aE)