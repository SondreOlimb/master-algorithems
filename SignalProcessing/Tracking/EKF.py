import numpy as np

def ekfilter(z, updateNumber):
    dt = 1.0
    j = updateNumber

    # Initialize State
    if updateNumber == 0: # First Update

        # compute position values from measurements
        # x = r*sin(b)
        temp_x = z[0][j]*np.sin(z[1][j]*np.pi/180)
        # y = r*cos(b)
        temp_y = z[0][j]*np.cos(z[1][j]*np.pi/180)

        # state vector
        # - initialize position values
        ekfilter.x = np.array([[temp_x],
                            [temp_y],
                            [0],
                            [0]])

        # state covariance matrix
        # - initialized to zero for first update
        ekfilter.P = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]])

        # state transistion matrix
        # - linear extrapolation assuming constant velocity
        ekfilter.A = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])


        # measurement covariance matrix
        ekfilter.R = z[2][j]

        # system error matrix
        # - initialized to zero matrix for first update
        ekfilter.Q = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]])

        # residual and kalman gain
        # - not computed for first update
        # - but initialized so it could be output
        residual = np.array([[0, 0],
                      [0, 0]])
        K = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]])

    # Reinitialize State
    if updateNumber == 1: # Second Update

        prev_x = ekfilter.x[0][0]
        prev_y = ekfilter.x[1][0]


        # x = r*sin(b)
        temp_x = z[0][j]*np.sin(z[1][j]*np.pi/180)
        # y = r*cos(b)
        temp_y = z[0][j]*np.cos(z[1][j]*np.pi/180)
        temp_xv = (temp_x - prev_x)/dt
        temp_yv = (temp_y - prev_y)/dt

        # state vector
        # - reinitialized with new position and computed velocity
        ekfilter.x = np.array([[temp_x],
                            [temp_y],
                            [temp_xv],
                            [temp_yv]])

        # state covariance matrix
        # - initialized to large values
        # - more accurate position values can be used based on measurement
        #   covariance but this example does not go that far
        ekfilter.P = np.array([[100, 0, 0, 0],
                                 [0, 100, 0, 0],
                                 [0, 0, 250, 0],
                                 [0, 0, 0, 250]])

        # state transistion matrix
        # - linear extrapolation assuming constant velocity
        ekfilter.A = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # measurement covariance matrix
        # - provided by the measurment source
        ekfilter.R = z[2][j]

        # system error matrix
        # - adds 4.5 meter std dev in x and y position to state covariance
        # - adds 2 meters per second std dev in x and y velocity to state covariance
        # - these values are not optimized but work for this example
        ekfilter.Q = np.array([[20, 0, 0, 0],
                                 [0, 20, 0, 0],
                                 [0, 0, 4, 0],
                                 [0, 0, 0, 4]])
        # residual and kalman gain
        # - not computed for first update
        # - but initialized so it could be output
        residual = np.array([[0, 0],
                      [0, 0]])
        K = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]])
    if updateNumber > 1: # Third + Updates

      # Predict State Forward
      x_prime = ekfilter.A.dot(ekfilter.x)

      # Predict Covariance Forward
      P_prime = ekfilter.A.dot(ekfilter.P).dot(ekfilter.A.T) + ekfilter.Q

      # state to measurement transition matrix
      x1 = x_prime[0][0]
      y1 = x_prime[1][0]
      x_sq = x1*x1
      y_sq = y1*y1
      den = x_sq+y_sq
      den1 = np.sqrt(den)
      ekfilter.H = np.array([[  x1/den1,    y1/den1, 0, 0],
                           [y1/den, -x1/den, 0, 0]])

      ekfilter.HT = np.array([[x1/den1, y1/den],
                              [y1/den1, -x1/den],
                              [0, 0],
                              [0, 0]])

      # measurement covariance matrix
      ekfilter.R = z[2][j]

      # Compute Kalman Gain
      S = ekfilter.H.dot(P_prime).dot(ekfilter.HT) + ekfilter.R
      K = P_prime.dot(ekfilter.HT).dot(np.linalg.inv(S))

      # Estimate State
      # temp_z = current measurement in range and azimuth
      temp_z = np.array([[z[0][j]],
                         [z[1][j]]])

      # compute the predicted range and azimuth
      # convert the predicted cartesian state to polar range and azimuth
      pred_x = x_prime[0][0]
      pred_y = x_prime[1][0]

      sumSquares = pred_x*pred_x + pred_y*pred_y
      pred_r = np.sqrt(sumSquares)
      pred_b = np.arctan2(pred_x, pred_y) * 180/np.pi
      h_small = np.array([[pred_r],
                       [pred_b]])

      # compute the residual
      # - the difference between the state and measurement for that data time
      residual = temp_z - h_small

      # Compute new estimate for state vector using the Kalman Gain
      ekfilter.x = x_prime + K.dot(residual)

      # Compute new estimate for state covariance using the Kalman Gain
      ekfilter.P = P_prime - K.dot(ekfilter.H).dot(P_prime)

    return [ekfilter.x[0], ekfilter.x[1], ekfilter.x[2], ekfilter.x[3], ekfilter.P, K, residual, updateNumber];
# End of ekfilter