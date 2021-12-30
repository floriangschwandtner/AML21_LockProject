import numpy as np

""" 
 Function definition for Least Median Square Estimator
 Implementation from: http://www-sop.inria.fr/odyssee/software/old_robotvis/Tutorial-Estim/node25.html
 x_vec: Vector of values in x-direction
 y_vec: Vector of values in y-direction
 epsilon: Percentage of outliers
 P: Percentage, that one sample is good
"""
def LMSQEstimator(x_vec, y_vec, epsilon=0.4, P=0.99, p=6):
    # Check vector length
    assert len(x_vec)==len(y_vec)
    n = len(x_vec)

    # define degree, two for line
    d = 2

    # Get number of samples
    m = int(np.log(1-P)/(np.log(1-(1-epsilon)**p)))
    print("Number of samples=",m)
    best_median = 1.0*10**20;

    beta = [0.0, 0.0]
    beta_est = [0.0, 0.0]

    for i in range(0,m):
        # choose two random Indexes
        randIndex = np.random.choice(n, p)

        # extract random points
        x_pts = np.take(x_vec, randIndex)
        y_pts = np.take(y_vec, randIndex)

        # Estimate y=kx+d
        beta_est = np.polyfit(x_pts, y_pts, d)
        # beta_est[0] = (y_pts[0]-y_pts[1])/(x_pts[0]-x_pts[1])
        # beta_est[1] = y_pts[1]+beta_est[0]*x_pts[1]

        #get Median
        median = MedianOfSquaredResiduals(x_vec, y_vec, beta_est)

        if median<best_median:
            print("New k=",beta[0]," d=",beta[1])
            beta = beta_est
            best_median = median

    return beta


"""
 Calculate the Median of square residuals
 x_vec: Vector of values in x-direction
 y_vec: Vector of values in y-direction
 beta: parameter vector
"""
def MedianOfSquaredResiduals(x_vec, y_vec, beta):
    
    assert len(x_vec)==len(y_vec)
    
    # Create residuals vector
    residuals = np.zeros(len(x_vec))

    a = beta[0]
    c = beta[1]
    b = -1.0

    for k in range(0, len(x_vec)):
        # Calculate residual to line for every point
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
        # residuals[k] = (y_vec[k]-(beta[0]*x_vec[k]+beta[1]))**2
        residuals[k] = abs(a*x_vec[k]+b*y_vec[k]+c)/(np.sqrt(a**2+b**2))

    return np.median(residuals)