import numpy as np
import math

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
    d = 1

    # Get number of samples
    m = int(np.log(1-P)/(np.log(1-(1-epsilon)**p)))
    print("Number of samples=",m)
    best_median = 1.0*10**20

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
        #residuals[k] = abs(a*x_vec[k]+b*y_vec[k]+c)/(np.sqrt(a**2+b**2))
        residuals[k] = Calc_distance(a,b,c,x_vec[k],y_vec[k])

    return np.median(residuals)

def Calc_distance(a,b,c,point_x,point_y):
    '''
    This calculate the distance between a point(point_x,pont_y) to the line(ax+by+c=0).
    This functino is used in the following function "getIfPointOnTheLine"
    :param a:
    :param b:
    :param c:
      ===> for ax+by+c=0
    :param point_x:
    :param point_y:
      ===> the point to be calculated
    :return: the distance
    '''
    numer = abs(a*point_x + b*point_y + c)
    denom = math.sqrt(pow(a,2)+pow(b,2))
    return numer/denom

def getIfPointsOnTheLine(a,b,c,points,buffer): #直線ax+by+c=0 点([x,y])
    '''
    This function decides if a point is close enough to the given line.
    :param a:
    :param b:
    :param c:
    :param points: The points to be decided
    :param buffer: The tolerance. If the point is in the area of buffer from the line, the point will be labeled 1
    :return: array with the results (0=> not on the line, 1=> on the line)
    '''
    conditions = np.zeros(points.shape[0])
    for i in range(len(points)-1):
        if Calc_distance(a,b,c,*points[i])<buffer:
            conditions[i]=1
    return conditions