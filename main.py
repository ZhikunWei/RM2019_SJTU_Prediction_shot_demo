import random

import math
import matplotlib.pyplot as plt
import numpy as np


# parameters for PID
err_sum = 0
last_err = 0
kp = 1.5
ki = 0.5
kd = 0.01
kr = 0.3


# parameters of Kalman filter for position
P = 0
Q = 1
R = 20
x_state = 0
def kf(z):
    global P, Q, R, x_state
    P = P + Q
    K = P / (P+R)
    x_result = x_state + K * (z - x_state)
    x_state = x_result
    P = (1-K) * P
    #print('P%.2f, K%.2f'%(P, K))
    return x_result

# parameters of Kalman filter for speed
Pv = 0
Qv = 1
Rv = 20
x_state_v = 0
def kf_v(z):
    global Pv, Qv, Rv, x_state_v
    Pv = Pv + Qv
    Kv = Pv / (Pv + Rv)
    thres = 2
    if abs(z-x_state_v) > thres:
        z = z * 0.1
        x_result = x_state_v = z*0.5
    else:
        x_result = x_state_v + Kv * (z - x_state_v)

    x_state_v = x_result
    Pv = (1-Kv) * Pv
    return x_result
    
# parameters of Kalman filter for accelerate
Pa = 0
Qa = 1
Ra = 100
x_state_a = 0
def kf_a(z):
    global Pa, Qa, Ra, x_state_a
    Pa = Pa + Qa
    Ka = Pa / (Pa + Ra)
    x_resilt = x_state_a + Ka *(z - x_state_a)
    x_state_a = x_resilt
    Pa = (1-Ka) * Pa
    return x_resilt
    
# parameters of Kalman filter for final prediction without acceleration
P_final = 0
Q_final = 1
R_final = 25
x_state_final = 0
def kf_final(z):
    global P_final, Q_final, R_final, x_state_final
    P_final = P_final + Q_final
    K_final = P_final / (P_final + R_final)
    x_result = x_state_final + K_final * (z - x_state_final)
    x_state_final = x_result
    P_final = (1-K_final) * P_final
    return x_result

# parameters of Kalman filter for final prediction with acceleration
P_final_a = 0
Q_final_a = 1
R_final_a = 25
x_state_final_a = 0
def kf_final_a(z):
    global P_final_a, Q_final_a, R_final_a, x_state_final_a
    P_final_a = P_final_a + Q_final_a
    K_final_a = P_final_a / (P_final_a + R_final_a)
    x_result_a = x_state_final_a + K_final_a * (z - x_state_final_a)
    x_state_final_a = x_result_a
    P_final_a = (1-K_final_a) * P_final_a
    return x_result_a

def pidv(err):
    global err_sum, last_err
    t = 0
    t += kp * err
    t += ki * err_sum
    t += kd * (err - last_err)
    last_err = err
    err_sum += err
    return t


# variables for simulation
x = []  # time
A = 5  # amplitude of the movement
noise = 0.1  # noise level, the noise is sampled from N(0, noise)
truth = []  # truth curve
sample = []  # sample points with noise
delay_step = 1  # delay steps from sample to truth
itr_len = 60  # the steps of each phase


# adding truth curve, this can be flexible if you want to simulate different movement
for i in range(int(1.5*itr_len), 3*itr_len):
    x.append(i)
    truth.append(A + A*math.sin( (i-int(1.5*itr_len)) / int(1.5*itr_len) *3.14))


# sample points from truth curve, there is a delay in the beginning
for i in range(delay_step):
    sample.append(0)
for i in range(delay_step, len(truth)):
    sample.append(truth[i-delay_step]+random.gauss(0, noise))

# variables for prediction
predict = []
last = 0
cur = 0
len = 1
delay = [0] * len
i = 0
target = 0

kf_predict = []
raw_v = []
kf_predict_v = []
raw_a = []
kf_predict_a = []
kf_predict_final = []
kf_predict_final_a = []
last_x = 0
last_v = 0
for index, y in enumerate(sample):
    output = kf(y)
    kf_predict.append(output)
    
    cur_v = output - last_x
    last_x = output
    output_v = kf_v(cur_v)
    raw_v.append(cur_v)
    kf_predict_v.append(output_v)
    
    cur_a = output_v - last_v
    last_v = output_v
    output_a = kf_a(cur_a)
    raw_a.append(cur_a)
    kf_predict_a.append(output_a)

    target = output + output_v * 15
    target_a = target + 1/2*output_a*15*15
    output_final = kf_final(target)
    kf_predict_final.append(output_final)
    output_final_a = kf_final_a(target_a)
    kf_predict_final_a.append(output_final_a)


plt.plot(x, truth, color='red', label="truth")
plt.scatter(x, sample, color='green', marker='.', label='sample')
plt.plot(x, kf_predict, color='pink', label='kf_position')

plt.plot(x, raw_v, label='raw_v')
plt.plot(x, kf_predict_v, label='kf_v')
plt.plot(x, raw_a, label='raw_a')
plt.plot(x, kf_predict_a, label='kf_a')
#plt.plot(x, predict, color='blue', label="kalman")
plt.plot(x, kf_predict_final, label='final')
plt.plot(x, kf_predict_final_a, label='final_a')
#plt.plot(x, [0]*itr_len)
plt.legend()
plt.show()
