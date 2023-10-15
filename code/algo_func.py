import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import math

from math import pi

import scipy.signal as sg
import scipy as sp
import pandas as pd

from tqdm import tqdm

# code for lms using mse

def lms_mse(ip,op,l,step_size = 0):

    ip_len = len(ip)

    if step_size == 0:

        sig_power = np.sum(np.square(ip))/ip_len
        step_size = 1/(10*w_len*sig_power)

    w_len = l+1

    op_len = len(op)

    w = np.zeros((op_len,w_len))
    e = np.zeros((op_len,1))
    y = np.zeros((op_len,1))

    for i in range(op_len):

        if i == 0:

            w_temp = np.zeros((w_len,1))

        ip_1 = ip[i:i+w_len]
        ip_1 = ip_1[::-1]
        y_pred = ip_1.T@w_temp
        y_pred = y_pred.item()

        e_temp = op[i].item() - y_pred

        w_temp = w_temp + step_size*e_temp*ip_1

        y[i] = y_pred
        e[i] = e_temp
        w[i,:] = w_temp[:,0]

    return y,e,w

# code for plotting the learning curve of the algorithm

def learn_curve(e):
    
    curve = []
    s = 0
    i=0
    
    while i<len(e):
        
        s +=(e[i]**2)
        tmp = s/(i+1)
        curve.append(tmp)
        i+=1
        
    return curve

# to determine the kernel value

def kern_1(x,kern_size):

    num = np.exp(-(x**2)/(2*(kern_size**2)))
    dum = 1/(math.sqrt(2*pi)*kern_size)
    return num/dum

# lms with MCC loss.

def lms_mcc(ip,op,l,step_size = 0,kern_size = 1):

    ip_len = len(ip)

    if step_size == 0:

        sig_power = np.sum(np.square(ip))/ip_len
        step_size = 1/(10*w_len*sig_power)

    w_len = l+1

    op_len = len(op)

    w = np.zeros((op_len,w_len))
    e = np.zeros((op_len,1))
    y = np.zeros((op_len,1))

    step_size = step_size/(kern_size**2)

    for i in range(op_len):

        if i == 0:

            w_temp = np.zeros((w_len,1))

        ip_1 = ip[i:i+w_len]
        ip_1 = ip_1[::-1]
        y_pred = ip_1.T@w_temp
        y_pred = y_pred.item()

        e_temp = op[i].item() - y_pred

        G = kern_1(e_temp,kern_size)

        w_temp = w_temp + step_size*e_temp*ip_1*G

        y[i] = y_pred
        e[i] = e_temp
        w[i,:] = w_temp[:,0]

    return y,e,w

def mse(ip,op,w,delay = 0):

    ip = ip[:,0]
    op = op[:,0]
    len_w = len(w)
    y = sg.lfilter(w,1,ip)
    if delay == 0:
        e = op - y[len_w:]
    else :
        e = op - y[len_w:-delay+1]
    mse = np.average(e**2)

    return mse,e

def wt_tracks(w):

    a,b = w.shape
    col = ['iterations']+[str(i) for i in range(b)]
    x_vals = np.expand_dims(np.arange(a),axis = 1)
    df = pd.DataFrame(columns =col,data = np.concatenate((x_vals,w),axis = 1))
    df.plot(x = 'iterations',y = col[1:])

def plot3d(kern_size_arr,step_size_arr,mse):

    kern_size_arr = kern_size_arr[1:]
    mse = mse[1:,:]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
    X, Y = np.meshgrid(kern_size_arr, step_size_arr)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, mse, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# code for KLMS and QKLMS

def kern_2(ip_1,ip_2,kern_size):

    x = np.linalg.norm((ip_1 - ip_2))
    return np.exp(-(x**2)/kern_size**2)

def kern_3(centers,ip_1,kern_size):

    dist_list = np.linalg.norm((centers - ip_1),axis = 1)
    dist_list = dist_list**2
    val = np.exp((-1*dist_list)/(kern_size**2))
    return val

def KLMS_mse(ip,op,l,k_1,step_size = 0.1):

    w_len = l+1
    length = len(op)
    e = np.zeros((length,1))
    y = np.zeros((length,1))
    cnt = np.zeros((length,1))
    count = 0

    for step in tqdm(range(length)):

        count = count + 1
        cnt[step] = count

        ip_1 = ip[step:step+w_len]
        y_temp = 0

        for i in range(step):

            ip_2 = ip[i:(i+w_len)]
            G = kern_2(ip_1,ip_2,k_1)
            y_temp = y_temp+step_size*e[i]*G

        e_temp = op[step] - y_temp

        if step == 0:

            e_temp = op[step]

        e[step] = e_temp
        y[step] = y_temp

    return y,e,cnt

def QKLMS_mse(ip,op,delay,L,h,step_size,threshold):
    
    w_len  = L+1
    
    len_ip = ip.shape[0]
   
    cnt = np.zeros([w_len])
    
    count = w_len
    
    element = ip[0:w_len]
    center_list = np.array([element])
    #print(center_list.shape)
    alpha = np.array([step_size*op[w_len-delay]])
    #print(alpha.shape)
    result = np.zeros(1)
    
    length = len_ip - w_len
    
    for step in tqdm(range(length)):
        
        element = ip[step:step+w_len]
        #print(element.shape)
        
        if step == 0:
            
            e = np.zeros(length)
            y = np.zeros(length)
            w = np.zeros(len_ip)
            
            e_temp = op[step+w_len-delay]
            e[0] = e_temp
            result[0] = 0
        
        dist_list = np.linalg.norm((center_list - element),ord = 2,axis = 1)
        #print(dist_list)
        dist_min = np.min(dist_list)
        
        min_index = np.argmin(dist_list)
        
        a = step_size * e[step-1]
        
        if dist_min<=threshold:
            
            alpha[min_index] = alpha[min_index]+a
            
            y_temp = result[min_index] + a*np.exp(-h*dist_list[-1]**2)
            e_temp = op[step+w_len -delay] - y_temp
            #print('true')
            
        else:
            
            count = count + 1
            
            y_temp = 0
            
            for index in range(center_list.shape[0]):
                
                G = np.exp(-h*(dist_list[index])**2)
                y_temp = y_temp + alpha[index]*G
                
            result = np.hstack((result,y_temp))
            
            e_temp = op[step+w_len-delay] - y_temp
            
            center_list = np.vstack((center_list,element))
            alpha = np.hstack((alpha,a))
            
        y[step] = y_temp
        e[step] = e_temp
        cnt = np.hstack((cnt,count))
        
        #print('the alpha shape is ',alpha.shape)
        #print('the center shape is ',center_list.shape)
    
    print('the number of voronoi cells are {0}'.format(count))
        
    return y,e,cnt



def KLMS_mse_2(ip,op,l,k_1=1,step_size = 0.1):

    w_len = l+1
    length = len(op)
    e = np.zeros((length,1))
    y = np.zeros((length,1))
    cnt = np.zeros((length,1))
    count = 1
    cnt[0] = count
    y[0] = 0
    e[0] = op[0]
    centers = np.array(ip[0:w_len].T)

    for step in range(1,length):

        ip_1 = ip[step:step+w_len].T
        G = kern_3(centers[0:step],ip_1,k_1)

        y[step] = (step_size*G)@e[0:step]

        e[step] = op[step] - y[step]

        centers = np.vstack((centers,ip_1))

        count += 1
        cnt[step] = count
    
    return y,e,centers,cnt

def QKLMS_mse_2(ip,op,l,k_1=1,step_size = 0.1,threshold = 0.1):

    w_len = l+1
    length = len(op)
    e = np.zeros((length,1))
    y = np.zeros((length,1))
    cnt = np.zeros((length,1))
    count = 1
    cnt[0] = count
    y[0] = 0
    e[0] = op[0]
    centers = np.array(ip[0:w_len].T)
    alpha = np.array(step_size*op[0])
    alpha = np.expand_dims(alpha,axis = 1)
    #print(alpha.shape)
    #print(centers.shape)

    for step in range(1,length):
        
        ip_1 = ip[step:step+w_len].T
        G = kern_3(centers,ip_1,k_1)
        #print(G.shape)
        y[step] = G@alpha
        e[step] = op[step] - y[step]

        dist_list = np.linalg.norm((centers - ip_1),axis = 1)
        dist_min = np.min(dist_list)
        min_i = np.argmin(dist_list)

        if dist_min <= threshold:

            alpha[min_i] = alpha[min_i]+step_size*e[step].item()
        
        else:

            count +=1
            centers = np.vstack((centers,ip_1))
            alpha = np.vstack((alpha,step_size*e[step].item()))

        cnt[step] = count

    return y,e,centers,alpha,cnt

# KLMS and QKLMS with MCC loss

def KLMS_mcc_2(ip,op,l,k_1=1,step_size=0.1,kern_size = 1):

    w_len = l+1
    length = len(op)
    e = np.zeros((length,1))
    y = np.zeros((length,1))
    cnt = np.zeros((length,1))
    count = 1
    cnt[0] = count
    y[0] = 0
    e[0] = op[0]
    centers = np.array(ip[0:w_len].T)
    step_size = step_size/(kern_size**2)

    for step in range(1,length):

        ip_1 = ip[step:step+w_len].T
        G = kern_3(centers[0:step],ip_1,k_1)

        y[step] = step_size*G@(e[0:step]*kern_1(e[0:step],kern_size))

        e[step] = op[step] - y[step]

        centers = np.vstack((centers,ip_1))

        count += 1
        cnt[step] = count
    
    return y,e,centers,cnt

def QKLMS_mcc_2(ip,op,l,k_1=1,step_size = 0.1,threshold = 0.1,kern_size = 1):

    w_len = l+1
    length = len(op)
    e = np.zeros((length,1))
    y = np.zeros((length,1))
    cnt = np.zeros((length,1))
    count = 1
    cnt[0] = count
    y[0] = 0
    e[0] = op[0]
    step_size = step_size/(kern_size**2)
    centers = np.array(ip[0:w_len].T)
    alpha = np.array(step_size*op[0]*kern_1(op[0],kern_size))
    alpha = np.expand_dims(alpha,axis = 1)
    #print(alpha.shape)
    #print(centers.shape)

    for step in range(1,length):
        
        ip_1 = ip[step:step+w_len].T
        G = kern_3(centers,ip_1,k_1)
        #print(G.shape)
        y[step] = G@alpha
        e[step] = op[step] - y[step]

        dist_list = np.linalg.norm((centers - ip_1),axis = 1)
        dist_min = np.min(dist_list)
        min_i = np.argmin(dist_list)

        if dist_min <= threshold:

            alpha[min_i] = alpha[min_i]+step_size*e[step].item()*kern_1(e[step],kern_size)
        
        else:

            count +=1
            centers = np.vstack((centers,ip_1))
            alpha = np.vstack((alpha,step_size*e[step].item()*kern_1(e[step],kern_size)))

        cnt[step] = count

    return y,e,centers,alpha,cnt

def klms_mseror(ip,op,l,e,centers,k_1,step_size):

    w_len = l+1
    length = len(op)
    y_pred = np.zeros((length,1))

    for step in range(length):

        ip_1 = ip[step:step+w_len].T
        G = kern_3(centers,ip_1,k_1)
        y_pred[step] = (step_size*G)@e

    erors = op - y_pred
    mse = np.average(erors**2)
    return mse ,erors

def klms_pred(ip,op,l,e,centers,k_1,step_size):

    w_len = l+1
    length = len(op)
    y_pred = np.zeros((length,1))

    for step in range(length):

        ip_1 = ip[step:step+w_len].T
        G = kern_3(centers,ip_1,k_1)
        y_pred[step] = (step_size*G)@e
    
    return y_pred

def lms_trac_gen(ip,op,w,init_pos,threshold):
    
    op = op[:,0]
    w_len = len(w)
    e = 0
    cnt = 0
        
    ip_1 = ip[init_pos - w_len:init_pos]
    ip_1 = ip_1[::-1]
    ip_1 = ip_1[:,0].tolist()
    
    y_track = []
    e_arr = []
    i = init_pos
    
    while (abs(e) < threshold):
        
        ip_2 = np.array(ip_1[:w_len])
        y = np.sum(ip_2*w)
        e = op[i-w_len] - y.item()
        e = e.item()
        y_track.append(y)
        ip_1.insert(0,y.item())
        i = i+1
        cnt+=1
        e_arr.append(e)
        
    return y_track,cnt,e_arr

def mse_lms_ahead(ip,op,w,ahead):
    
    w_len = len(w)
    length_op = len(op)
    e = np.zeros((length_op-ahead,1))

    for i in tqdm(range(length_op - ahead)):
        
        ip_1 = ip[i:i+w_len]
        ip_1 = ip[::-1]
        ip_1 = ip_1[:,0].tolist()
        
        for k in range(ahead):
            
            ip_2 = np.array(ip_1[:w_len])
            y = np.sum(ip_2*w)
            ip_1.insert(0,y.item())
            
        e[i] = op[i+ahead-1] - y
        
    
    mse = np.average(e**2)
    return mse

def mse_klms_ahead(ip,op,l,e,centers,k_1,step_size,ahead):

    w_len = l+1
    length = len(op)
    erors = np.zeros((length-ahead,1))

    for step in tqdm(range(length-ahead)):

        ip_1 = ip[step:step+w_len]
        ip_1 = ip_1[:,0].tolist()

        for k in range(ahead):

            ip_2 = np.array(ip_1[:w_len])
            ip_2 = np.expand_dims(ip_2,axis = 0)
            G = kern_3(centers,ip_2,k_1)
            y = (step_size*G)@e
            ip_1.insert(0,y.item())

        erors[step] = op[step+ahead-1] - y

    mse = np.average(erors**2)
    return mse

def klms_trac_gen(ip,op,l,err,centers,k_1,step_size,init_pos,threshold):

    w_len = l+1
    cnt = 0
    e = 0

    ip_1 = ip[init_pos - w_len:init_pos]
    ip_1 = ip_1[:,0].tolist()
    e_arr = []
    y_track = []

    while abs(e) < threshold :

        ip_2 = np.array(ip_1[:w_len])
        ip_2 = np.expand_dims(ip_2,axis = 0)
        #print(ip_2.shape)
        G = kern_3(centers,ip_2,k_1)
        y = (step_size*G)@err
        ip_1.insert(0,y.item())

        e = op[init_pos-w_len] - y

        e_arr.append(e)

        y_track.append(y)

        cnt+=1
        init_pos+=1
    
    return y_track,cnt,e_arr




