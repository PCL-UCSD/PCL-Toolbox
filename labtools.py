import numpy as np
import numpy.matlib
import scipy.io as sio

# scale the data to the desired range (default is 0:1)...will work on matrices down columns (i.e. matlab style)
def scaleData(data, minVal=0, maxVal=1):
    data = np.asanyarray(data)  # this will convert lists/tuples etc to array...if data already an array then no effect
    sz = data.shape
    minData = np.amin(data, axis=0)
    maxData = np.amax(data, axis=0)  
    # then scale
    scaled_data = data - np.matlib.repmat(minData,sz[0],1)
    scaled_data = np.multiply(np.divide(scaled_data, np.matlib.repmat(np.ptp(scaled_data,axis=0),sz[0],1)),(np.matlib.repmat(maxVal-minVal,sz[0],sz[1])))
    scaled_data = scaled_data + np.matlib.repmat(minVal,sz[0],sz[1])
    return scaled_data

# zscore and return mean and std as well (matrix or column vector input with row == samples/trials, columns==feature)
def zscore(data):
    data = np.asanyarray(data)
    sz=data.shape
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data,axis=0)
    zdata = np.divide((data-np.matlib.repmat(mean_data,sz[0],1)),np.matlib.repmat(std_data, sz[0], 1))
    return zdata, mean_data, std_data

def readMatData(file_name_in,field_name,file_name_out):
    # will read in data from "file_name_in.mat" and then return the data in field_name 
    # as a np.array and then will save a npy file if "file_name_out" is 1 (default 0)
    print("reading and converting:",file_name_in)
    # first load the data and get the desired field
    tmp = sio.loadmat(file_name_in)
    data = np.array(tmp[field_name])
    # then convert the data to numpy friendly load/save format
    np.save(file_name_out, data)
    return data

def binOrientations(labels, num_bins):
    # will take a vector of trial labels (orientations in this case) and bin them into
    # num_bins discrete categories. Note that num_bins must go into 180 evenly for this to 
    # work right...can modify to handle other cases as needed. 

    # do a quick check to see if num_bins goes evenly into 180
    if 180%num_bins!=0:
        print("error num_bins must go evenly into 180")
        return()

    # sort the trial labels into bins
    sz = labels.shape
    binned_labels = np.zeros(sz[0])

    # do the sorting
    bc = np.arange(180/num_bins, 180+180/num_bins, 180/num_bins)  
    for i in range(sz[0]):
        for j in range(len(bc)):
            if j==0:
                if labels[i] <= bc[j]:
                    binned_labels[i]=j
                    break
            else:
                if (labels[i] > bc[j-1]) & (labels[i] <= bc[j]):
                    binned_labels[i]=j
    
    return binned_labels
    
# make basis functions. input in radians. 
def make_basis_function(x,mu,num_chans):
    basis_func=np.power(np.cos(np.subtract(x, mu)),np.subtract(num_chans,(num_chans%2)))
    return basis_func