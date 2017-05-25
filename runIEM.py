import numpy as np
import labtools as lt

# see next cell for sample function call

# will run an ORIENTATION IEM with cos^num_chans-1 basis functions (by orientation i mean a 0-179 degree space)
# note that it will scale each row of the design matrix X in the forward model to have a range from 0-1
# assumes that input data are in our typical format of trials x features
# note - i think this can be cleaned up a bit by importing specific numpy functions at the top here, but i 
# opted instead to import them all as np...

def runIEM(trn_data, tst_data, trn_labels, tst_labels, num_chans):
    
    # x-values for evaluating functions. 
    x=np.arange(0, np.pi, np.pi/180) 

    # make the full basis set
    # first set up a matrix of zeros to store the basis functions
    basis_set = np.zeros((180,num_chans))

    # channel centers - in this case ) step pi/num_chans to pi
    chan_center = np.arange(0, np.pi, np.pi/num_chans) 

    # make the basis functions
    for c in range(num_chans):
         basis_set[:,c]=lt.make_basis_function(x,chan_center[c],num_chans)

    # now make the stimulus mask
    stim_mask = np.zeros((len(trn_labels),180))

    # loop over trials and put in a '1' at the feature on each trial
    for t in range(len(trn_labels)):
        stim_mask[t,trn_labels[t]-1]=1    #use the -1 here because the orientations are from 1:180 and we want 0 to 179

    # now loop to compute a 180 point tuning function (can also get this by multiplying by the basis set, but i 
    # like this looping approach better because i think its slightly more accurate and has less interpolation)
    sz=tst_data.shape
    chan_response = np.zeros((sz[0],180))  # to store the channel response functions on each iteration
    steps = np.int16(180/num_chans)        # convert to int from float so that we can use for indexing arrays 

    # Will transpose the tst data once outside the loop cause we need it in that form for the IEM
    tst_transposed = tst_data.T

    # start the loop - i.e. will iterate 20 times with 9 basis functions to fill the space between 0:20:180
    for basis_centers in range(steps):

        # use these to fill up the 180 point recon across loops 
        indices = np.arange(basis_centers, 180, steps)

        # roll (shift) the basis functions across orientation space on each iteration to estimate
        # each point in the space. 
        tmp_basis_set = np.roll(basis_set, basis_centers, axis=0)

        # matrix multiplication (not element by element) to get the design matrix
        # using @ operator instead of numpy.dot for readability. 
        X = stim_mask @ tmp_basis_set

        # then scale X so that the set of regressors on each trial has a range from 0 to 1...this will use my function
        # defined above, but not the need to transpose because it works down rows by default (and then transpose back)
        # could also include an axis arg to the scaleData function.  
        X = (np.transpose(lt.scaleData(np.transpose(X))))   

        # then do the weight estimation using the more numerically stable 'solve' 
        w = np.linalg.solve(X.T @ X, X.T) @ trn_data

        # compute IEM
        tmp = (np.linalg.solve(w @ w.T, w) @ tst_transposed).T

        # stick in the data from the basis functions that are centered at their current positions. 
        # there must be an elegant to do this with slicing...
        for c in range(num_chans):
            chan_response[:,indices[c]]=tmp[:,c]


    # then after the loop, roll (shift) the chan response on each trial to a common center
    # recall that we defined sz as the size of the wmDat above
    cent = 90
    cent_chan_response = np.zeros((sz[0],180))
    for c in range(sz[0]):
        cent_chan_response[c,:] = np.roll(chan_response[c,:], cent-tst_labels[c])

    return cent_chan_response