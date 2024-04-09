
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def PeronaMalik_Smoother(image,K,LAMBDA,gfunction,nIterations,convert_to_grayscale=True): # REFERENCE: https://github.com/krishanuskr/ImageRestoration/blob/master/imagerestoration.py
    
    print ('Image shape: ', image.shape)

    
    if convert_to_grayscale == True:
        try:
            image = np.mean(image,axis=2)
        except IndexError:
            raise IndexError('Image is already 2D, so is assumed grayscale already. Check shape.')

    
    #Get number of color channels of image [1 for grayscale, 3 for RGB, 4 for RGBA]
    #Assuming the image shape is heigth x width for grayscale,
    #or heigth x width x Nchannels for color. But not expecting more than 3 dimensions.
    nChannels = 1 if image.ndim == 2 else image.shape[2]
    print ('nChannels',nChannels)
    
    #In the case of a grayscale image, to make things easier later, just make the grayscale image have a 3rd axis of length 1
    if nChannels == 1:
        image = np.expand_dims(image,axis=2)
    
    #4D Container array of all iterations of image diffusion
    image_stack = np.expand_dims(image,axis=0)
    
    #Do nIterations of diffusion:
    for i in range(nIterations):
        if i % 10 == 0:
            print ('Starting iteration {0} of {1}'.format(i,nIterations))
        image_t = np.zeros(image.shape)
        for channel in range(nChannels):
            temp = image_stack[-1][:,:,channel]
            
            #Following equation 8 in paper: calculate nearest neighbor differences to approximate gradient of image intensity
            vert_diff = np.diff(temp,axis=0)
            horiz_diff = np.diff(temp,axis=1)
            nanrow = np.expand_dims(np.nan*np.ones(vert_diff.shape[1]),axis=0)
            nancol = np.expand_dims(np.nan*np.ones(horiz_diff.shape[0]),axis=0).T
            grad_S = np.vstack((vert_diff,nanrow)) #NaN on bottom row
            grad_N = np.vstack((nanrow,-vert_diff)) #NaN on top row, and negated diffs since going opposite direction from np.diff() default
            grad_E = np.hstack((horiz_diff,nancol)) #NaN on right column
            grad_W = np.hstack((nancol,-horiz_diff)) #NaN on left column, and negated diffs since going opposite direction from np.diff() default
            
            #Following equation 10 in paper: calculate conduction coefficients
            #Technically, the coefficients should be more appropriately be evaluated at the halfway point between pixels, not at the pixels themselves.
            #But this is more complicated for approximately same results (according to authors). So use the same values for gradients as above.
            if gfunction == 'Exponential':
                c_S = np.exp(-(grad_S/K)**2)
                c_N = np.exp(-(grad_N/K)**2)
                c_E = np.exp(-(grad_E/K)**2)
                c_W = np.exp(-(grad_W/K)**2)
                
            if gfunction == 'Cauchy':
                c_S = 1./(1.+(grad_S/K)**2)
                c_N = 1./(1.+(grad_N/K)**2)
                c_E = 1./(1.+(grad_E/K)**2)
                c_W = 1./(1.+(grad_W/K)**2)
            
            #Examine the conduction coefficients:
            '''if i == 7:
                plt.figure()
                plt.title('c_S Diffusion Coefficients\nChannel {}'.format(channel),fontsize=30)
                plt.tick_params(labelsize=0)
                plt.imshow(c_S,interpolation='none')
                cb=plt.colorbar()
                cb.ax.tick_params(labelsize=20) 
                plt.show()
#                print np.nanmin(c_S), np.nanmax(c_S), np.nanmean(c_S)
#                print c_S  '''          
            
            
            #Following equation 7 in paper: Update the image using the diffusion equation:
            temp2 = temp + LAMBDA*(c_S*grad_S + c_N*grad_N + c_E*grad_E + c_W*grad_W)
            
            #Reset boundaries since the paper uses adiabatic boundary conditions and above steps intentionally set boudnaries to NaNs
            temp2[:,0] = temp[:,0] #Left edge
            temp2[:,-1] = temp[:,-1] #Right edge
            temp2[-1,:] = temp[-1,:] #Bottom edge
            temp2[0,:] = temp[0,:] #Top edge
            
            #Update this channel of the image at this time step
            image_t[:,:,channel] = temp2

        image_t = np.expand_dims(image_t,axis=0)
        image_stack = np.append(image_stack,image_t,axis=0)    
    
    
    #image_stack is stack of all iterations.
    #iteration 0 is original image, iteration -1 is final image.
    #Intermediate images are also returned for visualization and diagnostics
    return image_stack

if __name__ == '__main__':

    #Load test image
    image = mpimg.imread('Material_P1/T2.png')
    print (image.shape)
        
    #Set algorithm parameters
    K = [1,5,8,15]
    LAMBDA = [0.01]
    nIterations = [45] #100
    gfunction = 'Exponential' #'Cauchy'


    
    #plt.title('Original',fontsize=30)
    #plt.imshow(image,interpolation='None',cmap='gray')
    #plt.show()

    #Add noise to image
    noise = np.random.normal(0,.01,image.shape)
    #image = image + noise

    #Add rician noise to the image
    noise = np.random.rayleigh(0.05,image.shape)
    image = image + noise
    plt.title('Original + Noise',fontsize=30)
    plt.imshow(image,interpolation='None',cmap='gray')
    #plt.show()
    #Plot grayscale example
    #Plot in the same plot the images with different K and Lambda values

    # Create a 6x4 grid plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 18))
    axes = axes.flatten()
    cont = 0
    for  k in K:
        for l in LAMBDA:
            for iteraciones in nIterations:
                for funcion in ['Exponential','Cauchy']:
                    
                    PMimage_stack = PeronaMalik_Smoother(image,k,l,funcion,iteraciones,convert_to_grayscale=False)
                    #I want to save each image into the axes array
                    axes[cont].set_title('K={0}, Lambda={1}, Iter={2}, Func={3}'.format(k,l,iteraciones,funcion),fontsize=10)
                    #I want it to be grayscale

                    axes[cont].imshow(np.squeeze(PMimage_stack[-1]),interpolation='None',cmap='gray')
                    axes[cont].axis('off')
                    cont+=1
                    print("ITERATION: ",cont)
    # Adjust layout
    plt.tight_layout()
    plt.show()