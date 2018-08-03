import numpy as np
from astropy.io import fits
import cv2
from skimage.filters import sobel

class PeakCount():
    """ Peak counting model."""
    
    def __init__(self, peak_counting_method = 'original',
                 resolution_arcmin = 1., 
                 ng = None, shape_noise=False,
                 bins = np.arange(-0.03,0.19,0.01),
                 covariance ='varying',
                 fiducial_values = (0.26,0.8),
                 seed = 42):
        self.peak_counting_method = peak_counting_method
        self.resolution_arcmin = resolution_arcmin
        self.bins = bins
        self.covariance = covariance
        self.fiducial_values = fiducial_values
        self.ng = ng
        self.shape_noise = shape_noise
        self.rng = np.random.RandomState(seed=seed)
        assert self.covariance in ['fixed','varying']
        assert self.peak_counting_method in ['original',
                                             'laplace_v1',
                                             'laplace_v2',
                                             'roberts_cross',
                                             'sobel']
        
        
    def fit(self, fn_list, omega_m_list, sigma_8_list):
        """Calculate the mean peak counts and covariances."""
        self.peak_list = {}
        for fn,o,s in zip(fn_list, omega_m_list, sigma_8_list):
            im = self.load_im(fn)
            hp = self.peak_count(im)
            if (o,s) in self.peak_list:
                self.peak_list[(o,s)].append(hp)
            else:
                self.peak_list[(o,s)] = [hp]                

        self.mean_peaks = {}
        for k,v in self.peak_list.iteritems():
            self.mean_peaks[k] = np.mean(v,axis=0)
            
        self.inv_cov = {}
        for k,v in self.peak_list.iteritems():
            if self.covariance == 'varying':
                vm = np.vstack(v).T
                self.inv_cov[k] = np.linalg.pinv(np.cov(vm))
            elif self.covariance == 'fixed':
                vm = np.vstack(self.peak_list[self.fiducial_values]).T
                self.inv_cov[k] = np.linalg.pinv(np.cov(vm))
            

    def find_peaks(self, im):
        """Find peaks in bw image."""
        p =  im[1:-1,1:-1]>im[:-2,:-2]  # top left
        p &= im[1:-1,1:-1]>im[:-2,1:-1]  # top center  
        p &= im[1:-1,1:-1]>im[:-2,2:]  # top right
        p &= im[1:-1,1:-1]>im[1:-1,:-2]  # center left 
        p &= im[1:-1,1:-1]>im[1:-1,2:]  # center right 
        p &= im[1:-1,1:-1]>im[2:,:-2]  # bottom left
        p &= im[1:-1,1:-1]>im[2:,1:-1]  # bottom center
        p &= im[1:-1,1:-1]>im[2:,2:]   # bottom right
        return p


    def load_im(self, fn):
        """Load image and degrade it."""
        f = fits.open(fn) 
        im = f[0].data.astype('float64')
        im[np.isnan(im)] = 0 # replace nans
        im = np.clip(im,-1,1)  # clip to meaningful interval
        s = int(round(1024.*(0.2/float(self.resolution_arcmin))))
        im = cv2.resize(im,(s,s), interpolation=cv2.INTER_AREA)
        
        if self.shape_noise:
            # shape noise formula from Matilla et al
            sige = 0.4
            A = self.resolution_arcmin**2.  # pixel area
            sigpix = sige / np.sqrt(2 * A * self.ng)  # final noise scatter
            
            # add shape noise to image
            im += self.rng.normal(loc=0, scale=sigpix, size=im.shape)
            
        return im 


    def predict(self, fns):
        """Predict on image with peak counts."""
        # im = self.load_im(fn)
        # hp = self.peak_count(im)
        ims = [self.load_im(fn) for fn in fns]
        hps = [self.peak_count(im) for im in ims]
        hp = np.mean(hps, axis=0)
        
        ks = sorted(self.mean_peaks.keys())
        chis = []
        for k in ks:
            d = np.matrix((hp - self.mean_peaks[k]))
            chis.append( d * self.inv_cov[k] * d.T )

        i = np.argmin(chis)
        po = ks[i][0]
        ps = ks[i][1]

        return po,ps


    def peak_count(self, im):
        """Peak counting statistics"""
        peaks = self.find_peaks(im)  # find peaks  
        
        # get the values for peaks
        if self.peak_counting_method == 'original':
            vals = im[1:-1,1:-1][peaks]
        elif self.peak_counting_method == 'laplace_v1':
            vals = self.laplace_v1(im)[peaks]
        elif self.peak_counting_method == 'laplace_v2':
            vals = self.laplace_v2(im)[peaks]
        elif self.peak_counting_method == 'roberts_cross':
            vals = self.roberts_cross(im)[peaks]
        elif self.peak_counting_method == 'sobel':
            vals = self.sobel_peak(im)[peaks]

        # make histogram
        hp = np.histogram(vals, bins=self.bins)[0]
        return hp
    
    
    def laplace_v1(self, im):
        """Characterize peaks with laplace kernel in image."""
        p =  (10./3)*im[1:-1,1:-1]
        p -= (1./6.)*im[:-2,:-2]  # top left
        p -= (2./3.)*im[:-2,1:-1]  # top center  
        p -= (1./6.)*im[:-2,2:]  # top right
        p -= (2./3.)*im[1:-1,:-2]  # center left 
        p -= (2./3.)*im[1:-1,2:]  # center right 
        p -= (1./6.)*im[2:,:-2]  # bottom left
        p -= (2./3.)*im[2:,1:-1]  # bottom center
        p -= (1./6.)*im[2:,2:]   # bottom right
        return p
    
    
    def laplace_v2(self, im):
        """Characterize peaks with laplace kernel in image."""
        p =  4*im[1:-1,1:-1]
        p -= im[:-2,1:-1]  # top center  
        p -= im[1:-1,:-2]  # center left 
        p -= im[1:-1,2:]  # center right 
        p -= im[2:,1:-1]  # bottom center
        return p
    
    
    def roberts_cross(self, im):
        """Evaluate Robert's cross gradient magnitude."""
        p0  =  (im[1:-1,1:-1] - im[:-2,:-2])**2  # top left
        p0  += (im[:-2,1:-1] - im[1:-1,:-2])**2  # top left
        p0 = np.sqrt(p0)
        p1  =  (im[:-2,1:-1] - im[1:-1,2:])**2  # top right
        p1  += (im[1:-1,1:-1] - im[:-2,2:])**2  # top right
        p1 = np.sqrt(p1)
        p2  =  (im[1:-1,:-2] - im[2:,1:-1])**2  # bottom left
        p2  += (im[2:,:-2] - im[1:-1,1:-1])**2  # bottom left
        p2 = np.sqrt(p2)
        p3  =  (im[1:-1,1:-1] - im[2:,2:])**2  # bottom right
        p3  += (im[2:,1:-1] - im[1:-1,2:])**2  # bottom right
        p3 = np.sqrt(p3)
        return (p0+p1+p2+p3)
    
    
    def sobel_peak(self, im):
        """Evaluate magnitude around a peak with Sobel filter."""
        sim = sobel(im)
        p =  sim[:-2,:-2]  # top left
        p += sim[:-2,1:-1]  # top center  
        p += sim[:-2,2:]  # top right
        p += sim[1:-1,:-2]  # center left 
        p += sim[1:-1,2:]  # center right 
        p += sim[2:,:-2]  # bottom left
        p += sim[2:,1:-1]  # bottom center
        p += sim[2:,2:]   # bottom right
        return p/8.
    
    
def rmse(y,y_pred):
    return np.sqrt(((np.array(y)-np.array(y_pred))**2).mean())


def rmse_bootstrap(y, y_pred, m = 10000):
    """Return bootstrap mean and std error."""
    e = []
    for j in range(m):
        idx = np.arange(len(y))
        sel = np.random.choice(idx, len(idx), replace=True)
        e.append(rmse(y[sel],y_pred[sel]))
    return rmse(y,y_pred),np.std(e)