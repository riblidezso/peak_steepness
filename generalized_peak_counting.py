import numpy as np
from astropy.io import fits
import cv2

class PeakCount():
    """ Peak counting model."""
    
    def __init__(self, peak_counting_method = 'original',
                 resolution_arcmin = 1., 
                 bins = np.arange(-0.03,0.19,0.01),
                 covariance ='varying',
                 fiducial_values = (0.26,0.8)):
        self.peak_counting_method = peak_counting_method
        self.resolution_arcmin = resolution_arcmin
        self.bins = bins
        self.covariance = covariance
        self.fiducial_values = fiducial_values
        assert self.covariance in ['fixed','varying']
        assert self.peak_counting_method in ['original',
                                             'laplace_v1',
                                             'laplace_v2',
                                             'roberts_cross']
        
        
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
        im = np.clip(im,-1,1)
        s = int(round(1024.*(0.2/float(self.resolution_arcmin))))
        im = cv2.resize(im,(s,s), interpolation=cv2.INTER_AREA)
        return im 


    def predict(self, fn):
        """Predict on image with peak counts."""
        im = self.load_im(fn)
        hp = self.peak_count(im)

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
        if self.peak_counting_method == 'original':
            return self.peak_count_original(im) 
        elif self.peak_counting_method == 'laplace_v1':
            return self.peak_count_laplace__v1(im)
        elif self.peak_counting_method == 'laplace_v2':
            return self.peak_count_laplace_v2(im)
        elif self.peak_counting_method == 'roberts_cross':
            return self.peak_count_roberts_cross(im) 

        return hp
    
    
    def peak_count_original(self,im):
        """Original peak counting"""
        p = im[1:-1,1:-1][self.find_peaks(im)]
        hp = np.histogram(p, bins=self.bins)[0]
        return hp
    
    
    def peak_count_laplace_v1(self,im):
        """Peak steepness counting with laplace kernel"""
        peaks = self.find_peaks(im)
        vals = self.mexican_hat(im)[peaks]
        hp = np.histogram(vals, bins=self.bins)[0]
        return hp
    
    
    def laplace_v1(self, im):
        """Characterize peaks with laplace kernel in image."""
        p =  4*im[1:-1,1:-1]
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
    
    
    def peak_count_laplace_v2(self,im):
        """Peak steepness counting with laplace kernel"""
        peaks = self.find_peaks(im)
        vals = self.mexican_hat_v1(im)[peaks]
        hp = np.histogram(vals, bins=self.bins)[0]
        return hp
    
    
    def laplace_v2(self, im):
        """Characterize peaks with laplace kernel in image."""
        p =  4*im[1:-1,1:-1]
        p -= im[:-2,1:-1]  # top center  
        p -= im[1:-1,:-2]  # center left 
        p -= im[1:-1,2:]  # center right 
        p -= im[2:,1:-1]  # bottom center
        return p
    
    
    def peak_count_roberts_cross(self,im):
        """Peak steepness counting with Roberts cross"""
        peaks = self.find_peaks(im)
        vals = self.roberts_cross(im)[peaks]
        hp = np.histogram(vals, bins=self.bins)[0]
        return hp
    
    
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
    
def rmse(y,y_pred):
    return np.sqrt(((np.array(y)-np.array(y_pred))**2).mean())