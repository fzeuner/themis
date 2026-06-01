"""
Line fitting module using lmfit VoigtModel.

Similar to atlas-fit's line fitting but using lmfit instead of scipy.
"""

import numpy as np
from lmfit.models import VoigtModel, ConstantModel


class LineFit:
    """
    Fit spectral line cores using Voigt profile with lmfit.
    
    Parameters
    ----------
    x : np.array
        Wavelength or pixel coordinates
    y : np.array
        Intensity values (absorption lines should be negative or inverted)
    center_guess : float
        Initial guess for line center position
    error_threshold : float, optional
        Error threshold for fit quality (default: 0.5)
    """
    
    def __init__(self, x: np.array, y: np.array, center_guess: float, error_threshold: float = 0.5):
        self.x = np.asarray(x, dtype='float64')
        self.y = np.asarray(y, dtype='float64')
        self.center_guess = center_guess
        self.error_threshold = error_threshold
        
        self.max_location = None
        self.fit_result = None
        self.model = None
        self._fitted = False
    
    def run(self):
        """
        Run the Voigt profile fit.
        
        Raises
        ------
        RuntimeError
            If fit fails or does not meet quality threshold
        """
        # Create composite model: Voigt (absorption) + Constant (continuum)
        voigt = VoigtModel(prefix='voigt_')
        const = ConstantModel(prefix='const_')
        self.model = voigt + const
        
        # Set initial parameters with constraints
        params = self.model.make_params()
        
        # Voigt parameters - for absorption lines, amplitude should be negative
        params['voigt_amplitude'].value = -0.1
        params['voigt_amplitude'].min = -1.1
        params['voigt_amplitude'].max = 0.0
        
        # Constrain center to be within the data range
        x_range = self.x.max() - self.x.min()
        params['voigt_center'].value = self.center_guess
        params['voigt_center'].min = self.center_guess -0.01
        params['voigt_center'].max = self.center_guess +0.01
        
        params['voigt_sigma'].value = 0.03
        params['voigt_sigma'].min = 0.001
        params['voigt_sigma'].max = 0.5
        
        params['voigt_gamma'].value = 1.0
        params['voigt_gamma'].min = 0.0
        params['voigt_gamma'].max = 5.0
        
        # Constant parameter - initialize at max of spectrum (continuum level)
        params['const_c'].value = 1.1
        params['const_c'].min = 0.8
        params['const_c'].max = 1.5
        
        # Fit the model directly to the absorption line 
        try:
            result = self.model.fit(self.y, params, x=self.x)
            self.fit_result = result
            
            # Check fit quality
            if not result.success:
                raise RuntimeError(f"Fit did not converge")
            
            # Check if center is within reasonable bounds (within the data range)
            fitted_center = result.params['voigt_center'].value
            if fitted_center < self.x.min() or fitted_center > self.x.max():
                raise RuntimeError(f"Fitted center {fitted_center:.4f} outside data range [{self.x.min():.4f}, {self.x.max():.4f}]")
            
            # Extract line center (minimum of absorption line)
            self.max_location = fitted_center
            self._fitted = True
            
        except Exception as e:
            raise RuntimeError(f"Line fitting failed: {e}")
    
    def eval(self, x=None):
        """Evaluate the fitted model at given x values."""
        if not self._fitted or self.fit_result is None:
            return None
        if x is None:
            x = self.x
        return self.model.eval(self.fit_result.params, x=x)
    
    @property
    def fitted_profile(self):
        """Return the fitted profile at the original x values."""
        if not self._fitted or self.fit_result is None:
            return None
        # Use the best_fit from lmfit result
        if hasattr(self.fit_result, 'best_fit'):
            return self.fit_result.best_fit
        # Fallback: evaluate manually
        return self.eval(self.x)
    
    @property
    def parameters(self):
        """Return fitted parameters."""
        if not self._fitted or self.fit_result is None:
            return None
        return self.fit_result.params
