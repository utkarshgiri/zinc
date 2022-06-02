import fire
import json
import numpy
import pathlib
import logging

from scipy import interpolate
from classylss import binding
from rich.logging import RichHandler
from lenstools.simulations import Gadget2SnapshotDE
from astropy.units import Mpc, km , m, s, Quantity
from astropy.utils.misc import JsonCustomEncoder
import astropy.units as u

from typing import Union

logging.basicConfig(level="NOTSET", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

#CLASS internal units are mpc

class Zinc:

    def __init__(self, init_config: Union[str, dict], voxels=512, fnl=0, seed=42, outfile='./gadgetIC', 
                scaling='growth_function', out_units='Mpch'):

        self.fnl = fnl
        self.seed = seed
        self.pixels = self.voxels = voxels
        self.scaling = scaling
        self.outfile = outfile
        self.out_units = out_units
        
        if isinstance(init_config, str) and pathlib.Path(init_config).exists():
            logger.debug('config is a file')
            if pathlib.Path(init_config).suffix == '.npy':
                self.config = numpy.load(init_config, allow_pickle=True).item()
            else:
                logger.debug('Assuming the provided init_confi option is a gadget file')
                snap = Gadget2SnapshotDE.open(init_config)
                self.config = snap.header

        elif isinstance(init_config, dict):
            self.config = init_config
        else:
            NotImplementedError
        logger.debug(self.config)
        self.hubble = Quantity(self.config['h'], km / (Mpc * s)) 

        self.kpc_over_h = u.def_unit("kpc/h",u.kpc/self.config["h"])
        self.Mpc_over_h = u.def_unit("Mpc/h",u.Mpc/self.config["h"])

        self.boxsize = self.config['box_size']
        self.redshift = self.config['redshift']
        
        num_particles = self.config['num_particles_total']
        self.config['num_particles_total'] = self.voxels**3
        self.config['num_particles_file'] = self.voxels**3
        self.config['num_particles_total_of_type'][1] = self.voxels**3
        self.config['num_particles_file_of_type'][1] = self.voxels**3
        self.config['masses'][1] = num_particles*self.config['masses'][1]/self.config['num_particles_total']
        self.config['num_particles_total_side'] = self.voxels
        logger.info('Gadget configs are: \n {}'.format(self.config))

        self.init_length_unit = self.boxsize.unit
        
        self.factor, self.out_length_unit = self.set_units()
        assert self.init_length_unit == self.out_length_unit

        self.class_parameters = self.gadgetToClass()

        logger.info('Class configs are: \n {}'.format(self.class_parameters))
        self.classy = binding.ClassEngine(self.class_parameters)
        self.background = binding.Background(self.classy)
        self.spectrum = binding.Spectra(self.classy)
        self.primordial = binding.Primordial(self.classy)
        
        self.mpcboxsize = self.boxsize.value / self.factor 

    def __call__(self):

        logger.info('Computing displaced position and velocity')
        position, velocity = self.positions(phik=self.deltak())


        position = numpy.array(position, dtype=numpy.float32) * self.factor * self.out_length_unit
        velocity = numpy.array(velocity, dtype=numpy.float32) * m / s
        logger.info('Writing out gadget file')

        snapshot = Gadget2SnapshotDE()
        snapshot.setPositions(position)
        snapshot.setVelocities(velocity)
        snapshot.setHeaderInfo(**self.config)
        snapshot.write(self.outfile)


        self.class_parameters.update({x:y for (x,y) in self.__dict__.items() if isinstance(y, float)})
        #with open(self.outfile.parent.joinpath('initial_configuration.json'), 'w+') as f:
        with open('./configuration/initial_configuration.json', 'w+') as f:
            json.dump(self.class_parameters, f)

    def gadgetToClass(self):
    
        default_config = {"output": "mTk mPk tCl lCl",
                      "A_s": 2.135e-09,
                      "n_s": 0.9624,
                      "h": 0.6711,
                      "P_k_max_1/Mpc": 1000.0,
                      "k_pivot": 0.05,
                      "omega_b": 0.022068,
                      "omega_cdm": 0.120925,
                      "z_pk": 2}

        default_config['h'] = self.config['h']
        default_config['z_pk'] = self.config['redshift']
        return default_config


    def primordial_powerspectrum(self):
        """ Returns an interpolator object which gives powerspectrum in the units of Mpc^3 """
        k = numpy.logspace(-7, 4, 10000)
        scale_invariant_powerpsectrum = self.class_parameters['A_s'] * (k/self.class_parameters['k_pivot'])**(self.class_parameters['n_s']-1.0)
        interpolator = interpolate.interp1d(k, (2*numpy.pi**2/k**3) * scale_invariant_powerpsectrum, fill_value='extrapolate')
        return interpolator


    def transfer_primordial_potential_to_cdm(self, field='d_tot'):
        """ A function which returns an interpolator for
        transfer function.
        Args:
            field (str): primordial field of interest.
                Default is 'phi'
            redshift (float): redshift of transfer function.
                Default is the configuration redshift
        Returns:
            An intterpolator that takes k value in 1/Mpc and
            returns the corresponding transfer function
            interpolator for the given field. """

        if self.scaling == 'trivial':
            logger.info('Trivial scaling. Using scale factor to scale the transfer function')
            scaling = 1./(1+self.redshift)
        elif self.scaling == 'growth_function':
            logger.info('Using growth factor to scale the transfer function')
            scaling = self.background.scale_independent_growth_factor(self.redshift)/self.background.scale_independent_growth_factor(0)
        else:
            NotImplementedError
        Tk = self.spectrum.get_transfer(z=0)

        Tk = interpolate.interp1d(Tk['k']*self.background.h, scaling*Tk[field], fill_value='extrapolate')
        return Tk


    def deltak(self):
        """ A function which samples phik modes on a 3D k-space grid
        to be used fro creating initial condition for N-body
        simulation

        Args:
            boxsize (float): box size in Mpc
            gridspacing (float): spacing of grids; boxsize/gridsize
            redshift (float): redshift at which the initial field is
                to be generated
            fnl (float): value of fnl. Default is 0

        Returns:
            The fourier space field phik """
        midpoint = int(self.pixels/2)
        k = numpy.zeros(shape=(self.pixels, self.pixels, midpoint+1))
        frequency = 2 * numpy.pi * numpy.fft.fftfreq(n=self.pixels, d=self.mpcboxsize/self.pixels)
        for i in range(self.pixels):
            for j in range(self.pixels):
                k[i,j,:] = numpy.sqrt(frequency[i]**2 + frequency[j]**2 + frequency[:(midpoint+1)]**2)

        powerspectrum = self.primordial_powerspectrum()(k)
        Tk = self.transfer_primordial_potential_to_cdm()(k) #in units of k

        sdev = numpy.sqrt(powerspectrum*(self.pixels**6/self.mpcboxsize**3)/2.0)
        real = numpy.random.normal(loc=0, scale=sdev, size=sdev.shape)
        imag = numpy.random.normal(loc=0, scale=sdev, size=sdev.shape)
        phik = real + 1j*imag

        phik[0,0,0] = 0;
        phik[midpoint,:,:] = 0; phik[:,midpoint,:] = 0; phik[:,:,midpoint] = 0

        #Adding non-gaussianity
        phik = self.hermitianize(phik)
        phi = numpy.fft.irfftn(phik)
        phi = phi + self.fnl*(phi**2 - numpy.mean(phi*phi))
        phik = numpy.fft.rfftn(phi)
        phik = self.hermitianize(phik)

        phik = Tk*phik
        phik = self.hermitianize(phik)

        return phik


    def displacement(self, deltak):
        """ Function to compute the displacement field from potential field
        Args:
            phik (ndarray): A 3D array containing the potential field in fourier space
            boxsize (float): Size of the box
            gridspacing (float): spacing of grids
            fnl (float): Value of fnl parameter. Defaults to 0
        Returns:
            displacement field vector for x, y and z.  """

        midpoint = int(self.pixels/2)
        frequency = 2 * numpy.pi * numpy.fft.fftfreq(n=self.pixels, d=self.mpcboxsize/self.pixels)
        kx, ky, kz = numpy.meshgrid(frequency, frequency, frequency[:midpoint+1])
        k = numpy.sqrt(kx**2 + ky**2 + kz**2)

        phik = numpy.divide(-deltak, k**2, out=numpy.zeros_like(deltak), where=k!=0)
        phikx = -1j*kx*phik; phiky = -1j*ky*phik; phikz = -1j*kz*phik

        psix = numpy.fft.irfftn(self.hermitianize(phikx))
        psiy = numpy.fft.irfftn(self.hermitianize(phiky))
        psiz = numpy.fft.irfftn(self.hermitianize(phikz))

        return (psix, psiy, psiz)


    def velocities(self, psix, psiy, psiz):
        """ Function to calculate velocity field from displacement vector fields
        Args:
            psix (ndarray): displacement field along x
            psiy (ndarray): displacement field along y
            psiz (ndarray): displacement field along z
        Returns:
            tuple containing velocities (vx, vy, vz) for all the particles """

        #f = self.cosmology.scale_independent_growth_factor_f(self.redshift)
        f = self.background.scale_independent_growth_rate(self.redshift)
        h = self.background.hubble_function(self.redshift)*3e5
        #h = self.classy.Hubble(self.redshift)*3e5
        a = 1./(1. + self.redshift)
        factor = f * a * h /numpy.sqrt(a)

        return (psix.flatten()*factor, psiy.flatten()*factor, psiz.flatten()*factor)


    def positions(self, phik):
        """ Function which takes potential field at a redshift and returns initial condition position and velocity
        Args:
            phik (ndarray): Potential field at initial
                redshift.
            boxsize (float): Size of the box
            gridspacing (float): Spacing of the grid
        Returns:
            (position, velocity) for the particles """

        psix, psiy, psiz = self.displacement(phik)
        gridspacing = self.mpcboxsize/self.pixels
        space = numpy.arange(start=0.0+gridspacing/2, stop=(self.mpcboxsize+gridspacing/2), step=gridspacing)
        x, y, z = numpy.meshgrid(space, space, space)
        x += psix; y += psiy; z += psiz

        position = numpy.column_stack([x.flatten(), y.flatten(), z.flatten()])
        velocity = numpy.column_stack(self.velocities(psix, psiy, psiz))
        position[position<0] += self.mpcboxsize
        position[position>self.mpcboxsize] -= self.mpcboxsize

        return position, velocity

    def hermitianize(self, x):
        """A function that self.hermitianizes the fourier array. A cleaner version of hermitianate.
        The logic is taken from:
        `https://github.com/nualamccullagh/zeldovich-bao/` """

        pixels = x.shape[0]; midpoint = int(pixels/2)
        for index in [0, midpoint]:
            x[midpoint+1:,1:,index]= numpy.conj(numpy.fliplr(numpy.flipud(x[1:midpoint,1:,index])))
            x[midpoint+1:,0,index] = numpy.conj(x[midpoint-1:0:-1,0,index])
            x[0,midpoint+1:,index] = numpy.conj(x[0,midpoint-1:0:-1,index])
            x[midpoint,midpoint+1:,index] = numpy.conj(x[midpoint,midpoint-1:0:-1,index])
        return x

    def set_units(self):
        if self.out_units.lower() == 'mpc':
            factor = 1
            out_length_unit = Mpc

        elif self.out_units.lower() == 'mpch':
            factor = self.config['h']
            out_length_unit = self.Mpc_over_h

        elif self.out_units.lower() == 'kpc':
            factor = 1e3
            out_length_unit = kpc
        
        elif self.out_units.lower() == 'kpch':
            factor = 1e3 * self.config['h']
            out_length_unit = self.kpc_over_h

        return factor, out_length_unit





if '__main__' == __name__:
    fire.Fire(Zinc)
