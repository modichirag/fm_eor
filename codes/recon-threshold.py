import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel,
                        LBFGS, ParticleMesh)

from abopt.algs.lbfgs import scalar as scalar_diag

from nbodykit.cosmology import Planck15, EHPower, Cosmology
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, BigFileMesh, BigFileCatalog, ArrayCatalog
import os, json
##
#Import the relevant map and objective function relevant to reconstruction
from cosmo4d.lab import mapthreshold as map
from cosmo4d.lab import objectives
from cosmo4d.lab import mapnoise
from cosmo4d.lab import dg
from solve import solve

##################################################################################

#Set parameters here
bs, nc = 256., 64
truth_pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f4')
nsteps = 5
aa = 1.0000
B = 1
b1, b2 = 2. , 1.
threshold = 10.

noisevar = 0.01
smooth = None
seed = 999

## Basic bookkeeping. Creating folders and paths to save files
ofolder = '/global/cscratch1/sd/chmodi/fm_eor/threshold/'
prefix = 'test'
fname = 's%d_%s'%(seed, prefix)
optfolder = ofolder + 'opt_%s/'%fname
if truth_pm.comm.rank == 0:    print('Output Folder is %s'%optfolder)

for folder in [ofolder, optfolder]:
    try:
        os.makedirs(folder)
    except:
        pass


#initiate

klin, plin = numpy.loadtxt('../data/pklin_1.0000.txt', unpack = True)
ipk = interpolate(klin, plin)
cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)

##################################################################################
##setup the model
##
stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
dynamic_model = NBodyModel(cosmo, truth_pm, B=B, steps=stages)
#dynamic_model = ZAModel(cosmo, truth_pm, B=B, steps=stages)

#noise
#Artifically low noise since the data is constructed from the model
truth_noise_model = map.NoiseModel(truth_pm, None, noisevar*(truth_pm.BoxSize/truth_pm.Nmesh).prod(), 1234)
mock_model = map.MockModel(dynamic_model, bias=[b1, b2], threshold=threshold)

##################################################################################

### Create and save data if not found
### Comment the following lines since we do not have any pregenerated data yet
##
s_truth = truth_pm.generate_whitenoise(seed, mode='complex')\
        .apply(lambda k, v: v * (ipk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()

#This is what make_observable does under the hood
#final, model = mock_model.get_code().compute(['final', 'model'], init={'parameters':s_truth})
data_p = mock_model.make_observable(s_truth)
data_p.save(optfolder+'data/')


#Add noise to the data
data_n = truth_noise_model.add_noise(data_p)
data_n.save(optfolder+'datan/')


##################################################################################
## Do Reconstruction
##
s_init = truth_pm.generate_whitenoise(777, mode='complex')\
        .apply(lambda k, v: v * (ipk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
        .c2r()*0.001


## Smooth the residual on small scales
sms = [4.0, 2.0, 1.0, 0.5, 0.0]

x0 = s_init
N0 = nc
C = x0.BoxSize[0] / x0.Nmesh[0]

for Ns in sms:
    if truth_pm.comm.rank == 0: print('\nDo for cell smoothing of %0.2f\n'%(Ns))
    #x0 = solve(N0, x0, 0.005, '%d-%0.2f'%(N0, Ns), Ns)
    sml = C * Ns
    rtol = 0.01
    maxiter = 101
    run = '%d-%0.2f'%(N0, Ns)
    obj = objectives.SmoothedObjective(mock_model, truth_noise_model, data_n, prior_ps=ipk, sml=sml)
    x0 = solve(N0, x0, rtol, run, Ns, prefix, mock_model, obj, data_p, truth_pm, optfolder, \
               saveit=50, showit=5, title=None, maxiter=maxiter)    

