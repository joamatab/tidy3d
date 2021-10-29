""" Tidy3d package imports"""
__version__ = "0.0.0"

from rich import pretty, traceback

# import component as `from tidy3d import Simulation` or `td.Simulation`
from .components import PML, StablePML, Absorber
from .components import Grid, Coords
from .components import Box, Sphere, Cylinder, PolySlab
from .components import Geometry
from .components import Structure
from .components import Medium, PoleResidue, Sellmeier, Debye, Lorentz
from .components import nk_to_eps_complex, nk_to_eps_sigma, eps_complex_to_nk
from .components import nk_to_medium, eps_sigma_to_eps_complex
from .components import GaussianPulse
from .components import VolumeSource, PlaneWave, ModeSource, GaussianBeam
from .components import FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor
from .components import ModeMonitor
from .components import Mode
from .components import Simulation
from .components import SimulationData, FieldData, FluxData, ModeData, FluxTimeData
from .components import data_type_map, ScalarFieldData, ScalarFieldTimeData

# constants imported as `C_0 = td.C_0` or `td.constants.C_0`
from .constants import inf, C_0, ETA_0

# plugins typically imported as `from tidy3d.plugins import DispersionFitter`
from . import plugins

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from .material_library import material_library

from .log import log

# make all stdout and errors pretty
pretty.install()
traceback.install()
