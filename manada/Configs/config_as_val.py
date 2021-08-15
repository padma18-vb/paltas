from manada.Configs.config_amp_and_slope import *
from manada.Sources.cosmos import COSMOSIncludeCatalog
import os
import pandas as pd

# config_dict and root path in the import *.
config_dict['source']['parameters']['source_inclusion_list'] = pd.read_csv(
	os.path.join(root_path,'manada/Sources/val_galaxies.csv'),
	names=['catalog_i'])['catalog_i'].to_numpy()
config_dict['source']['class'] = COSMOSIncludeCatalog
del config_dict['source']['parameters']['source_exclusion_list']
