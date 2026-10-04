import yaml
import dotmap

from data.spiking_digits import SpikingDigits

cfg = yaml.load(open('configs/digits.yaml', 'r'), Loader=yaml.FullLoader)
cfg = dotmap.DotMap(cfg)
dm = SpikingDigits(cfg)

dm.prepare_data()