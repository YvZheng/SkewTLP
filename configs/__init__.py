import yaml

fid = open("./configs/cfgs.yml")
cfgs = yaml.load(fid, Loader=yaml.FullLoader)