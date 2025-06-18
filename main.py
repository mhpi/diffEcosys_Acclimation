import sys
sys.path.append(".")
from utils.config_utils import set_config_args
from scripts.experiments import *


args = set_config_args()

if args['Action'] == 0:
    main_All(args)
    print("Models trained on all data and trained models were saved in output directory")
elif args['Action'] == 1:
    main_spatial(args)
elif args['Action'] == 2:
    main_temporal(args)
elif args['Action'] == 3:
    main_global(args)
else:
    print("experiment not set yet")
