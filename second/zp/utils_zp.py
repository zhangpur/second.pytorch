import argparse
import os
import yaml
def load_arg(p,config):
    # save arg
    parser = argparse.ArgumentParser()
    if  os.path.exists(config):
        with open(config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False
def save_arg(args,config):
    # save arg
    arg_dict = vars(args)
    with open(config, 'w') as f:
        yaml.dump(arg_dict, f)