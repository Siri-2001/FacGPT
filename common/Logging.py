import logging,datetime
import os.path


class Log:
    def __init__(self, msg, logging_path_file):
        self.msg = msg
        self.lpf = logging_path_file

    def log(self):

        logging.basicConfig(filemode ="w", filename=self.lpf, level=0)
        logging.info(self.msg)
        print(self.msg)

def build_log(model,config):
    ckpt_file = str(datetime.datetime.now()).replace(' ', '').replace('-', '').replace(':', '_').replace('.', '_')
    os.makedirs(config["train"]["checkpoint_path"], exist_ok=True)
    if os.path.exists(os.path.join(config["train"]["checkpoint_path"], ckpt_file)):
        exit('Too Fast!!!')
    else:
        os.mkdir(os.path.join(config["train"]["checkpoint_path"], ckpt_file))
    print(model, file=open(os.path.join(config["train"]["checkpoint_path"],ckpt_file, 'model_config.txt'),'w'))
    return ckpt_file