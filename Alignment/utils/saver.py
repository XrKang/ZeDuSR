import os
import torch
from collections import OrderedDict
import glob
from tensorboardX import SummaryWriter

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.output_path, args.dataset)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def print_log(self,out):
        print(out)
        with open(os.path.join(self.experiment_dir , 'trainout.txt'), 'a+') as f:
            print(out, file=f)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.experiment_dir))
        return writer


