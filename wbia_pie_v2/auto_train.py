import subprocess
import logging
import argparse

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path


STATUS_TRAINING = 'training'
STATUS_COMPLETE = 'complete'


def iter_train(args):

    config_dir = args.config_dir
    config_paths = [join(config_dir, f) for f in listdir(config_dir)
                    if isfile(join(config_dir, f)) and f.endswith('.yaml')]

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%m/%d %H:%M:%S'
    )
    logging.info(f'Found {len(config_paths)} config files in {config_dir}: {config_paths}.')

    while get_next_untrained_config_path(config_dir):
        config_path = get_next_untrained_config_path(config_dir)
        try_train(config_path)

    logging.info(f'Done with all training.')


def get_next_untrained_config_path(config_dir):
    config_paths = [join(config_dir, f) for f in listdir(config_dir)
                    if isfile(join(config_dir, f)) and f.endswith('.yaml')]
    config_paths.sort()
    for config_path in config_paths:
        if not trained_already(config_path):
            return config_path
    return None


def try_train(config_path):
    if (trained_already(config_path)):
        logging.info(f'Training results already found for {config_path}, continuing.')
        return

    training_command_args = ['python', 'train.py', '--cfg', config_path]
    try:
        logging.info(f'Beginning to train using {config_path}')
        # add training touchfile
        Path(status_touchfile_path(config_path, STATUS_TRAINING)).touch()
        subprocess.run(training_command_args)
        # remove training touchfile; add complete touchfile
        os.remove(status_touchfile_path(config_path, STATUS_TRAINING))
        Path(status_touchfile_path(config_path, STATUS_COMPLETE)).touch()
        logging.info(f'Done training using {config_path}')

    except:
        logging.error(f'Hit an exception on {config_path} with args {training_command_args}')


# TODO: write this! This method will discern if we've already trained a
def trained_already(config_path):
    return (
        os.path.isfile(status_touchfile_path(config_path, STATUS_TRAINING)) or
        os.path.isfile(status_touchfile_path(config_path, STATUS_COMPLETE))
    )


# this exists to help me standardize these as much as anything
def status_touchfile_path(config_path, status):
    tfile_path = f'{config_path}.{status}'
    return tfile_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config_dir', type=str, default='configs', help='path to config directory')
    # below should let us pass overwrite args to the training commands
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line',
    )
    args = parser.parse_args()

    iter_train(args)
