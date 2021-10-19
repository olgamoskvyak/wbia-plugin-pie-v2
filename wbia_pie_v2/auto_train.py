import subprocess
import logging
import argparse


from os import listdir
from os.path import isfile, join

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

    for config_path in config_paths:
        try_train(config_path)

    logging.info(f'Done with all training.')


def try_train(config_path):
    if (trained_already(config_path)):
        logging.info(f'Training results already found for {config_path}, continuing.')
        return

    training_command_args = ['python', 'train.py', '--cfg', config_path]
    try:
        logging.info(f'Beginning to train using {config_path}')
        subprocess.run(training_command_args)
        logging.info(f'Done training using {config_path}')

    except:
        logging.error(f'Hit an exception on {config_path} with args {training_command_args}')


# TODO: write this! This method will discern if we've already trained a
def trained_already(config_path):
    return False


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
