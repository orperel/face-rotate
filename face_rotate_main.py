import argparse
import logging

from sanity.gender_trainer import GenderFaderNetTrainer

from evaluator import report_status_to_visdom, show_random_samples
from sanity.gender_evaluator import report_gender_status_to_visdom, show_gender_random_samples
from trainer import FaderNetTrainer
from trainer_params import training_params


def begin_training():
    trainer = FaderNetTrainer(training_params)
    trainer.train()


def begin_training_gender():
    trainer = GenderFaderNetTrainer(training_params)
    trainer.train()


def continue_training(flush_params):
    with_params = training_params if flush_params else None
    FaderNetTrainer.continue_training(training_params['models_path'], with_params)


def continue_gender(flush_params):
    with_params = training_params if flush_params else None
    GenderFaderNetTrainer.continue_training(training_params['models_path'], with_params)


def evaluate():
    report_status_to_visdom(training_params['plot_path'] + 'last_plot.pth')
    show_random_samples()


def evaluate_gender():
    report_gender_status_to_visdom(training_params['plot_path'] + 'last_plot.pth')
    show_gender_random_samples()


logging.basicConfig(level=logging.DEBUG, format="%(message)s")
description = 'face_rotate_main --run [ start | continue | eval ]'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--run', help='start / continue training, or evaluate results')
parser.add_argument('--flush_params', help='continue training but use parameters from current training_params.py')
args = parser.parse_args()
if args.run == 'start':
    begin_training()
elif args.run == 'continue':
    flush_params = args.flush_params == 'Yes' or args.flush_params == 'True' or \
                   args.flush_params == 'true' or args.flush_params == '1'
    continue_training(flush_params)
elif args.run == 'eval':
    evaluate()
elif args.run == 'gender':
    begin_training_gender()
elif args.run == 'eval_gender':
    evaluate_gender()
else:
    print('Usage: ' + description)
