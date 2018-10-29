#!/usr/bin/env python3
import argparse
import pickle
import h5py
import imp
import logging
import numpy as np
import os
from shutil import copyfile
import sys
import time
import warnings

import theano as th
import theano.tensor as T

from sloika.cmdargs import (AutoBool, display_version_and_exit, FileExists, FilesExist, Maybe,
                            NonNegative, ParseToNamedTuple, Positive, proportion)

import sloika.module_tools as smt
from sloika import updates
from sloika.variables import DEFAULT_ALPHABET
from sloika.version import __version__


logging.getLogger("theano.gof.compilelock").setLevel(logging.WARNING)


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Train a simple neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    common_parser = argparse.ArgumentParser(add_help=False)

    # Optional arguments
    common_parser.add_argument('--adam', nargs=3, metavar=('rate', 'decay1', 'decay2'),
                               default=(1e-3, 0.9, 0.999), type=(NonNegative(float), NonNegative(float),
                                                                 NonNegative(float)), action=ParseToNamedTuple,
                               help='Parameters for Exponential Decay Adaptive Momementum')
    common_parser.add_argument('--bad', default=True, action=AutoBool,
                               help='Force blocks marked as bad to be stays')
    common_parser.add_argument('--batch_size', default=100, metavar='chunks', type=Positive(int),
                               help='Number of chunks to run in parallel')
    common_parser.add_argument('--chunk_len_range', nargs=2, metavar=('min', 'max'),
                               type=Maybe(proportion), default=(0.5, 1.0),
                               help="Randomly sample chunk sizes between min and max (fraction of chunk size in input file)"
                               )
    common_parser.add_argument('--ilf', default=False, action=AutoBool,
                               help='Weight objective function by Inverse Label Frequency')
    common_parser.add_argument('--l2', default=0.0, metavar='penalty', type=NonNegative(float),
                               help='L2 penalty on parameters')
    common_parser.add_argument('--lrdecay', default=5000, metavar='n', type=Positive(float),
                               help='Learning rate for batch i is adam.rate / (1.0 + i / n)')
    common_parser.add_argument('--min_prob', default=1e-30, metavar='p', type=proportion,
                               help='Minimum probability allowed for training')
    common_parser.add_argument('--niteration', metavar='batches', type=Positive(int), default=50000,
                               help='Maximum number of batches to train for')
    common_parser.add_argument('--overwrite', default=False, action=AutoBool, help='Overwrite output directory')
    common_parser.add_argument('--quiet', default=False, action=AutoBool,
                               help="Don't print progress information to stdout")
    common_parser.add_argument('--reweight', metavar='group', default='weights', type=Maybe(str),
                               help="Select chunk according to weights in 'group'")
    common_parser.add_argument('--save_every', metavar='x', type=Positive(int), default=500,
                               help='Save model every x batches')
    common_parser.add_argument('--sd', default=0.5, metavar='value', type=Positive(float),
                               help='Standard deviation to initialise with')
    common_parser.add_argument('--seed', default=None, metavar='integer', type=Positive(int),
                               help='Set random number seed')
    common_parser.add_argument('--smooth', default=0.45, metavar='factor', type=proportion,
                               help='Smoothing factor for reporting progress')
    common_parser.add_argument('--transducer', default=True, action=AutoBool,
                               help='Train a transducer based model')
    common_parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
                               help='Display version information.')

    common_parser.add_argument('--input_load', default=10, type=Positive(int),
                               help='Number of input training data files to load at once')
    common_parser.add_argument('--reload_after_batches', default=250, type=Positive(int),
                               help='Reload training data after this many batches')

    # Positional arguments
    common_parser.add_argument('model', action=FileExists,
                               help='File to read python model description from')
    common_parser.add_argument('output', help='Prefix for output files')
    common_parser.add_argument('input', action=FilesExist, nargs='+',
                               help='HDF5 file containing chunks')

    subparsers = parser.add_subparsers(help='command', dest='command')
    subparsers.required = True

    parser_ev = subparsers.add_parser('events', parents=[common_parser], help='Train from events',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_ev.add_argument('--drop', default=20, metavar='events', type=NonNegative(int),
                           help='Number of events to drop from start and end of chunk before evaluating loss')
    parser_ev.add_argument('--winlen', default=3, type=Positive(int),
                           help='Length of window over data')

    parser_raw = subparsers.add_parser('raw', parents=[common_parser], help='Train from raw signal',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_raw.add_argument('--drop', default=20, metavar='samples', type=NonNegative(int),
                            help='Number of labels to drop from start and end of chunk before evaluating loss')
    parser_raw.add_argument('--winlen', default=11, type=Positive(int),
                            help='Length of window over data')

    args = parser.parse_args()
    assert args.command in ["events", "raw"]
    return args


def main():
    args = get_arguments()
    np.random.seed(args.seed)
    make_output_directory(args)

    log = Logger(os.path.join(args.output, 'model.log'), args.quiet)
    log.write('* Command line\n')
    log.write(' '.join(sys.argv) + '\n')

    all_chunks, all_labels, all_weights, label_weights, min_chunk, max_chunk, data_chunk, \
        training_stride, max_batch_size = load_training_data(args, log)

    network = load_network(args, log, all_chunks, training_stride)
    fg = wrap_network(args, network, min_prob=args.min_prob, l2=args.l2, drop=args.drop)

    total_ev = 0
    score_smoothed = ExponentialSmoother(args.smooth)
    acc_smoothed = ExponentialSmoother(args.smooth)

    log.write('* Dumping initial model\n')
    save_model(network, args.output, 0)

    t0 = time.time()
    log.write('* Training\n')

    decay_start = None

    for i in range(args.niteration):
        learning_rate = get_learning_rate(args, i, decay_start)

        chunk_len = np.random.randint(min_chunk, max_chunk + 1)
        chunk_len = chunk_len - (chunk_len % training_stride)

        batch_size = int(args.batch_size * float(max_chunk) / chunk_len)

        start = np.random.randint(data_chunk - chunk_len + 1)
        start = start - (start % training_stride)

        label_lb = start // training_stride
        label_ub = (start + chunk_len) // training_stride

        idx = np.sort(np.random.choice(len(all_chunks), size=min(batch_size, max_batch_size),
                                       replace=False, p=all_weights))
        indata = np.ascontiguousarray(all_chunks[idx, start : start + chunk_len].transpose((1, 0, 2)))
        labels = np.ascontiguousarray(all_labels[idx, label_lb : label_ub].transpose())
        weights = label_weights[labels]

        fval, batch_acc = fg(indata, labels, weights, learning_rate)
        fval = float(fval)
        nev = np.size(labels)
        total_ev += nev
        score_smoothed.update(fval)
        acc_smoothed.update(batch_acc)

        if (i + 1) % args.save_every == 0:
            save_model(network, args.output, (i + 1) // args.save_every)
            log.write('C')
        else:
            log.write('.')

        if (i + 1) % 50 == 0:
            tn = time.time()
            dt = tn - t0
            t = ' {:5d}: score={:5.3f}, acc={:5.2f}%, time={:5.2f}s ({:.2f} kev/s), lr={}\n'
            log.write(t.format((i + 1) // 50, score_smoothed.value,
                      100.0 * acc_smoothed.value, dt, total_ev / 1000.0 / dt, learning_rate))
            total_ev = 0
            t0 = tn

        if (i + 1) % args.reload_after_batches == 0:
            all_chunks, all_labels, all_weights, label_weights, min_chunk, max_chunk, data_chunk, \
                training_stride, max_batch_size = \
                load_training_data(args, log, previous_stride=training_stride,
                                   previous_min_chunk=min_chunk, previous_max_chunk=max_chunk)

        if decay_start is None and acc_smoothed.value > 0.7:
            decay_start = i

    save_model(network, args.output)


def make_output_directory(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not args.overwrite:
        sys.stderr.write('Error: Output directory {} exists but --overwrite is false\n'.format(args.output))
        exit(1)
    if not os.path.isdir(args.output):
        sys.stderr.write('Error: Output location {} is not directory\n'.format(args.output))
        exit(1)
    copyfile(args.model, os.path.join(args.output, 'model.py'))


def load_network(args, log, all_chunks, training_stride):
    log.write('* Reading network from {}\n'.format(args.model))
    model_ext = os.path.splitext(args.model)[1]
    if model_ext == '.py':
        with h5py.File(args.input[0], 'r') as h5:
            klen = h5.attrs['kmer']
            try:
                alphabet = h5.attrs['alphabet']
                log.write("* Using alphabet: {}\n".format(alphabet.decode('ascii')))
            except:
                alphabet = DEFAULT_ALPHABET
                log.write("* Using default alphabet: {}\n".format(alphabet.decode('ascii')))
                warnings.warn("Deprecated hdf5 input file: missing 'alphabet' attribute")
            nbase = len(alphabet)
        netmodule = imp.load_source('netmodule', args.model)

        network = netmodule.network(klen=klen, sd=args.sd, nbase=nbase,
                                    nfeature=all_chunks.shape[-1],
                                    winlen=args.winlen, stride=training_stride)
    elif model_ext == '.pkl':
        with open(args.model, 'rb') as fh:
            network = pickle.load(fh)
    else:
        log.write('* Model is neither python file nor model pickle\n')
        exit(1)
    return network


def load_training_data(args, log, previous_stride=None,
                       previous_min_chunk=None, previous_max_chunk=None):
    file_count = min(args.input_load, len(args.input))
    input_files = np.random.choice(args.input, file_count, replace=False)

    all_chunks, all_labels, all_weights, all_bad = None, None, None, None

    for input_file in input_files:
        log.write('* Loading data from {}'.format(input_file))
        with h5py.File(input_file, 'r') as h5:
            file_chunks = h5['chunks'][:]
            file_labels = h5['labels'][:]
            file_bad = h5['bad'][:]
            chunk_count = len(file_chunks)
            if args.reweight is not None:
                file_weights = h5[args.reweight][:]
            else:
                file_weights = np.ones(chunk_count)
                file_weights = file_weights.astype('float64')
        log.write(': {:,} chunks\n'.format(chunk_count))

        if all_chunks is None:  # we've just loaded the first file loaded
            all_chunks = file_chunks
            all_labels = file_labels
            all_weights = file_weights
            all_bad = file_bad
        else:  # this is not the first file loaded
            all_chunks = np.concatenate((all_chunks, file_chunks))
            all_labels = np.concatenate((all_labels, file_labels))
            all_weights = np.concatenate((all_weights, file_weights))
            all_bad = np.concatenate((all_bad, file_bad))

    log.write('* Total training data: {:,} chunks\n'.format(len(all_chunks)))
    all_weights /= np.sum(all_weights)
    max_batch_size = (all_weights > 0).sum()

    #  Model stride is forced by training data
    training_stride = int(np.ceil(float(all_chunks.shape[1]) / all_labels.shape[1]))
    if previous_stride is None:
        log.write('* Stride is {}\n'.format(training_stride))
    else:
        if training_stride != previous_stride:
            sys.stderr.write('Error: inconsistent stride values in training data files')
            exit(1)

    min_chunk, max_chunk, data_chunk = check_chunk_len_range(args, all_chunks, log,
                                                             previous_min_chunk, previous_max_chunk)
    if not args.transducer:
        remove_blanks(all_labels)
    if args.bad:
        all_labels[all_bad] = 0

    label_weights = get_label_weights(args, all_labels, all_weights)

    return (all_chunks, all_labels, all_weights, label_weights, min_chunk, max_chunk, data_chunk,
            training_stride, max_batch_size)


def check_chunk_len_range(args, all_chunks, log, previous_min_chunk, previous_max_chunk):
    data_chunk = all_chunks.shape[1]
    if args.chunk_len_range[0] is None:
        min_chunk = 2 * args.drop + 1
    else:
        min_chunk = int(np.around(args.chunk_len_range[0] * data_chunk))
    if args.chunk_len_range[1] is None:
        max_chunk = data_chunk
    else:
        max_chunk = int(np.around(args.chunk_len_range[1] * data_chunk))

    if previous_min_chunk is None and previous_max_chunk is None:
        log.write('* Will use min_chunk, max_chunk = {}, {}\n'.format(min_chunk, max_chunk))
    else:
        if min_chunk != previous_min_chunk or max_chunk != previous_max_chunk:
            sys.stderr.write('Error: inconsistent chunk sizes in training data files')
            exit(1)

    assert max_chunk >= min_chunk, "Min chunk size (got {}) must be <= chunk size (got {})".format(min_chunk, max_chunk)
    assert data_chunk >= max_chunk, "Max chunk size (got {}) must be <= data chunk size (got {})".format(
        max_chunk, data_chunk)
    assert data_chunk >= (
        2 * args.drop + 1), "Data chunk size (got {}) must be > 2 * drop (got {})".format(data_chunk, args.drop)
    assert min_chunk >= (
        2 * args.drop + 1), "Min chunk size (got {}) must be > 2 * drop (got {})".format(min_chunk, args.drop)

    return min_chunk, max_chunk, data_chunk


def get_label_weights(args, all_labels, all_weights):
    if args.ilf:
        #  Calculate frequency of labels and convert into inverse frequency
        label_weights = np.zeros(np.max(all_labels) + 1, dtype='f4')
        for i, lbls in enumerate(all_labels):
            label_weights += all_weights[i] * np.bincount(lbls, minlength=len(label_weights))
        label_weights = np.reciprocal(label_weights)
        label_weights /= np.mean(label_weights)
    else:
        # Default equally weighted
        label_weights = np.ones(np.max(all_labels) + 1, dtype='f4')
    return label_weights


class ExponentialSmoother(object):
    def __init__(self, factor, val=0.0, weight=1e-30):
        assert 0.0 <= factor <= 1.0, "Smoothing factor was {}, should be between 0.0 and 1.0.\n".format(factor)
        self.factor = factor
        self.val = val
        self.weight = weight

    @property
    def value(self):
        return self.val / self.weight

    def update(self, val, weight=1.0):
        self.val = self.factor * self.val + (1.0 - self.factor) * val
        self.weight = self.factor * self.weight + (1.0 - self.factor) * weight


def remove_blanks(labels):
    for lbl_ch in labels:
        for i in range(1, len(lbl_ch)):
            if lbl_ch[i] == 0:
                lbl_ch[i] = lbl_ch[i - 1]
    return labels


def wrap_network(args, network, min_prob=0.0, l2=0.0, drop=0):
    ldrop = drop
    udrop = None if drop == 0 else -drop

    x = T.tensor3()
    labels = T.imatrix()
    weights = T.fmatrix()
    rate = T.scalar()
    post = min_prob + (1.0 - min_prob) * network.run(x)
    penalty = l2 * updates.param_sqr(network)

    loss_per_event, _ = th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])
    loss = penalty + T.mean((weights * loss_per_event)[ldrop : udrop])
    correct = T.eq(T.argmax(post, axis=2), labels)[ldrop : udrop]
    acc = T.mean(correct, dtype=smt.sloika_dtype, acc_dtype=smt.sloika_dtype)
    update_dict = updates.adam(network, loss, rate, (args.adam.decay1, args.adam.decay2))

    fg = th.function([x, labels, weights, rate], [loss, acc], updates=update_dict)
    return fg


def save_model(network, output, index=None):
    if index is not None:
        model_file = 'model_checkpoint_{:05d}.pkl'.format(index)
    else:
        model_file = 'model_final.pkl'

    with open(os.path.join(output, model_file), 'wb') as fh:
        pickle.dump(network, fh, protocol=pickle.HIGHEST_PROTOCOL)


def get_learning_rate(args, i, decay_start):
    """
    Learning rate decays each iteration, but only after decay_start is set (happens when the
    accuracy passes a threshold).
    """
    if decay_start is None:
        return args.adam.rate
    else:
        return args.adam.rate / (1.0 + (i - decay_start) / args.lrdecay)


class Logger(object):

    def __init__(self, log_file_name, quiet=False):
        #
        # Can't have unbuffered text I/O at the moment hence 'b' mode below.
        # See currently open issue http://bugs.python.org/issue17404
        #
        self.fh = open(log_file_name, 'wb', 0)

        self.quiet = quiet

    def write(self, message):
        if not self.quiet:
            sys.stdout.write(message)
            sys.stdout.flush()
        try:
            self.fh.write(message.encode('utf-8'))
        except IOError as e:
            print("Failed to write to log\n Message: {}\n Error: {}".format(message, repr(e)))


if __name__ == '__main__':
    main()
