#!/usr/bin/env python3
from Bio import SeqIO
from io import StringIO
from copy import deepcopy
import argparse
import os
import random
import fnmatch
import itertools
import sys
from glob import iglob

import numpy as np
import progressbar

from ont_fast5_api.fast5_interface import get_fast5_file
from sloika.cmdargs import (AutoBool, FileExists, Maybe, Positive)
from sloika.iterators import imap_mp

from sloika import util


program_description = "Extract refereces from fast5 files"
parser = argparse.ArgumentParser(description=program_description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_strand_list', default=None, action=FileExists,
                    help='Strand summary file containing subset')
parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                    help='Number of threads to use when processing data')
parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
                    help='Limit number of reads to process')
parser.add_argument('--overwrite', default=False, action=AutoBool,
                    help='Whether to overwrite any output files')
parser.add_argument('--section', default='template',
                    choices=['template', 'complement'], help='Section to call')
parser.add_argument('input_folder', action=FileExists,
                    help='Directory containing single-read fast5 files')
parser.add_argument('output', help='Output fasta file')


def reference_extraction_worker(file_name, section):
    with get_fast5_file(file_name, mode="r") as file_handle:
        try:
            fasta = file_handle.get_reference_fasta(section=section)
        except Exception as e:
            sys.stderr.write('No reference found for {}.\n{}\n'.format(file_name, repr(e)))
            return None

        iowrapper = StringIO(fasta)
        read_ref = str(next(SeqIO.parse(iowrapper, 'fasta')).seq)
        return (file_name, read_ref)
    
def readtsv(fname, fields=None, **kwargs):
    """Read a tsv file into a numpy array with required field checking

    :param fname: filename to read. If the filename extension is
        gz or bz2, the file is first decompressed.
    :param fields: list of required fields.
    """

    if not file_has_fields(fname, fields):
        raise KeyError('File {} does not contain requested required fields {}'.format(fname, fields))

    for k in ['names', 'delimiter', 'dtype']:
        kwargs.pop(k, None)
    table = np.genfromtxt(fname, names=True, delimiter='\t', dtype=None, encoding='utf8', **kwargs)
    #  Numpy tricks to force single element to be array of one row
    return table.reshape(-1)

def file_has_fields(fname, fields=None):
    """Check that a tsv file has given fields

    :param fname: filename to read. If the filename extension is
        gz or bz2, the file is first decompressed.
    :param fields: list of required fields.

    :returns: boolean
    """

    # Allow a quick return
    req_fields = deepcopy(fields)
    if isinstance(req_fields, str):
        req_fields = [fields]
    if req_fields is None or len(req_fields) == 0:
        return True
    req_fields = set(req_fields)

    inspector = open
    ext = os.path.splitext(fname)[1]
    if ext == '.gz':
        inspector = gzopen
    elif ext == '.bz2':
        inspector = bzopen

    has_fields = None
    with inspector(fname, 'r') as fh:
        present_fields = set(fh.readline().rstrip('\n').split('\t'))
        has_fields = req_fields.issubset(present_fields)
    return has_fields

def iterate_fast5(path='Stream', strand_list=None, paths=False, mode='r',
                  limit=None, shuffle=False, robust=False, progress=False,
                  recursive=False):
    """Iterate over directory of fast5 files, optionally only returning those in list

    :param path: Directory in which single read fast5 are located or filename.
    :param strand_list: List of strands, can be a python list of delimited
        table. If the later and a filename field is present, this is used
        to locate files. If a file is given and a strand field is present,
        the directory index file is searched for and filenames built from that.
    :param paths: Yield file paths instead of fast5 objects.
    :param mode: Mode for opening files.
    :param limit: Limit number of files to consider.
    :param shuffle: Shuffle files to randomize yield of files.
    :param robust: Carry on with iterating over FAST5 files after an exception was raised.
    :param progress: Display progress bar.
    :param recursive: Perform a recursive search for files in subdirectories of `path`.
    """
    if strand_list is None:
        #  Could make glob more specific to filename pattern expected
        if os.path.isdir(path):
            if recursive:
                files = recursive_glob(path, '*.fast5')
            else:
                files = iglob(os.path.join(path, '*.fast5'))
        else:
            files = [path]
    else:
        if isinstance(strand_list, list):
            files = (os.path.join(path, x) for x in strand_list)
        else:
            reads = readtsv(strand_list)
            if 'filename' in reads.dtype.names:
                #  Strand list contains a filename column
                files = (os.path.join(path, x) for x in reads['filename'])
            else:
                raise KeyError("Strand file does not contain required field 'filename'.\n")

    # shuffle means we can't be lazy
    if shuffle and limit is not None:
        files = np.random.choice(list(files), limit, replace=False)
    elif shuffle:
        random.shuffle(list(files))
    elif limit is not None:
        try:
            files = files[:limit]
        except TypeError:
            files = itertools.islice(files, limit)

    if progress:
        bar = progressbar.ProgressBar()
        files = bar(files)

    for f in files:
        if not os.path.exists(f):
            sys.stderr.write('File {} does not exist, skipping\n'.format(f))
            continue
        if not paths:
            try:
                fh = get_fast5_file(f, mode="r")
            except Exception as e:
                if robust:
                    sys.stderr.write("Could not open FAST5 file {}: {}\n".format(f, e))
                else:
                    raise e
            else:
                yield fh
                fh.close()
        else:
            yield os.path.abspath(f)


def recursive_glob(treeroot, pattern):
    # Emulates e.g. glob.glob("**/*.fast5"", recursive=True) in python3
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        for f in goodfiles:
            yield os.path.join(base, f)


def main(argv):
    args = parser.parse_args(argv[1:])

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)

    fast5_files = iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    i = 0
    kwarg_names = ['section']
    with open(args.output, 'w') as file_handle:
        for res in imap_mp(reference_extraction_worker, fast5_files, threads=args.jobs, unordered=True,
                           fix_kwargs=util.get_kwargs(args, kwarg_names)):
            if res is not None:
                i = util.progress_report(i)
                file_name, reference = res
                header = '>{}\n'.format(os.path.basename(os.path.splitext(file_name)[0]))
                file_handle.write(header)
                file_handle.write(reference + '\n')

if __name__ == '__main__':
    sys.exit(main(sys.argv[:]))
