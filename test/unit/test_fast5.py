import glob
import parameterized
import os
import unittest

from Bio import SeqIO
import numpy as np
import os
import sys

from glob import iglob
from copy import deepcopy
import random
import fnmatch
import itertools
import progressbar

from sloika import batch, bio, decode, helpers, transducer, util
from sloika.iterators import imap_mp
from sloika.maths import mad

from ont_fast5_api.fast5_interface import get_fast5_file

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

class IterationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']
        self.readdir = os.path.join(self.dataDir, 'reads')
        self.strand_list = os.path.join(self.dataDir, 'strands.txt')
        self.basenames = [
            'read1.fast5',
            'read2.fast5',
            'read3.fast5',
            'read4.fast5',
            'read5.fast5',
            'read6.fast5',
            'read7.fast5',
            'read8.fast5',
        ]
        self.strands = set([os.path.join(self.readdir, r) for r in self.basenames])

    def test_iterate_returns_all(self):
        fast5_files = set(iterate_fast5(self.readdir, paths=True))
        dir_list = set(glob.glob(os.path.join(self.readdir, '*.fast5')))
        self.assertTrue(fast5_files == dir_list)

    def test_iterate_respects_limits(self):
        _LIMIT = 2
        fast5_files = set(iterate_fast5(self.readdir, paths=True, limit=_LIMIT))
        self.assertTrue(len(fast5_files) == _LIMIT)

    def test_iterate_works_with_strandlist(self):
        fast5_files = set(iterate_fast5(self.readdir, paths=True,
                                              strand_list=self.strand_list))
        self.assertTrue(self.strands == fast5_files)


class GetAnyMappingDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    def test_unknown(self):
        basename = 'read6'
        filename = os.path.join(self.dataDir, 'reads', basename + '.fast5')

        with get_fast5_file(filename) as f5:
            ev, _ = f5.get_any_mapping_data('template')
            self.assertEqual(len(ev), 10750)


class ReaderAttributesTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    def test_filename_short(self):
        basename = 'read6'
        filename = os.path.join(self.dataDir, 'reads', basename + '.fast5')

        with get_fast5_file(filename) as f5:
            sn = f5.filename_short
            self.assertEqual(f5.filename_short, basename)


class GetSectionEventsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    @parameterized.expand([
        [os.path.join('reads', 'read3.fast5'), 'Segment_Linear', 9946],
        [os.path.join('reads', 'read6.fast5'), 'Segment_Linear', 11145],
    ])
    def test(self, relative_file_path, analysis, number_of_events):
        filename = os.path.join(self.dataDir, relative_file_path)

        with get_fast5_file(filename) as f5:
            ev = f5.get_section_events('template', analysis=analysis)
            self.assertEqual(len(ev), number_of_events)


class GetReadTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    @parameterized.expand([
        [os.path.join('reads', 'read3.fast5'), 51129, True],
        [os.path.join('reads', 'read6.fast5'), 55885, True],
        [os.path.join('reads', 'read2.fast5'), 69443, True],
        [os.path.join('reads', 'read1.fast5'), 114400, True],
    ])
    def test(self, relative_file_path, number_of_events, raw):
        filename = os.path.join(self.dataDir, relative_file_path)

        with get_fast5_file(filename) as f5:
            ev = f5.get_read(raw=raw)
            self.assertEqual(len(ev), number_of_events)
