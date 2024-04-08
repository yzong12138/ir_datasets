import codecs
from typing import NamedTuple, Tuple
import ir_datasets
from ir_datasets.indices import PickleLz4FullStore
from ir_datasets.util import Lazy, DownloadConfig, Migrator
from ir_datasets.datasets.base import Dataset, FilteredQueries, FilteredQrels, YamlDocumentation
from ir_datasets.formats import DocstoreBackedDocs, TsvQueries, TrecQrels, GenericDoc, TsvDocs, BaseQrels

NAME = 'natural-questions-simplified'

class NqQrel(NamedTuple):
    query_id: str
    doc_id: str
    relevance: int

class NqQrels(BaseQrels):
    def __init__(self, dlc):
        super().__init__()
        self.dlc = dlc

    def qrels_iter(self):
        with self.dlc.stream() as stream:
            stream = codecs.getreader('utf8')(stream)
            for line in stream:
                cols = line.rstrip().split()
                yield NqQrel(query_id=cols[0], doc_id=cols[1], relevance=1)

    def qrels_cls(self):
        return NqQrel


def _init():
    subsets = {}
    base_path = ir_datasets.util.home_path()/NAME
    dlc = DownloadConfig.context(NAME, base_path)
    documentation = YamlDocumentation(f'docs/{NAME}.yaml')

    collection = TsvDocs(dlc["doc"], namespace=NAME, lang='en')

    base = Dataset(
        collection,
        documentation('_')
    )

    subsets["train"] = Dataset(
        collection, 
        TsvQueries(dlc["train/queries"], namespace=NAME, lang='en'),
        NqQrels(dlc["train/qrels"]),
        documentation('train')
    )

    subsets["dev"] = Dataset(
        collection, 
        TsvQueries(dlc["dev/queries"], namespace=NAME, lang='en'),
        NqQrels(dlc["dev/qrels"]),
        documentation('dev')
    )

    subsets["dev/seen"] = Dataset(
        collection, 
        TsvQueries(dlc["dev/seen/queries"], namespace=NAME, lang='en'),
        NqQrels(dlc["dev/seen/qrels"]),
        documentation('seen')
    )

    subsets["dev/unseen"] = Dataset(
        collection, 
        TsvQueries(dlc["dev/unseen/queries"], namespace=NAME, lang='en'),
        NqQrels(dlc["dev/unseen/qrels"]),
        documentation('unseen')
    )

    ir_datasets.registry.register(NAME, base)
    for s in sorted(subsets):
        ir_datasets.registry.register(f'{NAME}/{s}', subsets[s])

    return base, subsets

base, subsets = _init()