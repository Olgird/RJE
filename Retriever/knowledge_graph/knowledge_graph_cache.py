"""A Cached KG"""

import json
from typing import Dict, List, Set

from config import cfg


class KnowledgeGraphCache(object):
    def __init__(self, kb_filename=cfg.localgraph["path"]) -> None:
        self.dict = self._load_localgraph(kb_filename)

    def _load_localgraph(self, kb_filename):
        localgraph = {}
        with open(kb_filename, "r") as f:
            for line in f:
                localgraph.update(json.loads(line))
        return localgraph

    def get_relation(self, src) -> Set[str]:
        return set(self.dict.get(src, {}).keys())

    def get_tail(self, src, relation) -> Set[str]:
        return set(self.dict.get(src, {}).get(relation, []))

    def get_rel_tail(self, src) -> Dict[str, List[str]]:
        return self.dict.get(src, {})

    def get_nodes_from_path(
        self,
        src: str,
        path: List[str],
        threshold=0,
        no_hop_flag: str = "END OF RELATIONAL PATH",
    ) -> Set[str]:
        nodes = set()
        head_set = {src}
        for relation in path:
            if relation == no_hop_flag:
                break
            tail_set = set()
            for head in head_set:
                tail_set = tail_set | self.get_tail(head, relation)
            head_set = tail_set
            if threshold == 0 or len(head_set) <= threshold:
                nodes = nodes | head_set
        nodes = nodes | head_set
        return nodes

    def get_tails_from_path(
        self, src: str, path: List[str], no_hop_flag: str = "END OF RELATIONAL PATH"
    ) -> Set[str]:
        head_set = {src}
        for relation in path:
            if relation == no_hop_flag:
                break
            tail_set = set()
            for head in head_set:
                tail_set = tail_set | self.get_tail(head, relation)
            head_set = tail_set
        return head_set

    def judge_src(self, entity: str, answers: Set[str], path: List[str]) -> str:
        return answers & self.get_tails_from_path(entity, path)

    def get_shortest_index(self, src: str, path: List[str], answers: set) -> int:
        head_set = {src}
        for index, relation in enumerate(path):
            tail_set = set()
            for head in head_set:
                tail_set.update(self.get_tail(head, relation))
            if tail_set & answers:
                return index + 1
            head_set = tail_set
        return -1
