from typing import List, Union

class TSS:
    chromosome: str
    strand: str
    locus_id: str
    pos: int
    

class ChromosomeTSS:
    def __init__(self, lines: List[List[str]]):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: Union[int, str]) -> TSS:
        pass


class DNATSS:
    def __init__(self, path_to_file: str):
        pass

    def __getitem__(self, index: Union[int, str, slice]) -> Union[ChromosomeTSS, TSS]:
        pass