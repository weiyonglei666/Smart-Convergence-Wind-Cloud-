from dataclasses import dataclass, field
from typing import List


@dataclass
class EDU:
    index: int
    text: str
    role: str = "N"


@dataclass
class RstRelation:
    head: int
    dep: int
    relation: str


@dataclass
class RstTree:
    chunk_id: int
    edus: List[EDU] = field(default_factory=list)
    relations: List[RstRelation] = field(default_factory=list)
    raw: str = ""

    def serialize(self) -> str:
        edu_str = " ".join(f"[{e.index}]({e.role}) {e.text}" for e in self.edus)
        rel_str = "; ".join(f"{r.relation}(EDU{r.head}->EDU{r.dep})" for r in self.relations)
        return f"[Chunk {self.chunk_id} RST Tree]\nEDUs: {edu_str}\nRelations: {rel_str}"
