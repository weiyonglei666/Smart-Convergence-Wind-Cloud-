INTRA_RELATIONS = [
    "ELABORATION", "EXPLANATION", "EVIDENCE", "EXAMPLE", "CONTRAST", "COMPARISON",
    "CONCESSION", "ANTITHESIS", "CAUSE", "RESULT", "CONSEQUENCE", "PURPOSE",
    "CONDITION", "TEMPORAL", "SEQUENCE", "BACKGROUND", "CIRCUMSTANCE", "SUMMARY",
    "RESTATEMENT", "EVALUATION", "INTERPRETATION", "ATTRIBUTION", "DEFINITION",
    "CLASSIFICATION",
]

INTER_RELATIONS = [
    "SUPPORTS", "CONTRADICTS", "ELABORATES", "EXEMPLIFIES", "CAUSES", "RESULTS_FROM",
    "ENABLES", "PREVENTS", "PRECEDES", "FOLLOWS", "SIMULTANEOUS", "BACKGROUND_FOR",
    "GENERALIZES", "SPECIFIES", "COMPARES_WITH", "CONTRASTS_WITH", "SUPPLEMENTS",
    "REPLACES", "MOTIVATES", "JUSTIFIES", "UNRELATED",
]

INTRA_RELATION_DEFINITIONS = """- ELABORATION: Satellite provides additional detail or information about the nucleus.
- EXPLANATION: Satellite explains or clarifies the nucleus content.
- EVIDENCE: Satellite provides evidence or proof for the nucleus claim.
- EXAMPLE: Satellite gives a specific example of the nucleus concept.
- CONTRAST: Satellite presents opposing or contrasting information.
- COMPARISON: Satellite compares two or more entities or concepts.
- CONCESSION: Satellite acknowledges opposing viewpoint while maintaining main claim.
- ANTITHESIS: Satellite presents directly opposite or contradictory information.
- CAUSE: Satellite describes the cause of an event or situation.
- RESULT: Satellite describes the result or consequence of an action.
- CONSEQUENCE: Satellite shows the outcome following from the nucleus.
- PURPOSE: Satellite explains the intended goal or purpose.
- CONDITION: Satellite specifies conditions under which something holds.
- TEMPORAL: Satellite indicates temporal relationship between events.
- SEQUENCE: Satellite shows sequential order of events or actions.
- BACKGROUND: Satellite provides background context or setting.
- CIRCUMSTANCE: Satellite describes circumstances surrounding an event.
- SUMMARY: Satellite summarizes or generalizes the nucleus content.
- RESTATEMENT: Satellite restates the nucleus in different words.
- EVALUATION: Satellite provides evaluation or assessment of the nucleus.
- INTERPRETATION: Satellite offers interpretation of the nucleus content.
- ATTRIBUTION: Satellite attributes information to a source.
- DEFINITION: Satellite defines a term or concept.
- CLASSIFICATION: Satellite classifies or categorizes information."""

INTER_RELATION_DEFINITIONS = """- SUPPORTS: Chunk provides support or evidence for another chunk.
- CONTRADICTS: Chunk contradicts or opposes another chunk.
- ELABORATES: Chunk elaborates on information in another chunk.
- EXEMPLIFIES: Chunk provides examples for another chunk's concepts.
- CAUSES: Chunk describes causes for events in another chunk.
- RESULTS_FROM: Chunk describes results from another chunk's events.
- ENABLES: Chunk describes what enables another chunk's situation.
- PREVENTS: Chunk describes what prevents another chunk's situation.
- PRECEDES: Chunk describes events that precede another chunk.
- FOLLOWS: Chunk describes events that follow another chunk.
- SIMULTANEOUS: Chunk describes simultaneous events with another chunk.
- BACKGROUND_FOR: Chunk provides background context for another chunk.
- GENERALIZES: Chunk provides general principles for another chunk's specifics.
- SPECIFIES: Chunk provides specific details for another chunk's generalizations.
- COMPARES_WITH: Chunk compares information with another chunk.
- CONTRASTS_WITH: Chunk contrasts information with another chunk.
- SUPPLEMENTS: Chunk supplements information in another chunk.
- REPLACES: Chunk replaces or updates information in another chunk.
- MOTIVATES: Chunk provides motivation for another chunk's content.
- JUSTIFIES: Chunk justifies claims or actions in another chunk.
- UNRELATED: Chunk has no rhetorical or semantic relation to another chunk."""
