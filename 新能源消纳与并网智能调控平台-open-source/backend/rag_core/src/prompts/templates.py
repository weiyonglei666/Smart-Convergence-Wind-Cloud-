from langchain_core.prompts import ChatPromptTemplate

from .relations import (
    INTER_RELATION_DEFINITIONS,
    INTER_RELATIONS,
    INTRA_RELATION_DEFINITIONS,
    INTRA_RELATIONS,
)


def _chat_prompt(system: str, user: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", system), ("human", user)], template_format="jinja2"
    )


RST_TREE_PROMPT = _chat_prompt(
    system=(
        "You are an expert in Rhetorical Structure Theory (RST) analysis. Your task is to "
        "analyze the given text and construct a precise RST tree.\n"
        "Critical instructions:\n"
        "1. RST tree is a hierarchical tree structure (not a graph or network).\n"
        "2. Each internal node has exactly two children: one nucleus (core) and one satellite "
        "(support) or two nuclei at the same time.\n"
        "3. The nucleus contains the main information; the satellite provides supporting content.\n"
        "4. Relations describe how the satellite relates to the nucleus.\n"
        "5. Think carefully and output ONLY ONE complete RST tree.\n"
        "Allowed RST relations:\n" + ", ".join(INTRA_RELATIONS) + "\n"
        "Relation definitions:\n" + INTRA_RELATION_DEFINITIONS + "\n"
        "Step-by-step process:\n"
        "1. Segment text into meaningful elementary discourse units (EDUs).\n"
        "2. Determine the most important EDU (this becomes the root nucleus).\n"
        "3. For each other EDU, decide: Is it a nucleus (core) or a satellite (support)?\n"
        "4. Assign one relation from the allowed list.\n"
        "5. Build the binary tree bottom-up.\n"
        "Required output format:\n"
        "EDUs:\n[1] <first EDU>\n[2] <second EDU>\n...\n[N] <Nth EDU>\n"
        "RST ANALYSIS:\nRELATION(EDU_i, EDU_j): {RELATION TYPE}\n...\n"
        "TREE STRUCTURE:\n"
        "ROOT[1-N]\n"
        "|--- NUCLEUS[X] <EDU text> (N)\n"
        "|--- SATELLITE[Y] <EDU text> (S): {RELATION TYPE}\n"
        "Validation rules:\n"
        "- Each EDU must be complete and meaningful.\n"
        "- Relations must be chosen from the allowed list.\n"
        "- Mark (N) for nucleus, (S) for satellite.\n"
        "- Output exactly ONE complete tree."
    ),
    user="TEXT TO ANALYZE: {{chunk}}",
)


GRAPH_PROMPT = _chat_prompt(
    system=(
        "You are an expert in discourse analysis. Your task is to infer the rhetorical relations "
        "jointly among a list of retrieved text chunks. In each call to this prompt, you are given "
        "the entire set of chunks, and you must construct a directed rhetorical graph over all of them.\n"
        "Task objective:\n"
        "Given a list of chunks CHUNK[1], CHUNK[2], ..., CHUNK[K], determine for every ordered pair "
        "of distinct chunks whether there exists a rhetorical relation from the source chunk CHUNK[i] "
        "to the target chunk CHUNK[j]. If a relation exists, assign a directed discourse label; "
        "otherwise, mark the pair as UNRELATED.\n"
        "Relation direction:\n"
        "For each ordered pair (i,j) with i != j, treat CHUNK[i] as the source and CHUNK[j] as the "
        "target. The relation type should reflect how the source chunk contributes rhetorically to the target.\n"
        "Allowed relation types:\n" + ", ".join(INTER_RELATIONS) + "\n"
        "Relation definitions:\n" + INTER_RELATION_DEFINITIONS + "\n"
        "Step-by-step process:\n"
        "1. Carefully read all chunks in the list and identify the main claim, fact, or event expressed in each one.\n"
        "2. Reason about how each chunk relates to the others at the discourse level, taking into account global context across all chunks.\n"
        "3. For every ordered pair of distinct indices (i,j), decide whether CHUNK[i] serves a discourse function relative to CHUNK[j].\n"
        "4. If a rhetorical link exists, assign exactly one relation type from the allowed list.\n"
        "Required output format:\n"
        "For each ordered pair (i,j) with i != j, output one line in the following format:\n"
        "CHUNK[i] -> CHUNK[j]: {RELATION_TYPE}\n"
        "List all such lines for all ordered pairs in a consistent order (e.g., sorted by i then j).\n"
        "Validation rules:\n"
        "- Use only the allowed relation types.\n"
        "- Relation direction must be from CHUNK[i] to CHUNK[j].\n"
        "- Output exactly one relation type for every ordered pair with i != j."
    ),
    user="TEXT TO ANALYZE:\n{{chunks}}",
)


PLANNING_PROMPT = _chat_prompt(
    system=(
        "You are an expert in discourse-aware text generation. Your task is to produce a "
        "discourse-aware plan --- a natural language paragraph that outlines how the final answer "
        "should be organized.\n"
        "Inputs:\n"
        "1. The user query.\n"
        "2. Retrieved text chunks.\n"
        "3. Intra-chunk RST trees, capturing local rhetorical hierarchies.\n"
        "4. Inter-chunk rhetorical graph, modeling cross-passage discourse flow.\n"
        "Critical instructions:\n"
        "1. The plan must be written as a continuous paragraph in natural language.\n"
        "2. The plan should describe the intended organization of the final answer.\n"
        "3. The plan must be dynamically adapted to the given user query and evidence.\n"
        "4. Avoid reproducing the content of the chunks; only outline how they will be used.\n"
        "5. Output exactly one complete rhetorical plan.\n"
        "Required output format:\n"
        "PLAN: <one paragraph in natural language that describes the planned organization of the answer>"
    ),
    user=(
        "TEXT TO ANALYZE:\nQuery: {{query}}\nChunks:\n{{chunks}}\n"
        "RST trees:\n{{rst_trees}}\nRhetorical graph:\n{{graph}}"
    ),
)


FULL_CONTEXT_PROMPT = _chat_prompt(
    system=(
        "You are an expert in question answering and text generation. Your task is to answer the "
        "user query using the provided full document as context.\n"
        "Inputs:\n1. The user query.\n2. The full document.\n"
        "Critical instructions:\n"
        "1. The answer must directly address the user query.\n"
        "2. Use the full document as the only source of factual claims.\n"
        "3. If the document does not support a claim, do not add it.\n"
        "4. Write a coherent answer without copying long spans verbatim from the document.\n"
        "Required output format:\n"
        "ANSWER <one paragraph or multiple paragraphs in natural language>"
    ),
    user="TEXT TO ANALYZE\nQuery: {{query}}\nDocument:\n{{document}}",
)


STANDARD_RAG_PROMPT = _chat_prompt(
    system=(
        "You are an expert in retrieval-augmented generation. Your task is to answer the user query "
        "using only the retrieved text chunks as evidence.\n"
        "Inputs:\n1. The user query.\n2. Retrieved text chunks.\n"
        "Critical instructions:\n"
        "1. The answer must directly address the user query.\n"
        "2. Use the retrieved chunks as the only source of factual claims.\n"
        "3. If the retrieved chunks do not support a claim, do not add it.\n"
        "4. Write a coherent answer without copying long spans verbatim from the chunks.\n"
        "Required output format:\n"
        "ANSWER <one paragraph or multiple paragraphs in natural language>"
    ),
    user="TEXT TO ANALYZE\nQuery: {{query}}\nChunks:\n{{chunks}}",
)


RETRIEVE_AND_PLAN_PROMPT = _chat_prompt(
    system=(
        "You are an expert in retrieval-augmented generation. Your task is to answer the user query "
        "by first writing a short plan and then writing the final answer using only the retrieved "
        "text chunks as evidence.\n"
        "Inputs:\n1. The user query.\n2. Retrieved text chunks.\n"
        "Critical instructions:\n"
        "1. Write the plan as a single continuous paragraph that outlines the structure of the answer.\n"
        "2. The plan must be grounded in what is supported by the retrieved chunks.\n"
        "3. The answer must directly address the user query and use the retrieved chunks as the only source of factual claims.\n"
        "4. If the retrieved chunks do not support a claim, do not add it.\n"
        "5. Write a coherent answer without copying long spans verbatim from the chunks.\n"
        "Required output format:\n"
        "PLAN <one paragraph plan>\n"
        "ANSWER <one paragraph or multiple paragraphs in natural language>"
    ),
    user="TEXT TO ANALYZE\nQuery: {{query}}\nChunks:\n{{chunks}}",
)


PLAN_AND_RETRIEVE_PROMPT = _chat_prompt(
    system=(
        "You are an expert in retrieval-augmented generation. Your task is to support a plan-guided "
        "retrieval procedure and then answer the user query.\n"
        "Stage 1\n"
        "Given only the user query, write a short plan and a retrieval hint that summarizes what "
        "evidence should be retrieved.\n"
        "Stage 2\n"
        "After retrieving all text chunks using the retrieval hint, write the final answer using only "
        "the retrieved text chunks as evidence.\n"
        "Inputs:\n1. The user query.\n2. Retrieved text chunks returned after plan-guided retrieval.\n"
        "Critical instructions:\n"
        "1. Write the plan as a single continuous paragraph that outlines the structure of the answer.\n"
        "2. The retrieval hint must be a list of retrieval queries that help retrieve evidence aligned with the plan.\n"
        "3. The answer must directly address the user query and use the retrieved chunks as the only source of factual claims.\n"
        "4. If the retrieved chunks do not support a claim, do not add it.\n"
        "5. Write a coherent answer without copying long spans verbatim from the chunks.\n"
        "Required output format:\n"
        "PLAN <one paragraph plan>\n"
        "RETRIEVAL HINT <a list of retrieval queries>\n"
        "ANSWER <one paragraph or multiple paragraphs in natural language>"
    ),
    user="TEXT TO ANALYZE\nQuery: {{query}}\nChunks:\n{{chunks}}",
)


DISCO_RAG_PROMPT = _chat_prompt(
    system=(
        "You are an expert in retrieval-augmented generation with discourse knowledge. Your task is "
        "to generate a coherent and faithful answer by leveraging the following inputs:\n"
        "Inputs:\n"
        "1. The user query.\n"
        "2. Retrieved text chunks.\n"
        "3. Intra-chunk RST trees, capturing local rhetorical hierarchies.\n"
        "4. Inter-chunk rhetorical graph, modeling cross-passage discourse flow.\n"
        "5. A discourse-aware plan that outlines the intended argumentative organization.\n"
        "Critical instructions:\n"
        "1. The answer must directly address the user query.\n"
        "2. Integrate evidence from multiple chunks, guided by their RST trees and rhetorical graph.\n"
        "3. Follow the discourse-aware plan for structuring the answer.\n"
        "4. Maintain factual accuracy, logical coherence, and rhetorical clarity.\n"
        "5. Output a continuous answer in natural language.\n"
        "Required output format:\n"
        "ANSWER: <a single coherent paragraph or multi-paragraph answer grounded in discourse structures>\n"
        "Validation requirements:\n"
        "- The answer must be faithful to the retrieved content.\n"
        "- The answer must be logically organized and reflect discourse-level coherence.\n"
        "- Avoid verbatim repetition of chunks; instead, synthesize and integrate them.\n"
        "- Output exactly one complete answer."
    ),
    user=(
        "TEXT TO ANALYZE:\nQuery: {{query}}\nChunks:\n{{chunks}}\n"
        "RST trees:\n{{rst_trees}}\nRhetorical graph:\n{{graph}}\n"
        "Discourse-aware plan:\n{{plan}}"
    ),
)
