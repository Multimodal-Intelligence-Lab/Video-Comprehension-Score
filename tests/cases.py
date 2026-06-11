"""Characterization-case matrix shared by the golden generator and the tests.

Each case is a fully deterministic ``compute_vcs_score`` invocation: fixed
texts, fixed offline embedders (see embedder.py), and explicit knob values.
The matrix covers both length regimes (m>n, m<n, m==n), every public knob
(chunk_size, Rn, context cutoff/control), the load-sharing penalty path,
empty-string segments, and the known m=n=1 pathology pinned for the
future math phase.
"""
from embedder import (
    embed_dim64,
    split_keep_empty,
    split_sentences,
)

REF_STORY_8 = (
    "A young farmer found a wounded crane in his rice field. "
    "He carried the bird home and bandaged its broken wing. "
    "Over the winter the crane slowly regained its strength. "
    "In early spring the bird finally flew away over the mountains. "
    "Soon after, a mysterious woman arrived at the farmer's door. "
    "She wove beautiful cloth that sold for a high price in the village. "
    "The farmer grew curious about how the cloth was made. "
    "One night he peeked into the weaving room and saw a crane at the loom."
)

GEN_SUMMARY_3 = (
    "A farmer rescued an injured crane and cared for it until it could fly. "
    "Later a strange woman appeared and wove expensive cloth for him. "
    "When he spied on her at night, he discovered the weaver was the crane."
)

SQUARE_REF_4 = (
    "The storm rolled in from the north at dusk. "
    "Fishermen hurried to tie down their boats. "
    "Rain hammered the village through the night. "
    "By morning the harbor was calm again."
)

SQUARE_GEN_4 = (
    "At sunset a storm arrived from the north. "
    "The boats were quickly secured by the fishermen. "
    "It rained heavily all night long. "
    "The harbor had calmed down by daybreak."
)

DISJOINT_GEN_4 = (
    "Quantum computers use qubits to perform parallel calculations. "
    "The stock market closed slightly higher on Tuesday. "
    "Photosynthesis converts sunlight into chemical energy. "
    "The recipe calls for two cups of flour and an egg."
)

# Three generated sentences all paraphrasing the SAME reference sentence:
# in the precision direction they share one reference chunk, which triggers
# the semantic load-sharing penalty.
DUPLICATION_GEN_4 = (
    "The storm rolled in from the north at dusk. "
    "A storm came in from the north as dusk fell. "
    "The storm arrived from the north at nightfall. "
    "Rain hammered the village through the night."
)

# Same eight story sentences, deterministically shuffled: exercises the
# chronology (NAS) penalties without changing semantic content.
REORDERED_GEN_8 = (
    "In early spring the bird finally flew away over the mountains. "
    "A young farmer found a wounded crane in his rice field. "
    "He carried the bird home and bandaged its broken wing. "
    "Over the winter the crane slowly regained its strength. "
    "One night he peeked into the weaving room and saw a crane at the loom. "
    "Soon after, a mysterious woman arrived at the farmer's door. "
    "She wove beautiful cloth that sold for a high price in the village. "
    "The farmer grew curious about how the cloth was made."
)

HALF_OMISSION_GEN_4 = (
    "A young farmer found a wounded crane in his rice field. "
    "He carried the bird home and bandaged its broken wing. "
    "Over the winter the crane slowly regained its strength. "
    "In early spring the bird finally flew away over the mountains."
)

SINGLE_SENTENCE = "A cat sat quietly on the warm mat."

EMPTY_SEG_REF = "First the crew loaded the ship. Then they sailed at dawn."
EMPTY_SEG_GEN = "The crew loaded their ship. They departed at sunrise."

SEGMENTERS = {
    "default": split_sentences,
    "keep_empty": split_keep_empty,
}

EMBEDDERS = {
    "dim64": embed_dim64,
}

CASES = [
    {"name": "typical_8v3_defaults", "ref": REF_STORY_8, "gen": GEN_SUMMARY_3, "params": {}},
    {"name": "typical_3v8_defaults", "ref": GEN_SUMMARY_3, "gen": REF_STORY_8, "params": {}},
    {"name": "square_4v4_defaults", "ref": SQUARE_REF_4, "gen": SQUARE_GEN_4, "params": {}},
    {"name": "identical_4", "ref": SQUARE_REF_4, "gen": SQUARE_REF_4, "params": {}},
    {"name": "disjoint_4", "ref": SQUARE_REF_4, "gen": DISJOINT_GEN_4, "params": {}},
    {"name": "disjoint_4_swapped", "ref": DISJOINT_GEN_4, "gen": SQUARE_REF_4, "params": {}},
    {"name": "duplication_load_sharing", "ref": SQUARE_REF_4, "gen": DUPLICATION_GEN_4, "params": {}},
    {"name": "reordered_8v8", "ref": REF_STORY_8, "gen": REORDERED_GEN_8, "params": {}},
    {"name": "half_omission_8v4", "ref": REF_STORY_8, "gen": HALF_OMISSION_GEN_4, "params": {}},
    # min(m, n) == 1: every alignment window spans the whole other side, so
    # the Global-NAS max-penalty normalizer is vacuously 0. Scored 0.0 in
    # v1/v2 (the old M1 pathology); since v3 it is vacuously perfect (1.0),
    # so identical single-sentence texts score VCS = 1.
    {"name": "single_chunk_identical", "ref": SINGLE_SENTENCE, "gen": SINGLE_SENTENCE, "params": {}},
    {"name": "empty_string_segments", "ref": EMPTY_SEG_REF, "gen": EMPTY_SEG_GEN, "params": {}, "segmenter": "keep_empty"},
    {"name": "chunk_size_2", "ref": REF_STORY_8, "gen": GEN_SUMMARY_3, "params": {"chunk_size": 2}},
    {"name": "chunk_size_3", "ref": REF_STORY_8, "gen": GEN_SUMMARY_3, "params": {"chunk_size": 3}},
    {"name": "rn_1", "ref": REF_STORY_8, "gen": REORDERED_GEN_8, "params": {"Rn": 1}},
    {"name": "rn_2", "ref": REF_STORY_8, "gen": REORDERED_GEN_8, "params": {"Rn": 2}},
    {"name": "rn_1_short_gen", "ref": REF_STORY_8, "gen": GEN_SUMMARY_3, "params": {"Rn": 1}},
    {"name": "context_loose", "ref": REF_STORY_8, "gen": GEN_SUMMARY_3,
     "params": {"context_cutoff_value": 0.3, "context_window_control": 2.0}},
    {"name": "context_tight", "ref": REF_STORY_8, "gen": GEN_SUMMARY_3,
     "params": {"context_cutoff_value": 0.9, "context_window_control": 8.0}},
    {"name": "combo_chunk2_rn1_loose", "ref": SQUARE_REF_4, "gen": SQUARE_GEN_4,
     "params": {"chunk_size": 2, "Rn": 1, "context_cutoff_value": 0.3, "context_window_control": 2.0}},
]


def build_call_kwargs(case):
    """Resolve a case dict into compute_vcs_score keyword arguments
    (excluding the return_* flags, which each test sets itself)."""
    kwargs = {
        "reference_text": case["ref"],
        "generated_text": case["gen"],
        "segmenter_fn": SEGMENTERS[case.get("segmenter", "default")],
        "embedding_fn": EMBEDDERS["dim64"],
    }
    kwargs.update(case["params"])
    return kwargs
