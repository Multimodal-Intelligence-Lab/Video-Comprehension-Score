"""VCS Configuration Constants

This module defines the default configuration parameters used throughout the VCS
library. These constants provide sensible defaults that work well for most text
comparison scenarios, but can be overridden in the compute_vcs_score function.

Examples
--------
Using default values:

>>> from vcs import compute_vcs_score
>>> result = compute_vcs_score(ref_text, gen_text, segmenter, embedder)
>>> # Uses all default configuration values

Overriding specific defaults:

>>> from vcs._config import DEFAULT_CONTEXT_CUTOFF_VALUE
>>> result = compute_vcs_score(
...     ref_text, gen_text, segmenter, embedder,
...     context_cutoff_value=DEFAULT_CONTEXT_CUTOFF_VALUE + 0.1,  # More restrictive
...     chunk_size=2  # Override default chunk size
... )

See Also
--------
compute_vcs_score : Main function that uses these configuration values
"""

DEFAULT_CONTEXT_CUTOFF_VALUE = 0.6
"""Default context cutoff value for best match finding.

Controls when context windows are applied during the best match selection process.
Higher values make context windows less likely to be applied, resulting in more
restrictive matching behavior.

Type
----
float

Value
-----
0.6

Notes
-----
* Range: 0.0 to 1.0
* Higher values (closer to 1.0) = more restrictive matching
* Lower values (closer to 0.0) = more permissive matching  
* Values around 0.6-0.7 work well for most text types
* Consider increasing for very noisy text or when strict matching is needed
* Consider decreasing for texts with significant paraphrasing

Examples
--------
>>> # More restrictive matching
>>> result = compute_vcs_score(ref, gen, seg, emb, context_cutoff_value=0.8)

>>> # More permissive matching  
>>> result = compute_vcs_score(ref, gen, seg, emb, context_cutoff_value=0.4)

See Also
--------
DEFAULT_CONTEXT_WINDOW_CONTROL : Controls context window size when applied
"""

DEFAULT_CONTEXT_WINDOW_CONTROL = 4.0
"""Default context window control parameter.

Controls the size of context windows when they are applied during best match
finding. Larger values create smaller context windows (more restrictive), while
smaller values create larger context windows (more permissive).

Type
----
float

Value
-----
4.0

Notes
-----
* Typical range: 1.0 to 10.0
* Higher values = smaller context windows = more restrictive
* Lower values = larger context windows = more permissive
* Works in conjunction with DEFAULT_CONTEXT_CUTOFF_VALUE
* Only affects matching when context windows are actually applied

Examples
--------
>>> # Smaller context windows (more restrictive)
>>> result = compute_vcs_score(ref, gen, seg, emb, context_window_control=6.0)

>>> # Larger context windows (more permissive)
>>> result = compute_vcs_score(ref, gen, seg, emb, context_window_control=2.0)

See Also
--------
DEFAULT_CONTEXT_CUTOFF_VALUE : Controls when context windows are applied
"""

DEFAULT_LCT = 0
"""Default Local Chronology Tolerance (LCT) value.

Controls how much flexibility is allowed in narrative chronological ordering.
Higher values permit more deviation from strict temporal sequence, while 0
requires strict chronological alignment.

Type
----
int

Value
-----
0

Notes
-----
* Range: 0 to small positive integers (typically 0-3)
* 0 = strict chronological order required
* 1 = small deviations from chronological order allowed
* 2+ = increasing flexibility in temporal ordering
* Affects NAS-D and NAS-L penalty calculations
* Useful for texts where some reordering is semantically acceptable

Use Cases
---------
* LCT=0: News articles, stories, procedural text (strict temporal order)
* LCT=1: Summaries, descriptions (minor reordering acceptable)  
* LCT=2+: Creative text, flexible narratives (more reordering acceptable)

Examples
--------
>>> # Strict chronological ordering (default)
>>> result = compute_vcs_score(ref, gen, seg, emb, lct=0)

>>> # Allow minor reordering
>>> result = compute_vcs_score(ref, gen, seg, emb, lct=1)

>>> # Allow more flexible ordering
>>> result = compute_vcs_score(ref, gen, seg, emb, lct=2)

See Also
--------
compute_vcs_score : Parameter description for detailed LCT behavior
visualize_distance_nas : See LCT effects on penalty calculations
"""

DEFAULT_CHUNK_SIZE = 1
"""Default chunk size for grouping text segments.

Determines how many consecutive text segments are grouped together to form
analysis chunks. Larger values create bigger comparison units but may lose
fine-grained alignment details.

Type
----
int

Value
-----
1

Notes
-----
* Range: 1 to reasonable positive integers (typically 1-5)
* 1 = each segment analyzed individually (most granular)
* 2 = pairs of segments grouped together
* 3+ = larger groups, less granular analysis
* Affects computational complexity and alignment precision
* Choose based on segment size and desired analysis granularity

Guidelines
----------
* Use 1 for sentence-level segmentation (most common)
* Use 2-3 for very short segments (phrases, clauses)
* Use larger values for paragraph-level or longer segments
* Consider text length and computational resources

Examples
--------
>>> # Individual segment analysis (default)
>>> result = compute_vcs_score(ref, gen, seg, emb, chunk_size=1)

>>> # Pair-wise segment analysis
>>> result = compute_vcs_score(ref, gen, seg, emb, chunk_size=2)

>>> # Larger chunk analysis for short segments
>>> result = compute_vcs_score(ref, gen, seg, emb, chunk_size=3)

See Also
--------
compute_vcs_score : Detailed chunk_size parameter description
visualize_text_chunks : See how chunking affects text organization
"""