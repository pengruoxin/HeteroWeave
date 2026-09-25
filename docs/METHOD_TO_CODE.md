# Method-to-code map

## Variable-cardinality composition

`HorizontalCompositeBlock` in
`mmcls_addon/models/backbones/dery.py` constructs all selected branches at one
functional position. Every branch has an optional input interface, a reusable
pretrained block, and an optional output interface. Branch outputs are aligned
and fused.

The fusion operator is not part of the architecture search space. During
search, all multi-branch candidates use parameter-free arithmetic mean fusion,
which is exposed by the historical `operator='sum'` configuration. Despite the
name, the implementation calls `stacked.mean(dim=0)`, so feature magnitude does
not grow linearly with cardinality. After architecture selection, full training
may use either the same mean fusion or `operator='gate'`, a learnable
softmax-gated fusion. The released HeteroWeave (30M/6G) configuration uses
mean fusion, HeteroWeave-P (10M/3G) uses gated fusion, and HeteroWeave-E has no
multi-branch position and therefore requires no fusion operator.

`NeuralAdapter` contains the deterministic CNN-to-CNN, CNN-to-token,
token-to-CNN, and token-to-token interfaces. `DeRy` is the inherited registry
name of the assembled backbone. It is not a second method in this code path.

`heteroweave/composition.py` provides a small validation reference for the
nonempty subset constraint. If one pretrained source is instantiated twice,
the instances must have distinct identifiers; repeating one identifier is not
representable as a mathematical set.

## CLAS

`heteroweave/clas.py` states the proxy calculation without framework
dependencies. The experiment implementation collects binary activation
patterns at evaluated functional positions and applies square-root aggregation
to position-level counts. Full hooks are in:

- `tools/search_nsga3_multiobj.py` for ImageNet search;
- `tools/run_clip_heteroweave_search_pool.py` for retrieval;
- `PositionLayerSwapCollector` in `tools/run_deeplab_heteroweave_search.py`;
- `tools/evaluate_detection_formal_proxy.py` for detection.

The paper-facing proxy name is **CLAS**. A low-level internal score key,
`layer_swap_sqrt`, is retained only inside legacy activation-counting utilities;
public search commands and exported records use **CLAS**.

## Three-objective search

The paper-level objective directions are: maximize trained performance,
minimize parameter count, and minimize FLOPs. Because trained performance is
not available for every search candidate, **CLAS** is used only as its
training-free proxy during search; parameter count and FLOPs are computed
directly from each candidate. The search therefore optimizes
`[maximize CLAS, minimize parameters, minimize FLOPs]`.
`tools/search_nsga3_multiobj.py` contains the NSGA-III search and genotype
operators. `simlarity/multi_objective.py` contains the original general ranking
helpers; `heteroweave/pareto.py` is the concise reviewer-facing reference
implementation.

Representative selection must be performed only after strict nondominance is
computed. Performance and efficiency labels describe selected operating points;
they are not separate training methods.
