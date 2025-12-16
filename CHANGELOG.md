# Project Changelog and Provenance Summary

## Versions

### v0
- Legacy code, experiments, and datasets
- Structure is nonexistent

### v1
- Refactored and documented from v1 code
- Uses modular code, config files, metadata files

#### v1.1
- Added gradient skipping and percentile scores as options for preprocessing and training
- These features are not tested but moving onto next features and refactoring (note the bugs and fix in v2.0)

---

## Roadmap

Planned for v2.x:

- [ ] v2.0.0: rework 06_esp_pipeline.ipynb
- [ ] v2.1.0: structural refactor, config versioning, logging improvements
- [ ] v2.2.0: support new model types (e.g., GRU-based predictor)
- [ ] v2.3.0: add new training/evaluation pipelines
- [ ] v2.4.0: add zero-prescriptor
- [ ] v2.5.0: add unrolling experiments and visualizations

---

## Major Transitions
- May 2025: Code Refactoring
- November 2025: Smaller-scale refactoring, closer to data structures used in NeuroAI ESP