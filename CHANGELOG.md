# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add link to draf demo paper and its case studies.
- Add `maxsell` and `maxbuy` to `EG` component template.
- Add wind turbine (`WT`) component.
- Add hydrogen components (`FC`, `Elc`, `H2S`).
- Add `H_level_target` to thermal generation components (`HOB`, `CHP`, `P2H`, `HP`).
- Add `cs.plot.collector_table()`.
- Add caption to `cs.plot.tables()`.
- Add option to provide series in `sc.plot.ts_balance()`.
- Add `only_scalars` and `number_format` arguments to `cs.plot.table()`.
- Add `cs.plot.capa_TES_table()`.
- Add `draf.helper.get_TES_volume()`.
- Add functionality to update dimensions with dictionary through `cs.add_scen()`.
- Add `sc.update_var_bound()`, `sc.update_upper_bound()`, `sc.update_lower_bound()`.
- Add `cs.scens.get()`.
- Add option to interate over `cs.scens`.
- Add `draf.helper.play_beep_sound()` and `play_sound` argument in `cs.optimize()`.
- Add possibility to use a solved scenario as base scenario.

### Removed

- Remove `sc.has_thermal_entities` property.

### Fixed

- Fix disappearing collector values and solving times, when solving scenarios in parallel.
- Fix missing registration of feed-in (FI) in `P_EL_source_T` for component templates `PV` and `CHP`.

### Changed

- Rename heat downgrading component `H2H1` to `HD`.

## [v0.2.0] - 2022-05-10

[Unreleased]: https://github.com/DrafProject/draf/compare/v0.2.0...HEAD
[v0.2.0]: https://github.com/DrafProject/draf/releases/tag/v0.2.0
