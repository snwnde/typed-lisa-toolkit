# Changelog
All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

- - -
## 0.3.7 - 2025-04-22
#### Bug Fixes
- make defaults private - (487d445) - Senwen Deng
- type hinting - (391db11) - Senwen Deng
#### Documentation
- fix scipy crossref - (8e7456d) - Senwen Deng
- noisemodel - (b38e153) - Senwen Deng
- scipy crossref - (0cf71c5) - Senwen Deng
#### Miscellaneous Chores
- rewording - (8e2b070) - Senwen Deng
- typo - (d40fa90) - Senwen Deng
#### Refactoring
- FDNoiseModel make - (5c1fb94) - Senwen Deng
- remove unused - (d5b2350) - Senwen Deng

- - -

## 0.3.6 - 2025-04-11
#### Bug Fixes
- wf typing - (8c0d181) - Senwen Deng
- type string union - (17a194b) - Senwen Deng
- waveform typing - (a556c7c) - Senwen Deng
#### Features
- func to make dense waveform reps - (757f96d) - Senwen Deng
#### Tests
- dense maker - (b62dbec) - Senwen Deng

- - -

## 0.3.5 - 2025-04-10
#### Bug Fixes
- python version requirement - (66fd518) - Senwen Deng

- - -

## 0.3.4 - 2025-03-28
#### Bug Fixes
- noisemodel instance init - (899aee4) - Senwen Deng
#### Documentation
- fix GB tutorial - (407fbeb) - Senwen Deng

- - -

## 0.3.3 - 2025-03-28
#### Bug Fixes
- fft convention - (638b3e0) - Senwen Deng
#### Miscellaneous Chores
- cog config - (5618971) - Senwen Deng

- - -

## 0.3.2 - 2025-03-26
#### Refactoring
- noisemodel name convention - (b67a97d) - Senwen Deng
- removed ldc dependency - (6e20e4c) - Senwen Deng

- - -

## 0.3.1 - 2025-03-20
#### Bug Fixes
- rfft, irfft, stfft dimension - (869498e) - Senwen Deng
- ts plotter type hints - (16a41d7) - Senwen Deng
- sft times - (f9300d0) - Senwen Deng
- legend kwargs - (7fe2064) - Senwen Deng
#### Features
- optional residual plot in compare - (3b6203c) - Senwen Deng

- - -

## 0.3.0 - 2025-01-14
#### Documentation
- improved gb example - (1d2c1f2) - Senwen Deng
#### Features
- more fs plot units - (19e6c85) - Senwen Deng
- time-freq representation - (af26103) - Senwen Deng
#### Refactoring
- abstractify `Representation` - (9039193) - Senwen Deng

- - -

## 0.2.4 - 2025-01-05
#### Bug Fixes
- comparison plot labels - (e77da02) - Senwen Deng
- add `__neg__` for `_NullValue` - (e49b2e4) - Senwen Deng
- value null arithdict - (a6dcbb3) - Senwen Deng
- keep names of `_SeriesData` - (0c8f4d4) - Senwen Deng
- type hinting `get_whitened` - (cc994f5) - Senwen Deng
- improved type hinting - (ba94d97) - Senwen Deng
- `get_integrand` factor order - (65aa0b6) - Senwen Deng
- use a protocol to annotate tapering callables - (c701864) - Senwen Deng
#### Documentation
- correct `get_interpolated` doc string - (f50e4bc) - Senwen Deng
#### Features
- take slice subset - (d53f28d) - Senwen Deng
#### Miscellaneous Chores
- rename `_NullDict` to `_NullValue` - (53320a9) - Senwen Deng
- remove trailing whitespace - (da8cb32) - Senwen Deng
#### Refactoring
- improve arithdicts arithmetics - (b7ec08a) - Senwen Deng

- - -

## 0.2.3 - 2024-11-01
#### Bug Fixes
- correct phasor representation - (3033cd9) - Senwen Deng
- improved series plotting - (c242bef) - Senwen Deng
- correct `to_tsdata` and `to_fsdata` - (2df64fb) - Senwen Deng
- align with the design purpose for `draw` & `plot` - (2154d3e) - Senwen Deng
#### Features
- add `get_embedded` - (8fac2cb) - Senwen Deng
#### Tests
- adapt for phasor - (4dded23) - Senwen Deng

- - -

## 0.2.2 - 2024-10-31
#### Bug Fixes
- warning message for `to_tsdata` - (eb70bf0) - Senwen Deng
- behaviour of `_draw` - (5cffe83) - Senwen Deng
- unwrap the angles for phasor - (e7d79e6) - Senwen Deng
- add back `dt` attribute for TimedFSData - (fb32630) - Senwen Deng
- promote `sum` method to `ArithDict` class - (4bb8da0) - Senwen Deng
- type annotations for phasor - (1ee219c) - Senwen Deng
#### Features
- allow custom interpolators for sensitivity - (a66d873) - Senwen Deng

- - -

## 0.2.1 - 2024-10-28
#### Bug Fixes
- do not use trim_interp - (e7f614b) - Senwen Deng
#### Documentation
- update waveforms - (6641548) - Senwen Deng

- - -

## 0.2.0 - 2024-10-26
#### Bug Fixes
- type hinting for waveforms - (3fa78a1) - Senwen Deng
- renamed cross-correlation - (20b9bd5) - Senwen Deng
#### Documentation
- minor update - (0da1cac) - Senwen Deng
- add tutorial for sampling GB - (d6f9228) - Senwen Deng
- add `extend_to` - (7d9434d) - Senwen Deng
- add likelihood module documentation - (6409eae) - Senwen Deng
- improve `get_cross_correlation` docstring - (11b7563) - Senwen Deng
- add sensitivity - (581c1e9) - Senwen Deng
#### Features
- add `extend_to` utility function - (ff8e22b) - Senwen Deng
- add likelihood - (3112c9f) - Senwen Deng
- add Template protocol - (c132d53) - Senwen Deng
- implemented sensitivity - (7d100ef) - Senwen Deng
#### Refactoring
- amend WhittleLikelihood - (0640b10) - Senwen Deng
- rename FDTemplate to FSWaveformGen - (80405aa) - Senwen Deng
- redesign `get_noise_psd` - (4d71b93) - Senwen Deng
#### Tests
- add test_likelihood.py - (d42f056) - Senwen Deng
- minor fix - (59dbfd9) - Senwen Deng
- add test_sensitivity.py - (b7fb088) - Senwen Deng

- - -

## 0.1.0 - 2024-10-22
#### Bug Fixes
- improve the signature of `load_ldc_data` - (49f9ca3) - Senwen Deng
#### Continuous Integration
- fix ci.yml - (ab9b39c) - Senwen Deng
- suppress warning for pip - (543fa87) - Senwen Deng
- bring back pages deployment - (a75d1f7) - Senwen Deng
- reduce the number of jobs - (3433fb4) - Senwen Deng
- add docs-requirements.txt - (93310c7) - Senwen Deng
- add a CI to build the documentation - (bcdb806) - Senwen Deng
#### Documentation
- correct typo - (dcf714c) - Senwen Deng
- update quickstart guide - (d2df38a) - Senwen Deng
- reorganize index - (7b61155) - Senwen Deng
- minor fix plotters docstring - (477191d) - Senwen Deng
- fix docs-requirements.txt - (c054191) - Senwen Deng
- add plotters module documentation - (36d8874) - Senwen Deng
- update docs-requirements.txt - (4f3e274) - Senwen Deng
- correct the docstring of `save` - (12eeda4) - Senwen Deng
- improve documentation for representations - (69ec6ba) - Senwen Deng
- update accordingly docs - (847befe) - Senwen Deng
- moved rst content to docstrings - (a3dede9) - Senwen Deng
- add sphinx - (1170a50) - Senwen Deng
#### Features
- Add a function to read LDC data - (ed990b6) - Senwen Deng
- Add `save` and `load` methods to _SeriesData - (a3e15fe) - Senwen Deng
- add trim interpolation - (20d4701) - Senwen Deng
- add `to_phasor` for freq series - (4014cc4) - Senwen Deng
- sum for series of different lengths - (9ee8dc8) - Senwen Deng
- add arithmetic operations for `PhasorSequence` - (eeb1643) - Senwen Deng
- initial project implementation - (d4740c0) - Senwen Deng
#### Miscellaneous Chores
- correct type hinting import in plotters.py - (4c5d549) - Senwen Deng
- update README & format representations.py - (42ed614) - Senwen Deng
#### Refactoring
- clean up viz module - (7e9d27d) - Senwen Deng
- polish representation type hints - (a5b2a23) - Senwen Deng
- reconsider representations and waveforms - (9b1d975) - Senwen Deng
- change `real` & `imag` from methods to properties - (be2ec47) - Senwen Deng
- improved `modes.py` - (a2481f3) - Senwen Deng
#### Tests
- fix tests - (2d23e1b) - Senwen Deng
- fix testing files - (eea7465) - Senwen Deng

- - -

Changelog generated by [cocogitto](https://github.com/cocogitto/cocogitto).