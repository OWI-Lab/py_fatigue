## 1.3.4
- ci(pages): Embed pages in CI/CD pipeline

## 1.3.3
- bump(patch): Bump version to 1.3.3 [skip ci]
- ci(pipeline): Merge release/v1.3.3 to main [skip ci]
- chore(change-log): Update change log

## 1.3.2
- bump(patch): Bump version to 1.3.2 [skip ci]
- ci(pipeline): Merge release/v1.3.2 to main [skip ci]  
- test(material): Increase coverage  
- test(nonlinear): Add few tests on Theil and Pavlou (DCA) stress-life methods  

## 1.3.1
- bump(patch): Bump version to 1.3.1 [skip ci]
- ci(pipeline): Merge release/v1.3.1 to main [skip ci]  

## 1.3.0
- bump(minor): Bump version to 1.3.0 [skip ci]
- ci(pipeline): Merge release/v1.3.0 to main [skip ci]  
- Merge pull request #9 from OWI-Lab/feature/#8-pavlou-non-linear-damage-accumulation-model  
- feat(nonlinear): Add Theil and Pavlou (DCA) stress-life methods  
- feat(stress_life): Add calc_nonlinear_damage_with_dca function  

## 1.2.2
- bump(patch): Bump version to 1.2.2 [skip ci]
- ci(pipeline): Merge release/v1.2.2 to main [skip ci]  
- chore(notebooks): Move notebooks to Binder Repo  
- Merge branch 'main' of https://github.com/OWI-Lab/py_fatigue  
- docs(binder): Add badge  

## 1.2.1
- bump(patch): Bump version to 1.2.1 [skip ci]
- ci(pipeline): Merge release/v1.2.1 to main [skip ci]  
- ci(bump): Add bump step to cicd pipeline [skip ci]  
- chore(relative-imports): Edit relative imports  

## 1.2.0
- bump(minor): 1.2.0 <- 1.1.2  
- ci(uv): Add uv integration in cd  
- ci(uv): Add uv integration in pages  
- ci(uv): Add uv integration in ci  
- docs(material): Upgraded Paris and SN curve definition  
- feat(material): Upgraded Paris and SN curve definition  

## 1.1.2
- bump(patch): 1.1.2  
- Update pages.yml  
- docs(pipeline): Add pipeline #4  

## 1.1.1
- bump(patch): 1.1.1  
- docs(pipeline): Add pipeline #3  
- docs(pipeline): Add pipeline #2  
- docs(pipeline): Add pipeline  

## 1.1.0
- fix(gitignore): Add docs folder  
- fix(pyproject): Fix CI pipeline  
- fix(pyproject): Fix dependencies. Max py must be 3.10 because of numba 0.56. From numba 0.57 onwards, support for py 3.8 has been dropped  
- fix(qa): Remove mypy for now since new version gives issues  
- Add test for support from 3.8 to 3.13  
- Drop poetry + multiple fixes  

## 1.0.20
- fix(dev): Fixed unwanted pylint update in pyproject.toml that broke quality assessment  
- doc(dev): #6 Run sphinx-build (inv docs)  
- ver(dev): Bumpversion patch 1.0.20  
- doc(dev): #6 Fixed error in get_des and get_dem documentation  

## 1.0.19
- new(doc): Updated documentation for version 1.0.19  
- new(usr): Improved badges  

## 1.0.18
- new(doc): Updated documentation for version 1.0.18  
- bumpversion 1.0.18  
- Update README.md  
- Update cd.yml  

## 1.0.16
- bumpversion 1.0.16  
- fix(usr): Added check on CycleCounts having different units being added  
- fix(doc): Run sphinx  
- fix(usr): Run quality assessment  

## 1.0.15
- bumpversion 1.0.15  
- fix(usr): Added unit to cycle_count.from_rainflow and to cycle_count.from_timeseries !wip  

## 1.0.14
- bumpversion 1.0.14  
- new(pkg): Edited README.md  
- new(pkg): Edited workflow  

## 1.0.13
- bumpversion 1.0.13  
- new(pkg): Added badges to README.md  
- new(pkg): Updated coverage.yml  
- new(pkg): Added coverage.yml  

## 1.0.12
- bumpversion 1.0.12  
- new(pkg): Added build without publish !wip  
- fix(dev): Misc changes !wip  
- fix(pkg): Publish runs only on branches main and release*  
- fix(dev): Run quality assessment before publishing  

## 1.0.11
- bumpversion 1.0.11  
- new(usr): Added save_residuals bool flag to aggregate_cc to speed up analysis !wip  

## 1.0.10
- bumpversion 1.0.10  
- fix(dev): Solved some circular imports  
- new(usr): Added function calc_aggregated_damage to pf.cycle_count.utils !wip  
- new(usr): Release 1.0.9  

## 1.0.8
- bumpversion 1.0.8  
- new(dev): Added an aggregation function !wip  
- fix(dev): Fixed case where error is thrown if no hist key is available in rainflow !bugfix  

## 1.0.6
- bumpversion 1.0.6 !bumpversion  
- new(usr): #6.0 Added unit property to cyclecount !feature  
- new(usr): #5.0 Added statistical moments to cyclecount !feature  

## 1.0.5
- bumpversion 1.0.5  
- fix(dev): #4.0 Substituted pietrodantuono with OWI-Lab links  
- Ownership transferred to OWI-Lab  

## 1.0.0
- bumpversion 1.0.0  
- new(dev): #1.0 First commit !wip  
- new(dev): #0.0 Initial commit !wip