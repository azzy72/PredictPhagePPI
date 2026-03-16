# PredictPhagePPI
DTU on predicting phage interactions given phage and target bacterias sequenced genomes as well as interaction

## Setup
Run 
1) "make"
2) chmod +rx ./scripts/*
3) pipreqsnb . --force
4) ./scripts/check_requirements.sh in a unix terminal
5) pip install -e . #@root

### notes
KU library preperation using Hackflex; https://github.com/GaioTransposon/Hackflex

Identify key proteins in genomes:
- Depolymerases; https://link.springer.com/article/10.1186/s12929-023-00928-0
- Anti-defense (CRISPR?)
- Modifying enzymes
- Integrases
