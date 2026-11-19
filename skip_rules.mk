# Model applicability rules shared by Makefile and scripts/conclude_simulation.py.
#
# IMPORTANT:
# - This file is also parsed by scripts/conclude_simulation.py.
# - Define ALL_MODELS, UNIVARIATE_MODELS, and MULTIVARIATE_MODELS explicitly.
# - Do not use variable references in these three definitions or in ALL_ALGOS.
# - In *_APPLY_MODELS, only the three model-group variables above may be referenced.
# - Each *_APPLY_MODELS value must be either a literal model list or one $(GROUP).
# - Update ALL_MODELS manually when adding or removing models.
# - If variable names or syntax change, update the parser in
#   scripts/conclude_simulation.py accordingly.

# Model groups (explicit lists only)
ALL_MODELS := U1 U2 U3 U4 U5 U6 U7 M1 M2 M3 M4
UNIVARIATE_MODELS := U1 U2 U3 U4 U5 U6 U7
MULTIVARIATE_MODELS := M1 M2 M3 M4

# Algorithms
ALL_ALGOS := ddpm gcds hyalg hy flex deepcde

# Applicable models for each algorithm
DDPM_APPLY_MODELS := $(ALL_MODELS)
GCDS_APPLY_MODELS := $(ALL_MODELS)
HY_APPLY_MODELS := U1 U2 U3 U4 U7
FLEX_APPLY_MODELS := $(UNIVARIATE_MODELS)
DEEPCDE_APPLY_MODELS := $(UNIVARIATE_MODELS)
