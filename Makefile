.PHONY: all U1 U2 U3 U4 U5 U6 U7 M1 M2 M3 M4 clean help \
		generate generate-all \
		eval eval-all \
        train-ddpm-single train-ddpm sample-ddpm-single sample-ddpm \
		train-gcds-single sample-gcds-single train-gcds sample-gcds \
		train-hy-single sample-hy-single train-hy sample-hy \
		train-deepcde-single sample-deepcde-single train-deepcde sample-deepcde \
		train-flex-single sample-flex-single train-flex sample-flex \
		train-drf-single sample-drf-single train-drf sample-drf \
		train-ctmtf-single train-ctmtf sample-ctmtf-single sample-ctmtf \
		conclude \
		conclude-ddpm conclude-gcds conclude-hy conclude-deepcde conclude-flex conclude-ctmtf conclude-drf \
		train-srm-single sample-srm-single train-srm sample-srm eval-srm conclude-srm \

# === General Python call with local imports ===
PYTHON := PYTHONPATH=. python

# Windows
# PYTHON := python

# Shared method/model applicability settings (also read by conclude_simulation.py)
include skip_rules.mk

# REPEATED RUN SETTING: SEEDS
ALL_SEEDS := 230 460 690 920 1150 1380 1610 1840 2070 2300

# === DEFAULT VALUE FOR TRAINING and VALIDATION and TEST SAMPLE SIZE ===
TRAIN  ?= 5000
VALID  ?= 2000
TEST   ?= 2000
ORACLE ?= 500 # Number of samples used to compute ES_oracle

# === SAMPLING SIZE ===
SAMPLES ?= 2000

# === Default DATA_MODEL & SEED & ALGO ===
# ALGO is used for evaluation

DATA_MODEL ?= U1
SEED   ?= 230
#ALGO ?= ddpm



# === DEFINE THE TARGET_MODELS, TARGET_SEEDS and TARGET_ALGOS ===

# *** Note that this is used for repeated run with among different models and seeds ***

# check if DATA_MODEL is given by command line
ifeq ($(origin DATA_MODEL), command line)
    TARGET_MODELS := $(DATA_MODEL)
else
    TARGET_MODELS := $(ALL_MODELS)
endif

# check if SEED is given by command line
ifeq ($(origin SEED), command line)
    TARGET_SEEDS := $(SEED)
else
    TARGET_SEEDS := $(ALL_SEEDS)
endif

# check if ALGO is given by command line
ifeq ($(origin ALGO), command line)
    TARGET_ALGOS := $(ALGO)
else
    TARGET_ALGOS := $(ALL_ALGOS)
endif

# Models used by each algorithm in repeated train/sample runs
DDPM_TARGET_MODELS    := $(filter $(DDPM_APPLY_MODELS),$(TARGET_MODELS))
GCDS_TARGET_MODELS    := $(filter $(GCDS_APPLY_MODELS),$(TARGET_MODELS))
HY_TARGET_MODELS      := $(filter $(HY_APPLY_MODELS),$(TARGET_MODELS))
DEEPCDE_TARGET_MODELS := $(filter $(DEEPCDE_APPLY_MODELS),$(TARGET_MODELS))
FLEX_TARGET_MODELS    := $(filter $(FLEX_APPLY_MODELS),$(TARGET_MODELS))
SRM_TARGET_MODELS     := $(filter $(SRM_APPLY_MODELS),$(TARGET_MODELS))
CTMTF_TARGET_MODELS   := $(filter $(CTMTF_APPLY_MODELS),$(TARGET_MODELS))
DRF_TARGET_MODELS     := $(filter $(DRF_APPLY_MODELS),$(TARGET_MODELS))


#
# =====================
# Data Generation
# =====================
GENERATE_SCRIPT = scripts.generate_synthetic
OUTDIR ?= datasets/synthetic

generate:
	$(PYTHON) -m $(GENERATE_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) --train-samples $(TRAIN) --valid-samples $(VALID) --test-samples $(TEST) --oracle-samples $(ORACLE) --out-dir $(OUTDIR)



GENERATE_ALL_CMD := $(foreach m,$(TARGET_MODELS),$(foreach s,$(TARGET_SEEDS),$(MAKE) generate DATA_MODEL=$(m) SEED=$(s) && ))
generate-all:
	@$(GENERATE_ALL_CMD) echo "All Generation Tasks Completed!"


clean:
	rm -rf $(OUTDIR)


# =====================
#    DDPM training
# =====================
TRAIN_DDPM_SCRIPT = scripts.train_ddpm
DDPM_TIME_STEP ?= 300
DDPM_BATCH  ?= 128
DDPM_EPOCHS ?= 100
DDPM_LR     ?= 1e-2
DDPM_LR_DROP_FACTOR ?= 0.5
DDPM_LR_DROP_PERIOD ?= 10
DDPM_HIDDEN_DIMS ?= 1024,512,256
DDPM_EARLY_STOP ?= --early-stop
DDPM_PATIENCE ?= 20
# Python default: normalization OFF; Make enables it via this flag.
DDPM_NORMALIZATION ?= --normalization

# ---- single run ----
train-ddpm-single:
	$(PYTHON) -m $(TRAIN_DDPM_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--timesteps $(DDPM_TIME_STEP) \
		--batch-size $(DDPM_BATCH) \
		--epochs $(DDPM_EPOCHS) \
		--hidden-dims $(DDPM_HIDDEN_DIMS) \
		--lr $(DDPM_LR) \
		--lr-drop-factor $(DDPM_LR_DROP_FACTOR) \
		--lr-drop-period $(DDPM_LR_DROP_PERIOD) \
		--patience $(DDPM_PATIENCE) \
		$(DDPM_EARLY_STOP) \
		$(DDPM_NORMALIZATION)

# ---- all runs ----
DDPM_TRAIN_ALL_CMD := \
	$(foreach m,$(DDPM_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-ddpm-single DATA_MODEL=$(m) SEED=$(s) && ))

train-ddpm:
	@$(DDPM_TRAIN_ALL_CMD) echo "All DDPM Training Completed!"


# =====================
#    DDPM sampling
# =====================
SAMPLE_DDPM_SCRIPT = scripts.sample_ddpm
DDPM_MODEL_CHOICE ?= best

# ---- single run ----
sample-ddpm-single:
	$(PYTHON) -m $(SAMPLE_DDPM_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--batch-size $(DDPM_BATCH) \
		--model-choice $(DDPM_MODEL_CHOICE)

# ---- all runs ----
DDPM_SAMPLE_ALL_CMD := \
	$(foreach m,$(DDPM_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-ddpm-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-ddpm:
	@$(DDPM_SAMPLE_ALL_CMD) echo "All DDPM Sampling Completed!"



# =====================
#    GCDS training
# =====================
TRAIN_GCDS_SCRIPT = scripts.train_gcds
GCDS_BATCH ?= 128
GCDS_EPOCHS ?= 500
GCDS_LR ?= 1e-4
GCDS_HIDDEN_DIM ?= 50


# ---- single run ----
train-gcds-single:
	$(PYTHON) -m $(TRAIN_GCDS_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--batch-size $(GCDS_BATCH) \
		--epochs $(GCDS_EPOCHS) \
		--lr $(GCDS_LR) \
		--hidden-dim $(GCDS_HIDDEN_DIM)

# ---- all runs ----
GCDS_TRAIN_ALL_CMD := \
	$(foreach m,$(GCDS_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-gcds-single DATA_MODEL=$(m) SEED=$(s) && ))

train-gcds:
	@$(GCDS_TRAIN_ALL_CMD) echo "All GCDS Training Completed!"


# =====================
#    GCDS sampling
# =====================

SAMPLE_GCDS_SCRIPT = scripts.sample_gcds
GCDS_NOISE_DIM ?= 3
GCDS_MODEL_CHOICE ?= last

# ---- single run ----
sample-gcds-single:
	$(PYTHON) -m $(SAMPLE_GCDS_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--batch-size $(GCDS_BATCH) \
		--noise-dim $(GCDS_NOISE_DIM) \
		--hidden-dim $(GCDS_HIDDEN_DIM) \
		--model-choice $(GCDS_MODEL_CHOICE)

# ---- all runs ----
GCDS_SAMPLE_ALL_CMD := \
	$(foreach m,$(GCDS_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-gcds-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-gcds:
	@$(GCDS_SAMPLE_ALL_CMD) echo "All GCDS Sampling Completed!"




# =====================
#    HY training / automatic bandwidth tuning (one model and seed)
# =====================
TRAIN_HY_SCRIPT = scripts.train_hy
HY_RADIUS ?= 1.0
# Leave blank for the 21-pair search; specify BOTH for manual fitting.
HY_H ?=
HY_H_CDF ?=
HY_N_SUB_VAL ?= 200
HY_NUM_Y_GRID_TUNING ?= 50
# HY theta estimation: batched float32 Adam on CUDA when available (otherwise CPU).
HY_ADAM_LR ?= 0.01
HY_ADAM_MAX_EPOCHS ?= 300
HY_ADAM_PATIENCE ?= 30

train-hy-single:
ifneq ($(filter $(DATA_MODEL),$(HY_APPLY_MODELS)),)
	$(PYTHON) -m $(TRAIN_HY_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--radius $(HY_RADIUS) \
		--n-sub-val $(HY_N_SUB_VAL) \
		--ny-tuning $(HY_NUM_Y_GRID_TUNING) \
		--adam-lr $(HY_ADAM_LR) \
		--adam-max-epochs $(HY_ADAM_MAX_EPOCHS) \
		--adam-patience $(HY_ADAM_PATIENCE) \
		$(if $(strip $(HY_H)),--h $(HY_H),) \
		$(if $(strip $(HY_H_CDF)),--H $(HY_H_CDF),)
else
	@echo "HY is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

HY_TRAIN_ALL_CMD := \
	$(foreach m,$(HY_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-hy-single DATA_MODEL=$(m) SEED=$(s) && ))
train-hy:
	@$(HY_TRAIN_ALL_CMD) echo "All HY Training Completed!"

# =====================
#    HY sampling (uses H saved by train-hy)
# =====================
SAMPLE_HY_SCRIPT = scripts.sample_hy
HY_NUM_Y_GRID ?= 200

sample-hy-single:
ifneq ($(filter $(DATA_MODEL),$(HY_APPLY_MODELS)),)
	$(PYTHON) -m $(SAMPLE_HY_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--ny $(HY_NUM_Y_GRID)
else
	@echo "HY is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

HY_SAMPLE_ALL_CMD := \
	$(foreach m,$(HY_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-hy-single DATA_MODEL=$(m) SEED=$(s) && ))
sample-hy:
	@$(HY_SAMPLE_ALL_CMD) echo "All HY Sampling Completed!"


# =====================
#   DEEPCDE training
# =====================

TRAIN_DEEPCDE_SCRIPT = scripts.train_deepcde
DEEPCDE_BATCH ?= 128
DEEPCDE_EPOCHS ?= 500
DEEPCDE_LR ?= 1e-4
DEEPCDE_HIDDEN_DIMS ?= 32,64,32
DEEPCDE_EARLY_STOP ?= --early-stop
DEEPCDE_PATIENCE ?= 50
DEEPCDE_BASIS ?= cosine
DEEPCDE_NUM_BASIS ?= 31

# --- single run ---
train-deepcde-single:
ifneq ($(filter $(DATA_MODEL),$(DEEPCDE_APPLY_MODELS)),)
	$(PYTHON) -m $(TRAIN_DEEPCDE_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--basis $(DEEPCDE_BASIS) \
		--n-basis $(DEEPCDE_NUM_BASIS) \
		--hidden-dims $(DEEPCDE_HIDDEN_DIMS) \
		--batch-size $(DEEPCDE_BATCH) \
		--epochs $(DEEPCDE_EPOCHS) \
		--lr $(DEEPCDE_LR) \
		$(DEEPCDE_EARLY_STOP) \
		--patience $(DEEPCDE_PATIENCE)
else
	@echo "DeepCDE is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

# --- all runs ---
DEEPCDE_TRAIN_ALL_CMD := \
	$(foreach m,$(DEEPCDE_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-deepcde-single DATA_MODEL=$(m) SEED=$(s) && ))

train-deepcde:
	@$(DEEPCDE_TRAIN_ALL_CMD) echo "All DeepCDE Training Completed!"



# =====================
#   DEEPCDE sampling
# =====================

SAMPLE_DEEPCDE_SCRIPT = scripts.sample_deepcde
DEEPCDE_MODEL_CHOICE ?= best
DEEPCDE_NUM_YGRID ?= 200

# --- single run ---
sample-deepcde-single:
ifneq ($(filter $(DATA_MODEL),$(DEEPCDE_APPLY_MODELS)),)
	$(PYTHON) -m $(SAMPLE_DEEPCDE_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--ny $(DEEPCDE_NUM_YGRID) \
		--batch-size $(DEEPCDE_BATCH) \
		--model-choice $(DEEPCDE_MODEL_CHOICE)
else
	@echo "DeepCDE is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

# --- all runs ---
DEEPCDE_SAMPLE_ALL_CMD := \
	$(foreach m,$(DEEPCDE_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-deepcde-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-deepcde:
	@$(DEEPCDE_SAMPLE_ALL_CMD) echo "All DeepCDE Sampling Completed!"



# =====================
#   FlexCode training
# =====================

TRAIN_FLEX_SCRIPT = scripts.train_flex
FLEX_BASIS ?= cosine
FLEX_MAX_BASIS ?= 31
FLEX_REG_METHOD ?= rf
FLEX_NUM_Y_GRID ?= 200

# --- single run ---
train-flex-single:
ifneq ($(filter $(DATA_MODEL),$(FLEX_APPLY_MODELS)),)
	$(PYTHON) -m $(TRAIN_FLEX_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--basis $(FLEX_BASIS) \
		--max-basis $(FLEX_MAX_BASIS) \
		--reg $(FLEX_REG_METHOD) \
		--ny $(FLEX_NUM_Y_GRID)
else
	@echo "FlexCode is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

# --- all run ---
FLEX_TRAIN_ALL_CMD := \
	$(foreach m,$(FLEX_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-flex-single DATA_MODEL=$(m) SEED=$(s) && ))

train-flex:
	@$(FLEX_TRAIN_ALL_CMD) echo "All FlexCode Training Completed!"


# =====================
#   Flexcode sampling
# =====================

SAMPLE_FLEX_SCRIPT = scripts.sample_flex
# Python default: tuned model OFF; Make enables it via --tuned-model.
FLEX_TUNED_MODEL ?= --tuned-model

# --- single run ---
sample-flex-single:
ifneq ($(filter $(DATA_MODEL),$(FLEX_APPLY_MODELS)),)
	$(PYTHON) -m $(SAMPLE_FLEX_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		$(FLEX_TUNED_MODEL)
else
	@echo "FlexCode is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

# ---  all run ---
FLEX_SAMPLE_ALL_CMD := \
	$(foreach m,$(FLEX_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-flex-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-flex:
	@$(FLEX_SAMPLE_ALL_CMD) echo "All FlexCode Sampling Completed!"


# =====================
#    SRM training
# =====================
TRAIN_SRM_SCRIPT = scripts.train_srm
# SRM_BATCH ?= 128  (.py default: training and sampling)
SRM_BATCH ?= 64
SRM_TRAIN_SAMPLES ?= 30
SRM_EPOCHS ?= 500
SRM_LR ?= 1e-4
SRM_PATIENCE ?= 50
SRM_HIDDEN_DIMS ?= 1024,512,256
SRM_EARLY_STOP ?= --early-stop

# ---- single run ----
train-srm-single:
	$(PYTHON) -m $(TRAIN_SRM_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--batch-size $(SRM_BATCH) \
		--epochs $(SRM_EPOCHS) \
		--lr $(SRM_LR) \
		--hidden-dims $(SRM_HIDDEN_DIMS) \
		--sampling-size-for-training $(SRM_TRAIN_SAMPLES) \
		--patience $(SRM_PATIENCE) \
		$(SRM_EARLY_STOP)

# ---- all runs ----
SRM_TRAIN_ALL_CMD := \
	$(foreach m,$(SRM_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-srm-single DATA_MODEL=$(m) SEED=$(s) && ))

train-srm:
	@$(SRM_TRAIN_ALL_CMD) echo "All SRM Training Completed!"


# =====================
#    SRM sampling
# =====================
SAMPLE_SRM_SCRIPT = scripts.sample_srm

# ---- single run ----
sample-srm-single:
	$(PYTHON) -m $(SAMPLE_SRM_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--batch-size $(SRM_BATCH)

# ---- all runs ----
SRM_SAMPLE_ALL_CMD := \
	$(foreach m,$(SRM_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-srm-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-srm:
	@$(SRM_SAMPLE_ALL_CMD) echo "All SRM Sampling Completed!"


# =====================
#    CTMTF training
# =====================
TRAIN_CTMTF_SCRIPT = scripts.train_ctmtf

# ---- single run ----
train-ctmtf-single:
ifneq ($(filter $(DATA_MODEL),$(CTMTF_APPLY_MODELS)),)
	$(PYTHON) -m $(TRAIN_CTMTF_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED)
else
	@echo "CTMTF is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

# ---- all runs ----
CTMTF_TRAIN_ALL_CMD := \
	$(foreach m,$(CTMTF_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-ctmtf-single DATA_MODEL=$(m) SEED=$(s) && ))

train-ctmtf:
	@$(CTMTF_TRAIN_ALL_CMD) echo "All CTMTF Training Completed!"


# =====================
#    CTMTF sampling
# =====================
SAMPLE_CTMTF_SCRIPT = scripts.sample_ctmtf

# ---- single run ----
sample-ctmtf-single:
ifneq ($(filter $(DATA_MODEL),$(CTMTF_APPLY_MODELS)),)
	$(PYTHON) -m $(SAMPLE_CTMTF_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES)
else
	@echo "CTMTF is not applied to DATA_MODEL=$(DATA_MODEL)"
endif

# ---- all runs ----
CTMTF_SAMPLE_ALL_CMD := \
	$(foreach m,$(CTMTF_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-ctmtf-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-ctmtf:
	@$(CTMTF_SAMPLE_ALL_CMD) echo "All CTMTF Sampling Completed!"


# =====================
#    DRF training
# =====================
TRAIN_DRF_SCRIPT = scripts.train_drf

# ---- single run ----
train-drf-single:
	$(PYTHON) -m $(TRAIN_DRF_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED)

# ---- all runs ----
DRF_TRAIN_ALL_CMD := \
	$(foreach m,$(DRF_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-drf-single DATA_MODEL=$(m) SEED=$(s) && ))

train-drf:
	@$(DRF_TRAIN_ALL_CMD) echo "All DRF Training Completed!"


# =====================
#    DRF sampling
# =====================
SAMPLE_DRF_SCRIPT = scripts.sample_drf

# ---- single run ----
sample-drf-single:
	$(PYTHON) -m $(SAMPLE_DRF_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES)

# ---- all runs ----
DRF_SAMPLE_ALL_CMD := \
	$(foreach m,$(DRF_TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-drf-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-drf:
	@$(DRF_SAMPLE_ALL_CMD) echo "All DRF Sampling Completed!"


#=====================
#	Evaluation
# =====================
EVAL_SCRIPT = scripts.eval

# ---- single run ----
eval-single:
ifeq ($(strip $(ALGO)),)
	@echo "Please specify ALGO (e.g., ALGO=flex)"
else
	@echo evaluating data for this configuration: ALGO=$(ALGO), DATA_MODEL=$(DATA_MODEL), SEED=$(SEED)
	$(PYTHON) -m $(EVAL_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) --algorithm $(ALGO)
endif

# ---- algorithm shortcuts ----
# Each shortcut passes only the data models supported by that algorithm.
# If DATA_MODEL is specified by the user, the corresponding *_TARGET_MODELS
# variable has already been filtered above.
eval-ddpm:
	$(MAKE) eval ALGO=ddpm DATA_MODEL="$(DDPM_TARGET_MODELS)"

eval-hy:
	$(MAKE) eval ALGO=hy DATA_MODEL="$(HY_TARGET_MODELS)"

eval-gcds:
	$(MAKE) eval ALGO=gcds DATA_MODEL="$(GCDS_TARGET_MODELS)"

eval-flex:
	$(MAKE) eval ALGO=flex DATA_MODEL="$(FLEX_TARGET_MODELS)"

eval-deepcde:
	$(MAKE) eval ALGO=deepcde DATA_MODEL="$(DEEPCDE_TARGET_MODELS)"

eval-srm:
	$(MAKE) eval ALGO=srm DATA_MODEL="$(SRM_TARGET_MODELS)"
eval-ctmtf:
	$(MAKE) eval ALGO=ctmtf DATA_MODEL="$(CTMTF_TARGET_MODELS)"
eval-drf:
	$(MAKE) eval ALGO=drf DATA_MODEL="$(DRF_TARGET_MODELS)"

# ---- all runs ----
# Use shell loops rather than expanding every algorithm/model/seed combination
# into one enormous command line. This avoids the OS "argument list too long"
# limit while preserving ALGO, DATA_MODEL, and SEED command-line filters.
eval:
	@set -e; \
	for algo in $(TARGET_ALGOS); do \
		case "$$algo" in \
			ddpm)    models="$(DDPM_TARGET_MODELS)" ;; \
			gcds)    models="$(GCDS_TARGET_MODELS)" ;; \
			hy)      models="$(HY_TARGET_MODELS)" ;; \
			flex)    models="$(FLEX_TARGET_MODELS)" ;; \
			deepcde) models="$(DEEPCDE_TARGET_MODELS)" ;; \
			srm)     models="$(SRM_TARGET_MODELS)" ;; \
			ctmtf)   models="$(CTMTF_TARGET_MODELS)" ;; \
			drf)     models="$(DRF_TARGET_MODELS)" ;; \
			*) echo "Unknown ALGO=$$algo"; exit 1 ;; \
		esac; \
		if [ -z "$$models" ]; then \
			echo "Skipping $$algo: no applicable data models selected."; \
			continue; \
		fi; \
		for m in $$models; do \
			for s in $(TARGET_SEEDS); do \
				$(MAKE) eval-single ALGO=$$algo DATA_MODEL=$$m SEED=$$s; \
			done; \
		done; \
	done; \
	echo "All Evaluation Tasks Completed!"




#=====================
#	Conclusion
# =====================

CONCLUDE_SCRIPT = scripts.conclude_simulation

# Optional filters are passed only when they are explicitly provided on the
# command line. With no filters, conclude_simulation summarizes all methods and
# all data models into workspaces/all.xlsx.
CONCLUDE_ARGS :=
ifeq ($(origin ALGO), command line)
    CONCLUDE_ARGS += --algorithm $(ALGO)
endif
ifeq ($(origin DATA_MODEL), command line)
    CONCLUDE_ARGS += --data-name $(DATA_MODEL)
endif

conclude:
	$(PYTHON) -m $(CONCLUDE_SCRIPT) $(CONCLUDE_ARGS)

# ---- algorithm shortcuts ----
conclude-ddpm:
	$(MAKE) conclude ALGO=ddpm

conclude-hy:
	$(MAKE) conclude ALGO=hy

conclude-gcds:
	$(MAKE) conclude ALGO=gcds

conclude-flex:
	$(MAKE) conclude ALGO=flex

conclude-deepcde:
	$(MAKE) conclude ALGO=deepcde

conclude-srm:
	$(MAKE) conclude ALGO=srm
conclude-ctmtf:
	$(MAKE) conclude ALGO=ctmtf
conclude-drf:
	$(MAKE) conclude ALGO=drf




# === Help menu ===
help:
	@echo ""
	@echo "======================================================="
	@echo "   Synthetic Data Pipeline (DDPM, Flex, GCDS, DeepCDE, HyAlg, HY)"
	@echo "======================================================="
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [VAR=value]..."
	@echo ""
	@echo "-------------------------------------------------------"
	@echo " 1. Main Targets"
	@echo "-------------------------------------------------------"
	@echo "  generate           : Generate a single dataset (uses DATA_MODEL, SEED)"
	@echo "  generate-all       : Generate datasets for target model-seed combinations (can specify DATA_MODEL and SEED)"
	@echo "  train-[algo]       : Train models for target model-seed combinations (can specify DATA_MODEL and SEED)"
	@echo "  sample-[algo]      : Generate samples for target model-seed combinations (can specify DATA_MODEL and SEED)"
	@echo "  eval-[algo]        : Evaluate results for target model-seed combinations (can specify DATA_MODEL and SEED)"
	@echo "  conclude-[algo]    : Summarize simulation results for target models (can specify DATA_MODEL)"
	@echo "  clean              : Remove dataset directories"
	@echo ""
	@echo "-------------------------------------------------------"
	@echo " 2. Control Variables"
	@echo "-------------------------------------------------------"
	@echo "  DATA_MODEL         : Target Model (U1..U7, M1..M4). Default: All"
	@echo "  SEED               : Target Seed (230..). Default: All"
	@echo "  ALGO               : Target Algo (for eval). Default: All"
	@echo ""
	@echo "-------------------------------------------------------"
	@echo " 3. Hyperparameters & Settings"
	@echo "-------------------------------------------------------"
	@echo "  Algorithm-specific settings (Epochs, LR, Hidden Dim, etc.)"
	@echo "  are defined as variables in this Makefile."
	@echo ""
	@echo "  Please check the Makefile directly to view or override defaults."
	@echo "  (e.g., DDPM_EPOCHS, GCDS_LR, HYALG_RADIUS...)"
	@echo ""
	@echo "-------------------------------------------------------"
	@echo " Examples"
	@echo "-------------------------------------------------------"
	@echo "  Ex. Run DeepCDE on U1 using seed=230:"
	@echo "             make generate DATA_MODEL=U1 SEED=230 train-deepcde sample-deepcde eval-deepcde"
	@echo ""
	@echo "  Ex. Run DeepCDE on U1 using 10 different random seeds:"
	@echo "          make generate-all DATA_MODEL=U1 train-deepcde sample-deepcde eval-deepcde conclude-deepcde"
	@echo "      or"
	@echo "          for s in 230 460 690 920 1150 1380 1610 1840 2070 2300; do"
	@echo "            make generate DATA_MODEL=U1 SEED=$s train-deepcde sample-deepcde eval-deepcde || exit 1"
	@echo "          done"
	@echo "          make conclude-deepcde DATA_MODEL=U1"
	@echo ""
