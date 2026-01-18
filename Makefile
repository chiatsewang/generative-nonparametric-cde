.PHONY: all M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 clean help \
		generate generate-all \
		eval eval-all \
        train-ddpm-single train-ddpm sample-ddpm-single sample-ddpm \
		train-gcds-single sample-gcds-single train-gcds sample-gcds \
		train-hyalg-single sample-hyalg-single train-hyalg sample-hyalg \
		train-deepcde-single sample-deepcde-single train-deepcde sample-deepcde \
		train-flex-single sample-flex-single train-flex sample-flex \
		conclude-single conclude \
		conclude-ddpm conclude-gcds conclude-hyalg conclude-deepcde conclude-flex \

# === General Python call with local imports ===
PYTHON := PYTHONPATH=. python

# Windows
# PYTHON := python

# REPEATED RUN SETTING MODELS x SEEDS and ALGOS for evaluation
ALL_MODELS := M1 M2 M3 M4 M5 M6 M7 M8 M9 M10
ALL_SEEDS := 230 460 690 920 1150 1380 1610 1840 2070 2300
ALL_ALGOS := ddpm gcds hyalg flex deepcde


# MODELS TO SKIP FOR SPECIFIC ALGOS
HY_SKIP_MODELS := M7 M8 M10
FLEX_SKIP_MODELS := M10
DEEPCDE_SKIP_MODELS := M10


# === DEFAULT VALUE FOR TRAINING and VALIDATION and TEST SAMPLE SIZE ===
TRAIN  ?= 5000
VALID  ?= 2000
TEST   ?= 2000

# === SAMPLING SIZE ===
SAMPLES ?= 2000

# === Default DATA_MODEL & SEED & ALGO ===
# ALGO is used for evaluation

DATA_MODEL ?= M1
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


#
# =====================
# Data Generation
# =====================
GENERATE_SCRIPT = scripts.generate_synthetic
OUTDIR ?= datasets/synthetic

generate:
	$(PYTHON) -m $(GENERATE_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) --train-samples $(TRAIN) --valid-samples $(VALID) --test-samples $(TEST) --out-dir $(OUTDIR)



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
DDPM_EPOCHS ?= 50
DDPM_LR     ?= 1e-2
DDPM_LR_DROP_FACTOR ?= 0.5
DDPM_LR_DROP_PERIOD ?= 10
DDPM_HIDDEN_DIM ?= 50

# ---- single run ----
train-ddpm-single:
	$(PYTHON) -m $(TRAIN_DDPM_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--timesteps $(DDPM_TIME_STEP) \
		--batch-size $(DDPM_BATCH) \
		--epochs $(DDPM_EPOCHS) \
		--hidden-dim $(DDPM_HIDDEN_DIM) \
		--lr $(DDPM_LR) \
		--lr-drop-factor $(DDPM_LR_DROP_FACTOR) \
		--lr-drop-period $(DDPM_LR_DROP_PERIOD) \
		--normalization

# ---- all runs ----
DDPM_TRAIN_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
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
		--timesteps $(DDPM_TIME_STEP) \
		--num-samples $(SAMPLES) \
		--batch-size $(DDPM_BATCH) \
		--hidden-dim $(DDPM_HIDDEN_DIM) \
		--model-choice $(DDPM_MODEL_CHOICE)

# ---- all runs ----
DDPM_SAMPLE_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
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
	$(foreach m,$(TARGET_MODELS), \
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
	$(foreach m,$(TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-gcds-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-gcds:
	@$(GCDS_SAMPLE_ALL_CMD) echo "All GCDS Sampling Completed!"




# =====================
#    HYALG training
# =====================

TRAIN_HYALG_SCRIPT = scripts.train_hyalg
HYALG_RADIUS ?= 1.0
HYALG_TRAIN_H = 0.7

# --- single run ---
train-hyalg-single:
ifeq ($(filter $(DATA_MODEL),$(HY_SKIP_MODELS)),)
	$(PYTHON) -m $(TRAIN_HYALG_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--radius $(HYALG_RADIUS) \
		--h $(HYALG_TRAIN_H)
else
	@echo "skip train-hyalg-single for DATA_MODEL=$(DATA_MODEL)"
endif

# --- all runs ---
HYALG_TRAIN_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-hyalg-single DATA_MODEL=$(m) SEED=$(s) && ))

train-hyalg:
	@$(HYALG_TRAIN_ALL_CMD) echo "All HYALG Training Completed!"



# =====================
#    HYALG sampling
# =====================

SAMPLE_HYALG_SCRIPT = scripts.sample_hyalg
HYALG_SAMPLE_H ?= 0.8
HYALG_Y_GRID ?= 200

# --- single run ---
sample-hyalg-single:
ifeq ($(filter $(DATA_MODEL),$(HY_SKIP_MODELS)),)
	$(PYTHON) -m $(SAMPLE_HYALG_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--H $(HYALG_SAMPLE_H) \
		--ygrid $(HYALG_Y_GRID)
else
	@echo "skip sample-hyalg-single for DATA_MODEL=$(DATA_MODEL)"
endif

# --- all runs ---
HYALG_SAMPLE_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-hyalg-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-hyalg:
	@$(HYALG_SAMPLE_ALL_CMD) echo "All HYALG Sampling Completed!"




# =====================
#   DEEPCDE training
# =====================

TRAIN_DEEPCDE_SCRIPT = scripts.train_deepcde
DEEPCDE_BATCH ?= 128
DEEPCDE_EPOCHS ?= 500
DEEPCDE_LR ?= 1e-4
DEEPCDE_EARLY_STOP ?= --early-stop
DEEPCDE_PATIENCE ?= 20
DEEPCDE_BASIS ?= cosine
DEEPCDE_NUM_BASIS ?= 31

# --- single run ---
train-deepcde-single:
ifeq ($(filter $(DATA_MODEL),$(DEEPCDE_SKIP_MODELS)),)
	$(PYTHON) -m $(TRAIN_DEEPCDE_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--basis $(DEEPCDE_BASIS) \
		--n-basis $(DEEPCDE_NUM_BASIS) \
		--batch-size $(DEEPCDE_BATCH) \
		--epochs $(DEEPCDE_EPOCHS) \
		--lr $(DEEPCDE_LR) \
		$(DEEPCDE_EARLY_STOP) \
		--patience $(DEEPCDE_PATIENCE)
else
	@echo "skip train-deepcde-single for DATA_MODEL=$(DATA_MODEL)"
endif

# --- all runs ---
DEEPCDE_TRAIN_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
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
ifeq ($(filter $(DATA_MODEL),$(DEEPCDE_SKIP_MODELS)),)
	$(PYTHON) -m $(SAMPLE_DEEPCDE_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--ny $(DEEPCDE_NUM_YGRID) \
		--batch-size $(DEEPCDE_BATCH) \
		--model-choice $(DEEPCDE_MODEL_CHOICE)
else
	@echo "skip sample-deepcde-single for DATA_MODEL=$(DATA_MODEL)"
endif

# --- all runs ---
DEEPCDE_SAMPLE_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
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

# --- single run ---
train-flex-single:
ifeq ($(filter $(DATA_MODEL),$(FLEX_SKIP_MODELS)),)
	$(PYTHON) -m $(TRAIN_FLEX_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--basis $(FLEX_BASIS) \
		--max-basis $(FLEX_MAX_BASIS) \
		--reg $(FLEX_REG_METHOD)
else
	@echo "skip train-flex-single for DATA_MODEL=$(DATA_MODEL)"
endif

# --- all run ---
FLEX_TRAIN_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) train-flex-single DATA_MODEL=$(m) SEED=$(s) && ))

train-flex:
	@$(FLEX_TRAIN_ALL_CMD) echo "All FlexCode Training Completed!"


# =====================
#   Flexcode sampling
# =====================

SAMPLE_FLEX_SCRIPT = scripts.sample_flex
FLEX_TUNED_MODEL ?= --tuned-model
FLEX_NUM_Y_GRID ?= 200

# --- single run ---
sample-flex-single:
ifeq ($(filter $(DATA_MODEL),$(FLEX_SKIP_MODELS)),)
	$(PYTHON) -m $(SAMPLE_FLEX_SCRIPT) --data-name $(DATA_MODEL) --seed $(SEED) \
		--num-samples $(SAMPLES) \
		--ny $(FLEX_NUM_Y_GRID) \
		$(FLEX_TUNED_MODEL)
else
	@echo "skip sample-flex-single for DATA_MODEL=$(DATA_MODEL)"
endif

# ---  all run ---
FLEX_SAMPLE_ALL_CMD := \
	$(foreach m,$(TARGET_MODELS), \
		$(foreach s,$(TARGET_SEEDS), \
			$(MAKE) sample-flex-single DATA_MODEL=$(m) SEED=$(s) && ))

sample-flex:
	@$(FLEX_SAMPLE_ALL_CMD) echo "All FlexCode Sampling Completed!"


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
eval-ddpm:
	$(MAKE) eval ALGO=ddpm

eval-hyalg:
	$(MAKE) eval ALGO=hyalg

eval-gcds:
	$(MAKE) eval ALGO=gcds

eval-flex:
	$(MAKE) eval ALGO=flex

eval-deepcde:
	$(MAKE) eval ALGO=deepcde

# ---- all runs ----
EVAL_ALL_CMD := \
	$(foreach a,$(TARGET_ALGOS), \
		$(foreach m,$(TARGET_MODELS), \
			$(foreach s,$(TARGET_SEEDS), \
				$(MAKE) eval-single ALGO=$(a) DATA_MODEL=$(m) SEED=$(s) && )))

eval:
	@$(EVAL_ALL_CMD) echo "All Evaluation Tasks Completed!"




#=====================
#	Conclusion
# =====================

CONCLUDE_SCRIPT = scripts.conclude_simulation

# ---- single run ----
conclude-single:
ifeq ($(strip $(ALGO)),)
	@echo "Please specify ALGO (e.g., ALGO=flex)"
else
	@echo concluding for this configuration: ALGO=$(ALGO), DATA_MODEL=$(DATA_MODEL)
	$(PYTHON) -m $(CONCLUDE_SCRIPT) --data-name $(DATA_MODEL) --algorithm $(ALGO)
endif

# ---- all runs (no SEED) ----
CONCLUDE_ALL_CMD := \
	$(foreach a,$(TARGET_ALGOS), \
		$(foreach m,$(TARGET_MODELS), \
			$(MAKE) conclude-single ALGO=$(a) DATA_MODEL=$(m) && ))

conclude:
	@$(CONCLUDE_ALL_CMD) echo "All Conclude Tasks Completed!"

# ---- algorithm shortcuts ----
conclude-ddpm:
	$(MAKE) conclude ALGO=ddpm

conclude-hyalg:
	$(MAKE) conclude ALGO=hyalg

conclude-gcds:
	$(MAKE) conclude ALGO=gcds

conclude-flex:
	$(MAKE) conclude ALGO=flex

conclude-deepcde:
	$(MAKE) conclude ALGO=deepcde




# === Help menu ===
help:
	@echo ""
	@echo "======================================================="
	@echo "   Synthetic Data Pipeline (DDPM, Flex, GCDS, DeepCDE, HyAlg)"
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
	@echo "  DATA_MODEL         : Target Model (M1..M10). Default: All"
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
	@echo "  Ex. Run DeepCDE on M1 using seed=230:"
	@echo "             make generate DATA_MODEL=M1 SEED=230 train-deepcde sample-deepcde eval-deepcde"
	@echo ""
	@echo "  Ex. Run DeepCDE on M1 using 10 different random seeds:"
	@echo "          make generate-all DATA_MODEL=M1 train-deepcde sample-deepcde eval-deepcde conclude-deepcde"
	@echo "      or"
	@echo "          for s in 230 460 690 920 1150 1380 1610 1840 2070 2300; do"
	@echo "            make generate DATA_MODEL=M1 SEED=$s train-deepcde sample-deepcde eval-deepcde || exit 1"
	@echo "          done"
	@echo "          make conclude-deepcde DATA_MODEL=M1"
	@echo ""
