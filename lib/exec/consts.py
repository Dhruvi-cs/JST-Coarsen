
# WandB settings
WDB_PROJECT = "DynGraphCoarseningV4"
WDB_MODE= "disabled"  # "offline", "online", "disabled"

CLF_TASK = 'clf'
REG_TASK = 'reg'
TRANS = 'trans' # transductive learning
INDUC = 'induc' # inductive learning
SMALL_NUMB = 1e-6
NE_THRESH = 5  
# threshold for number of edges left after coarsening,
# if any snapshot has <= this number of edges left, we skip this run

COARS_SCORE_TAG="symscore"  # tag to identify coarsened datasets

SPLIT_PROPS = (6,2,2)  # train/val/test split proportions
DIR = "log"
LR_DECAY = 10
LR_THRES = 1e-5 * (1 + 1 / LR_DECAY) # (stop training when lr < LR_THRES), lowest values would be 1e-4
IMP_ABS = 0
IMP_REL = 0

#
MAX_CACHE_SIZE = 64  # max number of windowed snapshots to cache in memory
TEMP_AGG_METHOD = 'max'

NEED_ASSERTIONS = False  # whether to perform assertions during execution