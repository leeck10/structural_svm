# structural_svm

package "ssvm_tool"

version "2013.7"

purpose "A command line utility to train/test a Structural SVM (SSVM) model."


# section "Training options"

option "cost" c "set cost of SSVMs" float default="1" no

option "model" m "set model file name" string no

option "binary" b "save/load model in binary format" flag off

option "source" - "source model file for domain adaptation" string no 

option "skip_eval" - "skip test set evaluation in the middle of training" flag off

option "owps_format" - "use One Word Per Sentence (OWPS) format" flag off

option "hash" - "use hash feature and set number of predicates" int default="0" no

option "support" s "use support feature (default is all feature)" flag off

option "general" - "general feature mode (feature:value format) (default is binary feature)" flag off

option "random" r "use random_shuffle in train_data (disabled if use 0)" int default="0" no

option "train_num" - "set number of sentence in train_data for training (for experiments) (disabled if use 0)" int default="0" no

option "verbose" v "verbose mode" flag off


# section "Structural SVM options"

option "epsilon" e "set epsilon (fsmo, fsmo_joint)" float default="0.01" no

option "buf" - "set the number of new constraints to accumulated before recomputing the QP (fsmo, pegasos)" int default="100" no

option "rm_inactive" - "inactive constraints are removed (iteration) (fsmo, fsmo_joint)" int default="50" no

option "final_opt" - "do final optimal check in shrinking (fsmo, fsmo_joint)" flag off

option "comment" - "use comment info in save_slack() (fsmo)" flag off


# section "Pegasos options"

option "iter" i "iterations for training algorithm (pegasos)" int default="100" no

option "period" - "save model periodically (pegasos)" int default="0" no


# section "Latent Strucutral SVM options"

option "latent" - "use CCCP-based Latent SSVM (doesn't support sequence labeling)" flag off

option "latent_SPL" - "use Self-Paced Learning for Latent SSVM (doesn't support sequence labeling)" flag off


# section "Joint Strucutral SVM options"

option "joint" - "use Joint model (y+z) using modified Latent SSVM   (with --y_data, --z_data, --y_cost, and --z_cost options)" flag off

option "joint_SPL" - "use Joint model (y+z) using Self-Paced Learning for modified Latent SSVM (with --y_data, --z_data, --y_cost, and --z_cost options)" flag off

option "y_data" - "set file name for y_train_data (y is visible and z is hidden)" string no

option "z_data" - "set file name for z_train_data (y is hidden and z is visible)" string no

option "y_cost" - "set cost of y_train_data in joint model" float default="1" no

option "z_cost" - "set cost of z_train_data in joint model" float default="1" no

option "y_train_num" - "set number of sentences of y_train_data for training (for experiments) (disabled if use 0)" int default="0" no

option "z_train_num" - "set number of sentences of y_train_data for training (for experiments) (disabled if use 0)" int default="0" no

option "init_iter" - "initial iterations for Joint SSVM training algorithm" int default="10" no


# section "Predict options"

option "output" o "prediction output filename" string no

option "nbest" - "print N-best result" int default="1" no

option "beam" - "set number of beam in search (disabled if use 0)" int default="0" no


# section "Convert option"

option "threshold" t "set threshold (convert mode)" float default="1e-04" no


defgroup "MODE"

groupoption "predict" p "prediction mode, default is training mode" group="MODE"

groupoption "show" - "show-feature mode" group="MODE"

groupoption "convert" - "convert mode ('txt model to bin model' or 'bin model to txt model (with -b)') and remove zero features (with --threshold option)" group="MODE"

groupoption "convert2" - "convert2 mode (all_feaure model to support_feature model) and remove zero features (with --threshold option)" group="MODE"

groupoption "convert3" - "convert3 mode (support_feature model to all_feature model) and remove zero features (with --threshold option)" group="MODE"

groupoption "modify" - "modify mode (modify feature weight), the option file is a list of feature weight" string group="MODE"

groupoption "domain" - "domain adaptation (Prior model) for structural SVM (with fsmo/fsmo_joint/pegasos algorithms and source option)" group="MODE"


defgroup "Parameter Estimate Method for structural SVM"

groupoption "fsmo" - "use Fixed-threshold SMO for structural SVM (shared slack)" group="Parameter Estimate Method for structural SVM"

groupoption "fsmo_joint" - "use FSMO + joint constraint (1-slack) using Gram matrix" group="Parameter Estimate Method for structural SVM"

groupoption "fsmo_joint2" - "use FSMO + joint constraint (1-slack) without Gram matrix (slow version)" group="Parameter Estimate Method for structural SVM"

groupoption "pegasos" - "use Pegasos in primal optimization (random shuffled train_data) (default method)" group="Parameter Estimate Method for structural SVM"
