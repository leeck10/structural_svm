# structural_svm

ssvm_tool 2013.7

A command line utility to train/test a Structural SVM (SSVM) model.

## Usage: ssvm_tool [OPTIONS]... [FILES]...

    -h, --help             Print help and exit
    -V, --version          Print version and exit

### Training options:
    -c, --cost=FLOAT       set cost of SSVMs  (default=`1')
    -m, --model=STRING     set model file name
    -b, --binary           save/load model in binary format  (default=off)
        --source=STRING    source model file for domain adaptation
        --skip_eval        skip test set evaluation in the middle of training (default=off)
        --owps_format      use One Word Per Sentence (OWPS) format (default=off)
        --hash=INT         use hash feature and set number of predicates (default=`0')
    -s, --support          use support feature (default is all feature) (default=off)
        --general          general feature mode (feature:value format) (default is binary feature)  (default=off)
    -r, --random=INT       use random_shuffle in train_data (disabled if use 0) (default=`0')
        --train_num=INT    set number of sentence in train_data for training (for experiments) (disabled if use 0)  (default=`0')
    -v, --verbose          verbose mode  (default=off)

### Structural SVM options:
    -e, --epsilon=FLOAT    set epsilon (fsmo, fsmo_joint)  (default=`0.01')
        --buf=INT          set the number of new constraints to accumulated before recomputing the QP (fsmo, pegasos) (default=`100')
        --rm_inactive=INT  inactive constraints are removed (iteration) (fsmo, fsmo_joint)  (default=`50')
        --final_opt        do final optimal check in shrinking (fsmo, fsmo_joint) (default=off)
        --comment          use comment info in save_slack() (fsmo) (default=off)

### Pegasos options:
    -i, --iter=INT         iterations for training algorithm (pegasos) (default=`100')
        --period=INT       save model periodically (pegasos)  (default=`0')

### Latent Strucutral SVM options:
    --latent           use CCCP-based Latent SSVM (doesn't support sequence labeling)  (default=off)
    --latent_SPL       use Self-Paced Learning for Latent SSVM (doesn't support sequence labeling)  (default=off)

### Joint Strucutral SVM options:
    --joint            use Joint model (y+z) using modified Latent SSVM (with --y_data, --z_data, --y_cost, and --z_cost options)  (default=off)
    --joint_SPL        use Joint model (y+z) using Self-Paced Learning for modified Latent SSVM (with --y_data, --z_data, --y_cost, and --z_cost options)  (default=off)
    --y_data=STRING    set file name for y_train_data (y is visible and z is hidden)
    --z_data=STRING    set file name for z_train_data (y is hidden and z is visible)
    --y_cost=FLOAT     set cost of y_train_data in joint model  (default=`1')
    --z_cost=FLOAT     set cost of z_train_data in joint model  (default=`1')
    --y_train_num=INT  set number of sentences of y_train_data for training (for experiments) (disabled if use 0)  (default=`0')
    --z_train_num=INT  set number of sentences of y_train_data for training (for experiments) (disabled if use 0)  (default=`0')
    --init_iter=INT    initial iterations for Joint SSVM training algorithm (default=`10')

### Predict options:
    -o, --output=STRING    prediction output filename
        --nbest=INT        print N-best result  (default=`1')
        --beam=INT         set number of beam in search (disabled if use 0) (default=`0')

### Convert option:
    -t, --threshold=FLOAT  set threshold (convert mode)  (default=`1e-04')

Group: MODE
    -p, --predict          prediction mode, default is training mode
        --show             show-feature mode
        --convert          convert mode ('txt model to bin model' or 'bin model to txt model (with -b)') and remove zero features (with --threshold option)
        --convert2         convert2 mode (all_feaure model to support_feature model) and remove zero features (with --threshold option)
        --convert3         convert3 mode (support_feature model to all_feature model) and remove zero features (with --threshold option)
        --modify=STRING    modify mode (modify feature weight), the option file is a list of feature weight
        --domain           domain adaptation (Prior model) for structural SVM (with fsmo/fsmo_joint/pegasos algorithms and source option)

 Group: Parameter Estimate Method for structural SVM
    --fsmo             use Fixed-threshold SMO for structural SVM (shared slack)
    --fsmo_joint       use FSMO + joint constraint (1-slack) using Gram matrix
    --fsmo_joint2      use FSMO + joint constraint (1-slack) without Gram matrix (slow version)
    --pegasos          use Pegasos in primal optimization (random shuffled train_data) (default method)
