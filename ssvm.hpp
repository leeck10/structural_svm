/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
 	@file ssvm.hpp
	@brief (linear chain) Structural SVMs
    @author Changki Lee (leeck@kangwon.ac.kr)
	@date 2013/3/1
*/
#ifndef SSVM_H
#define SSVM_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>

// for M, alpha, beta matrix
#define MAT2(I,X)    ((n_outcome * (I)) + X)

#define MAX(X,Y)    ((X)>(Y)?(X):(Y))
#define MIN(X,Y)    ((X)<(Y)?(X):(Y))
#define ABS(X)      ((X)>0?(X):(-(X)))
#define SQUARE(X)      ((X)*(X))

using namespace std;
//using namespace __gnu_cxx;

/// feature type
typedef struct feature_struct {
    int pid;
#ifndef BINARY_FEATURE
    // use general feature if not defined BINARY_FEATURE
    float fval;
#endif
} feature_t;

/// context type
typedef vector<feature_t> context_t;

/// node type
typedef struct node_struct {
    int outcome;
    int start;
    int end;
    context_t context;
} node_t;

/// sentence type
typedef vector<node_t> sent_t;


/// for FSMO
typedef struct vect_struct {
    vector<pair<int, float> > vect; ///< feature/value by increasing feature num. (sparse)
    double  twonorm_sq;     ///< squared euclidian length of the vector.
    double  factor;         ///< factor is multiplied in the sum. */
} single_vect_t;

/// for linear constraints which are a sum of multiple feature vectors
typedef vector<single_vect_t> vect_t;


/**
 @brief Linear Chain Structural Support Vector Machine.
 @class SSVM
 */
class SSVM {
    public:
        SSVM();
        virtual ~SSVM();

        /// load model
        virtual void load(const string model);
        /// load binary model
        virtual void load_bin(const string model);

        /// save model
        virtual void save(const string model);
        /// save binary model
        virtual void save_bin(const string model);
    
        /// set n_pred (in hash_feature mode)
        void set_n_pred(int pred_num) {
            if (hash_feature) {
                n_pred = pred_num;
            } else {
                cerr << "Error: is not hash_feature mode!" << endl;
                exit(1);
            }
        }

        /// show feature weight
        void show_feature();

        /// get feature weight
        double get_feature_weight(int pid, int oid) {
            if (pid >= 0 && pid < n_pred && oid >= 0 && oid < n_outcome) {
                int fid = make_fid(pid, oid);
                if (fid >= 0) return theta[fid];
            }
            return 0;
        }
        double get_feature_weight(int pid) {
            if (support_feature) {
                double obj = 0;
                vector<pair<int, int> >& param = params[pid];
                for (size_t j = 0; j < param.size(); ++j) {
                    int fid = param[j].second;
                    obj += SQUARE(theta[fid]);
                }
                return sqrt(obj);
            } else {
                double obj = 0;
                for (size_t oid = 0; oid < n_outcome; oid++) {
                    int fid = make_fid(pid, oid);
                    obj += SQUARE(theta[fid]);
                }
                return sqrt(obj);
            }
        }

        /// set feature weight
        int set_feature_weight(string feature, string label, float weight) {
            if (pred_map.find(feature) != pred_map.end() && outcome_map.find(label) != outcome_map.end()) {
                int pid = pred_map[feature];
                int oid = outcome_map[label];
                int fid = make_fid(pid, oid);
                if (fid >= 0) {
                    theta[fid] = weight;
                    return 1;
                }
            } 
            return 0;
        }
        int set_feature_weight(int pid, int oid, float weight) {
            if (pid >= 0 && pid < n_pred && oid >= 0 && oid < n_outcome) {
                int fid = make_fid(pid, oid);
                if (fid >= 0) {
                    theta[fid] = weight;
                    return 1;
                }
            }
            return 0;
        }

        /// remove zero feature
        void remove_zero_feature(double threshold);
        /// convert all feature to support feature
        void to_support_feature();
        /// convert support feature to all feature
        void to_all_feature();

        /// load event and make param
        virtual void load_event(const string file);
        /// load test event
        virtual void load_test_event(const string file);
        /// for Joint SSVM
        virtual void load_latent_event(const string file, bool is_y_train_data) {}

        /// random_shuffle train_data
        void random_shuffle_train_data() {
            random_shuffle(train_data.begin(), train_data.end());
        }

        /// predict
        virtual int predict(ostream& f);
        /// predict N-best
        virtual int predict_nbest(ostream& f, int nbest);
        /// predict One Word Per Sentence
        virtual int predict_owps(ostream& f);

        /// make sentence type
        virtual void make_sent(const vector<vector<string> >& cont_seq, sent_t &sent);
        virtual void make_sent(const vector<vector<int> >& cont_seq, sent_t &sent);
        virtual void make_sent(const vector<vector<pair<int,float> > >& cont_seq, sent_t &sent);

        /// for prediction
        virtual double eval(sent_t& sent, vector<string>& label);
        /// eval with constraint
        double eval_with_constraint(sent_t& sent, const vector<string>& constraint, vector<string>& label);
        /// for korean spacing
        double eval_with_loss(sent_t& sent, double weight, vector<string>& input_label, vector<string>& label);
        /// for n-best prediction
        vector<double> eval_nbest(sent_t& sent, vector<vector<string> >& label, int n=5);
        /// for One Word Per Sentence
        double eval_owps(sent_t& sent, string& label);
        vector<double> eval_owps_all(sent_t& sent);
        /// for NHN parser: cont is the vector of pid
        vector<double> eval_owps_all(const vector<int>& cont);

        /// for previous version CRF tool
        double eval(const vector<vector<string> >& cont_seq, vector<string>& label);
        vector<double> eval_nbest(const vector<vector<string> >& cont_seq, vector<vector<string> >& label, int n=5);
        double eval_owps(const vector<string>& cont, string& label);
        vector<double> eval_owps_all(const vector<string>& cont);

        /// training
        void train(string estimate);

        /// clear
        void clear();

        /// init theta
        virtual void init_theta() {
            if (n_theta == 0) {
                n_theta = n_pred * n_outcome;
                cerr << endl << "n_theta set by init_theta(): " << n_theta << endl;
            }
            if (theta == NULL) {
                theta = new float[n_theta];
                cerr << "theta allocated by init_theta(): " << n_theta << endl;
            }
            for (int i=0; i < n_theta; i++) theta[i] = 0;
        }

        /// print start status
        void print_start_status(string estimate);

        /// M matrix (m_vec) (log scale) : 처음 초기화 시에 한번만 불러주면 됨
        void make_M_matrix();
    
        /// get hash code
        unsigned int hash(const string key);

        /// get predicate string
        string get_pred_str(int pid) {
            return pred_vec[pid];
        }

        /// transition-based parser를 위해 추가
        virtual int make_fid(int pid, int oid);
        int make_oid(string label) {
            if (outcome_map.find(label) != outcome_map.end()) {
                return outcome_map[label];
            }
            return -1;
        }
        virtual int make_pid(string feature) {
            if (hash_feature) {
                return hash(feature) % n_pred;
            }
            if (pred_map.find(feature) != pred_map.end()) {
                return pred_map[feature];
            }
            return -1;
        }

        int is_train;   ///< for make_pid4train
        /// hash_feature일때는 make_pid와 똑 같음
        virtual int make_pid4train(string feature) {
            if (hash_feature) {
                return hash(feature) % n_pred;
            }
            if (pred_map.find(feature) != pred_map.end()) {
                return pred_map[feature];
            } else if (is_train) { // for training data
                int pid;
                #pragma omp critical (outcome)
                {
                pid = pred_vec.size();
                pred_map[feature] = pid;
                pred_vec.push_back(feature);
                }
                return pid;
            }
            return -1;
        }
        vector<string> get_outcome_vec() {
            return outcome_vec;
        }
        void add_outcome(string label) {
            if (outcome_map.find(label) == outcome_map.end()) {
                int oid = outcome_vec.size();
                outcome_map[label] = oid;
                outcome_vec.push_back(label);
                n_outcome = outcome_vec.size();
                cerr << label << " ";
            }
        }
        float get_theta(int fid) {
            return theta[fid];
        }
        void set_theta(int fid, float w) {
            theta[fid] = w;
        }
        void update_theta(int fid, float d) {
            theta[fid] += d;
        }
        int get_n_theta() {
            return n_theta;
        }

        // util
        void tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ");
		void split(const string& str, vector<string>& tokens, const string& delimiter = " ");

        // number of train set exampel, test set example
        int n_event;
        int n_test_event;

        // parameter
        string model_file;
        int use_comment;
        int owps_format;
        int hash_feature;       ///< for hash ssvm
        int support_feature;    ///< for large number of label (class)
        int general_feature;    ///< for non-binary feature
        int incremental;
        int beam;
        int verbose;
        int binary;
        int skip_eval;
        int train_num;
        double threshold;

        // Pegasos
        int iter;
        int period;

        // SVM
        double cost;
        int rm_inactive;
        int buf;
        double eps;
        // final optimality check for shrinking
        int final_opt_check;

        // domain adaptation
        int domain_adaptation;

        // Joint SSVM;
        double y_cost;
        double z_cost;
        int init_iter;

    protected:	// 상속을 위해 private -> protected
        int default_oid;
        string edge;

        int n_pred;                   ///< predicate number == hash bucket number
        int n_theta;				  ///< number of feature weights
        int n_outcome;                ///< number of outcome

        // vector (vector (outcome, fid)) : support_feature일 경우 사용
        // params for builtin model
        vector<vector<pair<int, int> > > params;
        // all_feature일 경우 params를 안쓰는 대신에 다음과 같이 사용
        // fid = pid * n_outcome + oid
        // pid * n_outcome <= fid < (pid+1) * n_outcome

        float *theta;  ///< feature weight

        map<string, int> pred_map;
        vector<string> pred_vec;

        map<string, int> outcome_map;
        vector<string> outcome_vec;

        vector<int> edge_pid; ///< edge feature의 pid

        vector<sent_t> train_data;
        vector<sent_t> test_data;
        vector<string> train_data_comment;
        vector<string> test_data_comment;

        // matrix
        vector<double> m_vec;

		double acc;

        // for SVM
        vector<double> alpha;       ///< alpha : size = const num
        vector<int> alpha_history;  ///< alpha_history : size = const num
        vector<vect_t> work_set;    ///< working set : size = const num
        vector<double> loss;        ///< loss : size = const num
        vector<double> x_norm_vec;  ///< x_norm_vec : size = const num
        vector<int> sent_ids;       ///< sent_ids : size = const num : work_set id를 train_data id로 바꾼다
        vector<vector<int> > work_set_ids;  ///< work_set_ids : size = train_data.size() : train_data id->work_set id list
        vector<vector<int> > y_seq_vec;     ///< y_seq_vec : size = const num : 나중에 error detection에 사용
        vector<int> opti;           ///< shirink : size = train_data.size()
        vector<double> sum_alpha;   ///< sum_alpha : size = train_data.size()

        vector<double> slacks;      ///< slacks : size = train_data.size()
        vector<int> slacks_id;      ///< slacks_id : size = train_data.size() : 실제 slack인 work_set id를 가리킨다

        vector<double> cost_diff_vec;   ///< cost_diff_vec : size = const num

        vector<vector<float> > gram;    ///< GRAM for 1-slack
        int gram_size;

        // for SVM
        double precision;

        // function
        /// add edge feature
        virtual void add_edge();
        /// make edge predicate id
        virtual void make_edge_pid();

        // M, R matrix (log scale) (make_M_matrix 함수는 public)
        virtual void make_R_matrix(vector<double>& r_vec, sent_t& sent);
        void constrain_R_matrix(vector<double>& r_vec, const vector<string>& constraint);

        virtual void make_M_matrix4owps(vector<double>& r_vec, sent_t& sent);


        // viterbi
        virtual vector<int> viterbi(vector<double>& r_vec, sent_t& sent, double& prob);
        vector<vector<int> > viterbi_nbest(vector<double>& r_vec, sent_t& sent, vector<double>& prob, int n=5);
        virtual vector<int> viterbi4owps(vector<double>& r_vec, sent_t& sent, double& prob);

        // training - each machine learnign algorithm
        double train_fsmo();    ///< traing using FSMO
        double train_fsmo_joint(bool use_gram); ///< traing using 1-slack FSMO
        double train_pegasos();     ///< traing using Pegasos algorithm
        virtual double train_latent_ssvm(int use_SPL=0) {return 0;}   ///< for latent SSVM
        virtual double train_joint_ssvm(int use_SPL=0) {return 0;}    ///< for joint SSVM

        /// print status at each iteration (for overriding)
        virtual void print_status() {}

        // for fsmo
        /// 정답 및 y_seq에 해당하는 vector를 구해서 (정답벡터 - y_seq벡터)를 생성
        virtual vect_t make_diff_vector(sent_t& sent, vector<int>& y_seq);
        /// fsmo_joint에 사용됨
        virtual void append_diff_vector(vector<float>& dense_vect, vect_t& vect);

        /// for training
        virtual vector<int> find_most_violated_constraint(vector<double>& r_vec, sent_t& sent, double wscale=1);

        /// loss (for overriding)
        virtual double calculate_loss(sent_t& sent, vector<int>& y_seq);

        /// calculate cost
        double calculate_cost(vect_t& vect);
        double calculate_cost(int vect);

        /// kernel
        double kernel4gram(int vect1, int vect2);
        double kernel(vect_t& vect1, vect_t& vect2);

        // optimize
        /// using FSMO
        void optimize_dual4fsmo(double cost, double eps);
        /// 1-slack formulation
        void optimize_dual4fsmo_joint(double cost, double eps, int use_gram);

        /// dot product
        double dot_product(single_vect_t& svect1, single_vect_t& svect2);

        /// update weight
        void update_weight(vect_t& vect, double d);

        /// for SVM slack
        void save_slack(double eps);
        /// length of longest vector
        double longest_vector();
};
#endif // SSVM_H

