/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
 	@file joint_ssvm.hpp
	@brief Joint model using Latent Structural SVMs
        support sequence labeling
        assume F(x,y,z) = F(x) * Y * Z
    @author Changki Lee (leeck@kangwon.ac.kr)
	@date 2013/7/16
*/

#ifndef JOINT_SSVM_H
#define JOINT_SSVM_H

#include "ssvm.hpp"

using namespace std;

/**
 @brief Joint_SSVM is a child-class of SSVM class.
 support sequence labeling.
 assume F(x,y,z) = F(x) * Y * Z, the size of train set for y > train set for y+z
 (ex. y = korean spacing tag, z = korean POS tag)
 we can view the Joint_SSVM is a latent SSVM that has a little train set for z
 @class Joint_SSVM
 */
class Joint_SSVM : public SSVM {
    public:
        Joint_SSVM();

        /// first, load latent train data for y or z
        void load_latent_event(const string file, bool is_y_train_data);
        /// second, load joint train data: y+z
        void load_event(const string file);
        
        /// Joint learning using Pegasos. use_SPL: Self-Paced learning
        double train_joint_ssvm(int use_SPL=0);

	protected:
        // n_outcome: for y+h
        int n_y_outcome;    ///< for y
        int n_z_outcome;    ///< for z

        // for y
        map<string, int> y_outcome_map; ///< y outcome map
        vector<string> y_outcome_vec;
        // for z
        map<string, int> z_outcome_map; ///< z outcome map
        vector<string> z_outcome_vec;

        map<int, int> joint2y_map;  ///< (y,z) to y mapping
        map<int, int> joint2z_map;  // (y,z) to z mapping

        // we assume train_data.size() < y_train_data.size()
        // train_data: train set for y+z
        vector<sent_t> y_train_data;    ///< train set for y (z is hidden)
        vector<sent_t> z_train_data;    ///< train set for z (y is hidden)


        /**
         @brief find most violated constraint for latent SSVM.
            argmax_{y,z} {w*F(x_i,y,z} + L(y_i,y)} --> return {y,z} or
            argmax_{y,z} {w*F(x_i,y,z} + L(z_i,z)} --> return {y,z}
         @param sent is an example of y_train_data or z_train_data
         */
        vector<int> find_most_violated_constraint(vector<double>& r_vec, sent_t& sent, bool is_y_train_data, double wscale=1);

        /// find hidden variable.
        /// argmax_{z} {w*f(x_i,y_i,z)} --> return {y_i,z*}
        /// argmax_{y} {w*f(x_i,y,z_i)} --> return {y*,z_i}
        vector<int> find_hidden_variable(vector<double>& r_vec, sent_t& sent, bool is_y_train_data);

        /// calculate loss for latent SSVM.
        /// L(y_i,y) or L(z_i,z)) instead of L(y_i,z_i,y,z)
        double calculate_latent_loss(sent_t& sent, vector<int>& y_seq, bool is_y_train_data);

        /// make_diff_vector: return {f(x_i,y_i,z_i*) - f(x_i,y,z)} or
        ///                   return {f(x_i,y_i*,z_i) - f(x_i,y,z)}
        /// z_star : {y_i,z_i*} or {y_i*,z_i} 
        /// y_seq : {y,z}
        vect_t make_diff_vector(sent_t& sent, vector<int>& z_star, vector<int>& y_seq);
};

#endif // JOINT_SSVM_H
