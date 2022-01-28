/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
 	@file latent_ssvm.hpp
	@brief CCCP-based Latent Structural SVMs
        currently does not support sequence labeling
    @author Changki Lee (leeck@kangwon.ac.kr)
	@date 2013/3/4
*/

#ifndef LATENT_SSVM_H
#define LATENT_SSVM_H

#include "ssvm.hpp"

using namespace std;

/**
 @brief Latent_SSVM is a child-class of SSVM class.
    Currently does not support sequence labeling.
    (h is the position of node_t in sent_t)
 @class Latent_SSVM
 */
class Latent_SSVM : public SSVM {
    public:
        Latent_SSVM();
        
        /// CCCP + Pegasos.
        /// use_SPL : Self-Paced learning
        double train_latent_ssvm(int use_SPL=0);

        /// for prediction
        int predict(ostream& f);
        /// for N-best prediction
        int predict_nbest(ostream& f, int nbest);

        /// for prediction.
        /// h is the position of node_t in sent_t
        double eval(sent_t& sent, int& y, int& h);

	protected:
        vect_t make_diff_vector(sent_t& sent, int yi, int hi, int y, int h);
};

#endif // LATENT_SSVM_H
