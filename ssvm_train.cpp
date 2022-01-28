/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
    @file ssvm_train.cpp
	@brief (linear chain) Structural SVMs
    @author Changki Lee (leeck@kangwon.ac.kr)
	@date 2013/3/1
*/
#ifdef WIN32
#pragma warning(disable: 4786)
#pragma warning(disable: 4996)
#pragma warning(disable: 4267)
#pragma warning(disable: 4244)
#pragma warning(disable: 4018)
#endif

#include <cassert>
#include <stdexcept> //for std::runtime_error
#include <memory>    //for std::bad_alloc
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "ssvm.hpp"
#include "timer.hpp"
#include "pqueue.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;


/// train
void SSVM::train(string estimate) {
    if (domain_adaptation && support_feature) {
        cerr << "Error: domain adaptation does not support support_feature mode!" << endl;
        exit(1);
    }
    
    add_edge();
    print_start_status(estimate);

    if (estimate == "fsmo") {
        train_fsmo();
    } else if (estimate == "fsmo_joint") {
        train_fsmo_joint(true);
    } else if (estimate == "fsmo_joint2") {
        train_fsmo_joint(false);
    } else if (estimate == "pegasos") {
        train_pegasos();
    } else if (estimate == "latent_SSVM") {
        train_latent_ssvm();
    } else if (estimate == "latent_SPL") {
        train_latent_ssvm(1);
    } else if (estimate == "joint_SSVM") {
        train_joint_ssvm();
    } else if (estimate == "joint_SPL") {
        train_joint_ssvm(1);
    } else {
        cerr << "!Error: " << estimate << " is not supported." << endl;
        exit(1);
    }
}


/// fixed-threshold SMO (Sequential Minimum Optimization) + openMP
double SSVM::train_fsmo() {
    int n = train_data.size();
    int niter = 1, const_num = work_set.size(), old_const_num = work_set.size();
    int correct = 0, error = 0, argmax_count = 0;
    int i, j;
    double max_kkt = 0, obj = 0, diff_obj = 0, additional_value = 0;
    double time = 0, time_qp = 0, time_viol = 0, time_psi = 0;

    // for domain adaptation
    float *prior_theta = NULL;
    if (domain_adaptation) {
        cerr << "[Domain adaptation mode]" << endl;
        prior_theta = new float[n_theta];
        for (int i=0; i < n_theta; i++) {
            prior_theta[i] = theta[i];
        }
    }

#ifdef _OPENMP
    if (buf < omp_get_max_threads()) omp_set_num_threads(buf);
	cerr << "[OpenMP] Number of threads: " << omp_get_max_threads() << endl;
#endif

    // comment
    if (!use_comment) vector<string>().swap(train_data_comment);

    // shirink 초기화
    int opti_round = 0;
    opti.resize(n, 0);

    // work_set_ids 초기화
    work_set_ids.clear();
    for (i=0; i < train_data.size(); i++) {
        vector<int> temp;
        work_set_ids.push_back(temp);
    }

    // sum_alpha 초기화
    // shared_slack인 경우만 사용: size=train_data.size()
    sum_alpha.resize(train_data.size(), 0);

    // slack 초기화
    // shared_slack인 경우만 사용: size=train_data.size()
    slacks.resize(train_data.size(), 0);
    slacks_id.resize(train_data.size(), -1);

    // inactive constraint 제거
    int opt_count = 0;

    if (domain_adaptation) {
        printf("iter   acc   train time  |w| |w-w0|   primal      dual active const   SV\n");
        printf("========================================================================");
    } else {
        printf("iter   acc   train time   |w|    primal      dual active const   SV\n");
        printf("===================================================================");
    }
    fflush(stdout);

    // C/n
    double cost1 = cost / n;
    double eps1;

    if (owps_format) {
        eps1 = 1.28;
    } else {
        eps1 = 32;
    }
    int active_num = 0, old_active_num = 0;
    int sv_num = 0, old_sv_num = 0;
    int newconstraints = 0, new_precision = 1;
    int infinite_loop = 0;

    do { // increase precision
    eps1 = MAX(eps1*0.49999999, eps);
    new_precision = 1;

    cerr << endl << "# eps = " << eps1 << " ";

    opti_round++;
    active_num = n;
    old_active_num = active_num;

	// make M_i matrix
	make_M_matrix();

    do { // new constatrains
        timer t;

        old_const_num = const_num;
        old_active_num = active_num;
        correct = 0;
        error = 0;
        max_kkt = 0.0;

		for (size_t sent_i = 0; sent_i < train_data.size();) {
			#pragma omp parallel for
			for (int buf_i=0; buf_i < buf; buf_i++) {
				int sent_index, skip = 0;
				#pragma omp critical (sent_i)
				{
				sent_index = sent_i;
				if (sent_i++ >= train_data.size()) skip = 1;
				} // omp
				if (skip) continue;

				// shrink : 마지막에서는 shrink를 하도록 or 안하도록 수정
				if (opti[sent_index] != opti_round || (final_opt_check && eps1 == eps)) {
					// find most violated contraint
					sent_t& sent = train_data[sent_index];
					vector<int> y_seq;
					timer t_viol;
					#pragma omp atomic
					argmax_count++;
					// make M_i matrix
					vector<double> r_vec;
					make_R_matrix(r_vec, sent);
					y_seq = find_most_violated_constraint(r_vec, sent);
					#pragma omp atomic
					time_viol += t_viol.elapsed();

					double cur_loss = calculate_loss(sent, y_seq);

					timer t_psi;
					vect_t max_vect = make_diff_vector(sent, y_seq);
					#pragma omp atomic
					time_psi += t_psi.elapsed();

					if (max_vect.empty()) {
						if (opti[sent_index] != opti_round) {
							active_num--;
							opti[sent_index] = opti_round;
						}
						continue;
					}

					double cost_diff = calculate_cost(max_vect);
					double H_y = cur_loss - cost_diff;

					#pragma omp critical (slack)
					{
					// slack
					double slack = 0;
					int sid = -1;
					for (i=0; i < sent_ids.size(); i++) {
						int id = sent_ids[i];
						if (id == sent_index) {
							double cur_cost_diff = calculate_cost(work_set[i]);
							double cur_H_y = loss[i] - cur_cost_diff;
							if (cur_H_y > slack) {
								slack = cur_H_y;
								sid = i;
							}
						}
					}
					slacks[sent_index] = slack;
					slacks_id[sent_index] = sid;

					max_kkt = MAX(max_kkt, H_y - slack);

					if (H_y > slack + eps1) {
						//cerr << ".";
						// alpha
						alpha.push_back(0);
						// alpha_history
						alpha_history.push_back(opt_count);
						// work set
						work_set.push_back(max_vect);
						// loss
						loss.push_back(cur_loss);
						// x_norm_vec
						x_norm_vec.push_back(kernel(max_vect, max_vect));
						// sent_ids
						sent_ids.push_back(sent_index);
						// work_set_ids
						work_set_ids[sent_index].push_back(const_num);
						// y_seq
						y_seq_vec.push_back(y_seq);

						const_num++;
						newconstraints++;
					} else {
						if (opti[sent_index] != opti_round) {
							active_num--;
							opti[sent_index] = opti_round;
						}
					}
					} // omp
				}
			} // buf loop

			// get new QP solution
			if ((newconstraints >= buf)
				|| (newconstraints > 0 && sent_i >= n-1)
				|| (const_num > 0 && new_precision)) {
				cerr << "*";
				timer t_qp;
				// using SMO if bounded
				optimize_dual4fsmo(cost1, eps1);
				time_qp += t_qp.elapsed();

				// make M_i matrix
				make_M_matrix();

				new_precision = 0;
				newconstraints = 0;

				// inactive constraint 제거
				// rm_inactive iteration 동안 active된 적이 없는 것
				// svm-light는 50
				opt_count++;
				if (1) {
					int remove_count = 0;
					for (i=0; i < work_set.size() - remove_count; i++) {
						// active constraint
						if (alpha[i] > 0) {
							alpha_history[i] = opt_count;
						}
						// rm_inactive번 안에 active 된 적이 없는 constraint
						else if (opt_count - alpha_history[i] >= rm_inactive) {
							// 맨 뒤의 원소부터 swap한 후, remove_count만큼 맨뒤 원소들을 삭제
							int sw_i = work_set.size() - 1 - remove_count;
							work_set[i] = work_set[sw_i];
							alpha[i] = alpha[sw_i];
							alpha_history[i] = alpha_history[sw_i];
							loss[i] = loss[sw_i];
							x_norm_vec[i] = x_norm_vec[sw_i];
							sent_ids[i] = sent_ids[sw_i];
							y_seq_vec[i] = y_seq_vec[sw_i];
							// 나머지 변수 처리
							i--;
							//old_const_num--;
							const_num--;
							remove_count++;
						}
					}
					if (remove_count > 0) {
						// 실제 제거
						int rm_i = work_set.size() - remove_count;
						work_set.erase(work_set.begin() + rm_i, work_set.end());
						alpha.erase(alpha.begin() + rm_i, alpha.end());
						alpha_history.erase(alpha_history.begin() + rm_i, alpha_history.end());
						loss.erase(loss.begin() + rm_i, loss.end());
						x_norm_vec.erase(x_norm_vec.begin() + rm_i, x_norm_vec.end());
						sent_ids.erase(sent_ids.begin() + rm_i, sent_ids.end());
						y_seq_vec.erase(y_seq_vec.begin() + rm_i, y_seq_vec.end());
						// work_set_ids 다시 작성
						work_set_ids.clear();
						for (i=0; i < train_data.size(); i++) {
							vector<int> temp;
							work_set_ids.push_back(temp);
						}
						for (i=0; i < sent_ids.size(); i++) {
							int id = sent_ids[i];
							work_set_ids[id].push_back(i);
						}
						// test
						cerr << "r" << remove_count;
					}
				}
			} // QP
		} // example loop

        // time
        double iter_time = t.elapsed();
        time += iter_time;

        old_sv_num = sv_num;
        sv_num = 0;
        double sum = 0, alphasum = 0;
        for (i=0; i < alpha.size(); i++) {
            if (alpha[i] != 0) {
                sum += alpha[i];
                alphasum += alpha[i] * loss[i];
                sv_num++;
            }
        }

        // 무한 루프 체크
        if (active_num == old_active_num && sv_num == old_sv_num && const_num == old_const_num) {
            infinite_loop++;
            if (infinite_loop >= 2) {
                cerr << endl << endl << "Warning: infinite loop (" << infinite_loop << "): ";
                cerr << "change rm_inactive or buf!" << endl;
                cerr << " changed active_num=0" << endl;
                infinite_loop = 0;
                active_num = 0;
            }
        } else {
            infinite_loop = 0;
        }

        // obj --> model length |w|: non-linear인 경우 domain adaption은 아직 고려 안됨
        obj = 0.0;
        diff_obj = 0;
        additional_value = 0;
        if (!skip_eval || active_num == 0) {
            for (i=0; i < n_theta; i++) {
                obj += SQUARE(theta[i]);
                if (domain_adaptation && theta[i] != prior_theta[i]) {
                    diff_obj += SQUARE(theta[i] - prior_theta[i]);
                    additional_value += prior_theta[i] * (prior_theta[i] - theta[i]);
                }
            }
            obj = sqrt(obj);
            diff_obj = sqrt(diff_obj);
        }

        // continue evaluations
        //double acc = 100.0*(correct)/(correct+error);

        // test_data 성능
        correct = 0;
        if (!skip_eval || active_num == 0) {
            vector<sent_t>::iterator it = test_data.begin();
            // make M_i matrix
            make_M_matrix();
			#pragma omp parallel for private(j)
            for (i = 0; i < test_data.size(); i++) {
                sent_t& sent = test_data[i];
                double prob;
                vector<int> y_seq;
                // make M_i matrix
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);
                y_seq = viterbi(r_vec, sent, prob);
            
                for (j=0; j < sent.size(); j++) {
					if (sent[j].outcome == y_seq[j]) {
						#pragma omp atomic
						correct++;
					}
                }
            }
        }
        double test_acc = test_data.size() > 0 ? 100*double(correct)/double(n_test_event) : 0;

        // primal cost 계산: 0.5 * |w|^2 + C/n * L
        double primal_cost = 0, dual_cost = 0;
        double slack_sum = 0;
        for (i=0; i < slacks.size(); i++) {
            if (slacks[i] > 0) slack_sum += slacks[i] + eps1;
            else slack_sum += eps1;
        }
        primal_cost += cost1 * slack_sum;
        if (obj != 0 || diff_obj != 0) {
            if (domain_adaptation) primal_cost = 0.5 * diff_obj * diff_obj + cost1 * slack_sum;
            else primal_cost = 0.5 * obj * obj + cost1 * slack_sum;
            if (domain_adaptation) dual_cost = alphasum - (0.5 * diff_obj * diff_obj) + additional_value;
            else dual_cost = alphasum - (0.5 * obj * obj);
        }

        if (domain_adaptation) {
            printf("\n%3d %5.2f%% %6.1f %5.0f %5.1f %4.1f %9.2e %9.2e %5d %5d %5d ", 
                    niter++ , test_acc, iter_time, time, obj, diff_obj, primal_cost, dual_cost, active_num, const_num, sv_num);
        } else {
            printf("\n%3d %5.2f%% %6.1f %5.0f %5.1f %9.2e %9.2e %5d %5d %5d ", 
                    niter++ , test_acc, iter_time, time, obj, primal_cost, dual_cost, active_num, const_num, sv_num);
        }
        if (!skip_eval || active_num == 0) print_status();
        fflush(stdout);
    } while (active_num > 0);
        // eps 마다 저장: 0.5, 0.25, 0.1 -- by leeck
        if (period > 0 && eps1 > eps && eps1 < 1)
        {
            cerr << endl << "model saving to " << model_file << "." << eps1 << " ... ";
            char temp[100];
            sprintf(temp, "%s.%g", model_file.c_str(), eps1);
			if (binary) save_bin(string(temp));
			else save(string(temp));
            cerr << "done." << endl;
        }
    } while (eps1 > eps);

    try {
        cerr << endl;
        int const_num = 0, sv_num = 0;
        double sum = 0, alphasum = 0;
        double max_alpha = 0.0;
        const_num = alpha.size();
        for (i=0; i < alpha.size(); i++) {
            if (alpha[i] > 0) {
                sum += alpha[i];
                alphasum += alpha[i] * loss[i];
                sv_num++;
                max_alpha = MAX(max_alpha, alpha[i]);
            }
        }

        // slack
        int bounded_sv = 0;
        for (i=0; i < work_set_ids.size(); i++) {
            double slack = 0;
            int sid = -1;
            for (j=0; j < work_set_ids[i].size(); j++) {
                int id = work_set_ids[i][j];
                double cur_cost_diff = calculate_cost(work_set[id]);
                double cur_H_y = loss[id] - cur_cost_diff;
                if (cur_H_y > slack) {
                    slack = cur_H_y;
                    sid = id;
                }
                // bounded SV
                if (sum_alpha[i] >= cost1-precision && alpha[id] > 0) bounded_sv++;
            }
            slacks[i] = slack;
            slacks_id[i] = sid;
        }

        double slack_sum = 0;
        int slack_num = 0;
        for (i=0; i < slacks.size(); i++) {
            if (slacks[i] > 0) {
                slack_sum += slacks[i] + eps1;
                slack_num++;
            } else {
                slack_sum += eps1;
            }
        }

        // primal cost 계산: 0.5 * |w|^2 + C/n * L
        // dual object value 계산: sum(Loss*a) - 0.5 * |w|^2
        double primal_cost, dual_cost;
        if (domain_adaptation) primal_cost = 0.5 * diff_obj * diff_obj + cost1 * slack_sum;
        else primal_cost = 0.5 * obj * obj + cost1 * slack_sum;
        if (domain_adaptation) dual_cost = alphasum - (0.5 * diff_obj * diff_obj) + additional_value;
        else dual_cost = alphasum - (0.5 * obj * obj);

        cerr << "Training time= " << time << endl;
        cerr << endl << "const=" << const_num << " SV=" << sv_num << " bounded_SV=" << bounded_sv << endl;
        cerr << "alphasum=" << alphasum << " sum(a)=" << sum << " max(a)=" << max_alpha << endl;
        cerr << "slack_num=" << slack_num << " sum(slack)=" << slack_sum << endl;
        cerr << "|w|=" << obj << endl;
        cerr << "primal_cost(upper bound)=" << primal_cost << endl;
        cerr << "dual object=" << dual_cost << endl;
        cerr << "duality gap=" << primal_cost - dual_cost << endl;
        cerr << "longest ||Psi(x,y)-Psi(x,ybar)||=" << longest_vector() << endl;
		cerr << "Runtime(sec): QP=" << time_qp << " Argmax=" << time_viol << " psi=" << time_psi << endl;
        cerr << "Runtime(%): QP=" << 100*time_qp/time << " Argmax=" << 100*time_viol/time;
        cerr << " psi=" << 100*time_psi/time << " others=" << 100*(time-time_qp-time_viol-time_psi)/time << endl;
        cerr << "Number of calls to 'find_most_violated_constraint': " << argmax_count << endl;

        save_slack(eps);
    } catch (std::exception& e) {
        cerr << endl << "std::exception caught:" << e.what() << endl;
    }

    // free
    if (domain_adaptation) {
        delete[] prior_theta;
    }

    return time;
}


/// fixed threshold SMO + joint constraint (like SVM-Perf)
double SSVM::train_fsmo_joint(bool use_gram) {
    int n = train_data.size();

    int niter = 1, const_num = work_set.size();
    int correct = 0, argmax_count = 0;
    double ceps = 0, obj = 0, diff_obj = 0, additional_value = 0;
    double time = 0, time_qp = 0, time_viol = 0, time_psi = 0;

    // for domain adaptation
    float *prior_theta = NULL;
    if (domain_adaptation) {
        cerr << "[Domain adaptation mode]" << endl;
        prior_theta = new float[n_theta];
        for (int i=0; i < n_theta; i++) {
            prior_theta[i] = theta[i];
        }
    }

#ifdef _OPENMP
    if (buf < omp_get_max_threads()) omp_set_num_threads(buf);
	cerr << "[OpenMP] Number of threads: " << omp_get_max_threads() << endl;
#endif

    // comment
    if (!use_comment) vector<string>().swap(train_data_comment);

    // gram 초기화
    if (use_gram) {
        gram.clear();
        for (int i=0; i < gram_size; i++) {
            vector<float> gram_i;
            for (int j=0; j < gram_size; j++) {
                gram_i.push_back(-1);
            }
            gram.push_back(gram_i);
        }
    }

    // slack 초기화
    double slack = 0;

    // inactive constraint 제거
    int opt_count = 0;

    if (domain_adaptation) {
        printf("iter     accuracy  training  time    |w| |w-w0|    primal      dual const SV\n");
        printf("============================================================================");
    } else {
        printf("iter     accuracy  training  time    |w|    primal      dual const  SV\n");
        printf("======================================================================");
    }
    fflush(stdout);

    double cost1 = cost;
    double eps1 = 100;
    double old_eps = eps1;

    // dense vector
    vector<float> dense_vect(n_theta, 0);

    do { // increase precision
        timer t;
        
        correct = 0;
        ceps = 0.0;

        // a joint vector : linear만 고려
        vect_t joint_vect;
        single_vect_t s_joint_vect;
        double joint_loss = 0, joint_cost_diff = 0;
        // dense vector 초기화
        for (size_t i=0; i < dense_vect.size(); i++) {
            dense_vect[i] = 0;
        }

        // slack 계산
        slack = 0;
		#pragma omp parallel for
        for (int i=0; i < (int)work_set.size(); i++) {
            double cur_cost_diff;
            cur_cost_diff = calculate_cost(work_set[i]);
			#pragma omp critical (slack)
            slack = MAX(slack, loss[i] - cur_cost_diff);
        }

        if (!owps_format) {
            // make M_i matrix
            make_M_matrix();
        }

        // find a violated joint contraint
        double sum_viol = 0;
        
		#pragma omp parallel for
		for (int sent_index = 0; sent_index < (int)train_data.size(); sent_index++) {
			sent_t& sent = train_data[sent_index];
            vector<int> y_seq;

            // find most violated contraint
            timer t_viol;
			#pragma omp atomic
            argmax_count++;
			// for openMP
			vector<double> r_vec;
			// make M_i matrix
            make_R_matrix(r_vec, sent);
            y_seq = find_most_violated_constraint(r_vec, sent, 1);
			#pragma omp atomic
            time_viol += t_viol.elapsed();

            // sentence가 맞았는지 검사
            bool all_correct = true;
            for (size_t i=0; i < sent.size(); i++) {
                if (sent[i].outcome == y_seq[i]) correct++;
                else all_correct = false;
            }
            // 다 맞았으면 skip
            if (all_correct) continue;

            double cur_loss = calculate_loss(sent, y_seq);

            // for CPA
            timer t_psi;
            vect_t vect;
            vect = make_diff_vector(sent, y_seq);
			#pragma omp critical (dense_vect)
            append_diff_vector(dense_vect, vect);

			#pragma omp atomic
            time_psi += t_psi.elapsed();
            // loss 값은 다 더한다
			#pragma omp atomic
            joint_loss += cur_loss;
        } // example loop

        joint_loss = joint_loss / double(n);

        double norm = 0.0;
        size_t non_empty_count = 0;
        for (int i=0; i < n_theta; i++) {
            if (dense_vect[i] != 0) {
                norm += dense_vect[i] * dense_vect[i];
                non_empty_count++;
            }
        }
        s_joint_vect.twonorm_sq = norm;
        s_joint_vect.factor = 1 / double(n);

        // sparse vector로 변경한다
        for (int i=0; i < n_theta; i++) {
            if (dense_vect[i] != 0) {
                s_joint_vect.vect.push_back(make_pair(i,dense_vect[i]));
            }
        }
        joint_vect.push_back(s_joint_vect);

        // joint vector의 H_y 계산
        joint_cost_diff = calculate_cost(joint_vect);
        ceps = MAX(0, joint_loss - joint_cost_diff - slack);

        // w*x - b 에서 b 제거
        if (verbose || slack > (joint_loss - joint_cost_diff + 1e-12)) {
            if (slack > (joint_loss - joint_cost_diff + 1e-12)) {
                cerr << endl << "WARNING: Slack of most violated constraint is smaller than slack of working" << endl;
                cerr << "         set! There is probably a bug in 'find_most_violated_constraint_*'.";
            }
            cerr << endl << "H(y)=" << joint_loss-joint_cost_diff << " slack=" << slack << " ceps=" << joint_loss-joint_cost_diff-slack << endl;
            cerr << "loss=" << joint_loss << " cost=" << joint_cost_diff << endl;
        }

        // if error, then add a joint constraint
        if (ceps > eps) {
            //cerr << ".";
            // alpha
            alpha.push_back(0);
            // alpha_history
            alpha_history.push_back(opt_count);
            // work set
            work_set.push_back(joint_vect);
            // loss
            loss.push_back(joint_loss);
            // x_norm_vec
            x_norm_vec.push_back(s_joint_vect.factor * s_joint_vect.factor * s_joint_vect.twonorm_sq);

            const_num++;

            old_eps = eps1;
            eps1 = MIN(eps1, MAX(ceps, eps));
            if (old_eps != eps1) {
                cerr << endl << "# eps = " << eps1 << " ";
            }

            // get new QP solution
            cerr << "*";
            timer t_qp;
            optimize_dual4fsmo_joint(cost1, eps1, use_gram);
            time_qp += t_qp.elapsed();

            // inactive constraint 제거
            // rm_inactive iteration 동안 active된 적이 없는 것
            // svm-light는 50
            opt_count++;
            int remove_count = 0;
            for (int i=0; i < work_set.size() - remove_count; i++) {
                // active constraint
                if (alpha[i] > 0) {
                    alpha_history[i] = opt_count;
                }
                // rm_inactive번 안에 active 된 적이 없는 constraint
                else if (opt_count - alpha_history[i] >= rm_inactive) {
                    // 맨 뒤의 원소부터 swap한 후, remove_count만큼 맨뒤 원소들을 삭제
                    int sw_i = work_set.size() - 1 - remove_count;
                    work_set[i] = work_set[sw_i];
                    alpha[i] = alpha[sw_i];
                    alpha_history[i] = alpha_history[sw_i];
                    loss[i] = loss[sw_i];
                    x_norm_vec[i] = x_norm_vec[sw_i];
                    // gram matrix
                    if (use_gram) {
                        // 1차 배열 수정
                        gram[i] = gram[sw_i];
                        // 2차 배열 수정
                        for (int j=0; j < gram_size; j++) {
                            gram[j][i] = gram[j][sw_i];
                        }
                        // 삭제되는 곳에 -1
                        for (int j=0; j < gram_size; j++) {
                            gram[sw_i][j] = -1;
                            gram[j][sw_i] = -1;
                        }
                    }
                    // cost_diff_vec
                    if (use_gram) {
                        cost_diff_vec[i] = cost_diff_vec[sw_i];
                    }
                    // 나머지 변수 처리
                    i--;
                    const_num--;
                    remove_count++;
                }
            }
            if (remove_count > 0) {
                // 실제 제거
                int rm_i = work_set.size() - remove_count;
                work_set.erase(work_set.begin() + rm_i, work_set.end());
                alpha.erase(alpha.begin() + rm_i, alpha.end());
                alpha_history.erase(alpha_history.begin() + rm_i, alpha_history.end());
                loss.erase(loss.begin() + rm_i, loss.end());
                x_norm_vec.erase(x_norm_vec.begin() + rm_i, x_norm_vec.end());
                // cost_diff_vec
                if (use_gram) {
                    cost_diff_vec.erase(cost_diff_vec.begin() + rm_i, cost_diff_vec.end());
                }
                // test
                cerr << "r";
            }
        }

        // time
        double iter_time = t.elapsed();
        time += iter_time;

        // sv number
        int sv_num = 0;
        double sum = 0, alphasum = 0;
        for (size_t i=0; i < alpha.size(); i++) {
            if (alpha[i] != 0) {
                sum += alpha[i];
                alphasum += alpha[i] * loss[i];
                sv_num++;
            }
        }

        // obj --> model length |w|
        obj = 0;
        diff_obj = 0;
        additional_value = 0;
        if (!skip_eval || ceps < eps ) {
            for (int i=0; i < n_theta; i++) {
                obj += SQUARE(theta[i]);
                if (domain_adaptation && theta[i] != prior_theta[i]) {
                    diff_obj += SQUARE(theta[i] - prior_theta[i]);
                    additional_value += prior_theta[i] * (prior_theta[i] - theta[i]);
                }
            }
            obj = sqrt(obj);
            diff_obj = sqrt(diff_obj);
        }

        // continue evaluations
        double acc = 100*double(correct)/double(n_event);

        // test_data 성능
        correct = 0;
        if (obj != 0 || diff_obj != 0) {
            vector<sent_t>::iterator it = test_data.begin();
            // make M_i matrix
            make_M_matrix();
            for (; it != test_data.end(); it++) {
                sent_t& sent = *it;
                double prob;
                vector<int> y_seq;
                // make M_i matrix
                vector<double> r_vec; // for openMP
                make_R_matrix(r_vec, sent);
                y_seq = viterbi(r_vec, sent, prob);
            
                for (size_t i=0; i < sent.size(); i++) {
                    if (sent[i].outcome == y_seq[i]) correct++;
                }
            }
        }
        double test_acc = test_data.size() > 0 ? 100*double(correct)/double(n_test_event) : 0;

        // primal cost 계산: 0.5 * |w|^2 + C/n * L
        double primal_cost = 0, dual_cost = 0;
        if (obj != 0 || diff_obj != 0) {
            if (domain_adaptation) primal_cost = 0.5 * diff_obj * diff_obj + cost1 * (slack + ceps);
            else primal_cost = 0.5 * obj * obj + cost1 * (slack + ceps);
            if (domain_adaptation) dual_cost = alphasum - (0.5 * diff_obj * diff_obj) + additional_value;
            else dual_cost = alphasum - (0.5 * obj * obj);
        }

        if (domain_adaptation) {
            printf("\n%3d %5.2f%% %5.2f%% %6.2f %8.2f %6.2f %6.2f %9.2e %9.2e %3d %3d ", 
                    niter++ , acc, test_acc, iter_time, time, obj, diff_obj, primal_cost, dual_cost, const_num, sv_num);
        } else {
            printf("\n%3d %5.2f%% %5.2f%% %6.2f %8.2f %6.2f %9.2e %9.2e %4d %4d ", 
                    niter++ , acc, test_acc, iter_time, time, obj, primal_cost, dual_cost, const_num, sv_num);
        }
        if (!skip_eval || eps1 != old_eps || ceps < eps) print_status();
        fflush(stdout);

        // eps 마다 저장: 0.5, 0.1 -- by leeck
        if (period > 0 && eps1 > old_eps && eps1 < 1)
        {
            cerr << endl << "model saving to " << model_file << "." << eps1 << " ... ";
            char temp[100];
            sprintf(temp, "%s.%g", model_file.c_str(), eps1);
			if (binary) save_bin(string(temp));
            else save(string(temp));
            cerr << "done." << endl;
        }
    } while (ceps > eps);

    // test
    if (domain_adaptation) {
        printf("\niter     accuracy  training  time    |w| |w-w0|    primal      dual const SV\n");
    } else {
        printf("\niter     accuracy  training  time    |w|    primal      dual const  SV\n");
    }

    try {
        int const_num = 0, sv_num = 0;
        double sum = 0, alphasum = 0;
        double max_alpha = 0.0;
        const_num = alpha.size();
        for (size_t i=0; i < alpha.size(); i++) {
            if (alpha[i] != 0) {
                sum += alpha[i];
                alphasum += alpha[i] * loss[i];
                sv_num++;
            }
            max_alpha = MAX(max_alpha, alpha[i]);
        }

        slack = 0;
        for (size_t i=0; i < work_set.size(); i++) {
            slack = MAX(slack, loss[i] - calculate_cost(work_set[i]));
        }

        // primal cost 계산: 0.5 * |w|^2 + C/n * L
        double primal_cost, dual_cost;
        if (domain_adaptation) primal_cost = 0.5 * diff_obj * diff_obj + cost1 * (slack + ceps);
        else primal_cost = 0.5 * obj * obj + cost1 * (slack + ceps);
        if (domain_adaptation) dual_cost = alphasum - (0.5 * diff_obj * diff_obj) + additional_value;
        else dual_cost = alphasum - (0.5 * obj * obj);

        cerr << "Training time= " << time << endl;
        cerr << endl << "Final epsilon on KKT-Conditions: " << ceps << endl;
        cerr << "const=" << const_num << " SV=" << sv_num << " alphasum=" << alphasum << " sum(a)=" << sum << " max(a)=" << max_alpha << endl;
        cerr << "slack=" << slack << endl;
        cerr << "|w|=" << obj << endl;
        cerr << "primal_cost(upper bound)=" << primal_cost << endl;
        cerr << "dual object=" << dual_cost << endl;
        cerr << "duality gap=" << primal_cost - dual_cost << endl;
        cerr << "longest ||Psi(x,y)-Psi(x,ybar)||=" << longest_vector() << endl;
		cerr << "Runtime(sec): QP=" << time_qp << " Argmax=" << time_viol << " psi=" << time_psi << endl;
        cerr << "Runtime(%): QP=" << 100*time_qp/time << " Argmax=" << 100*time_viol/time;
        cerr << " psi=" << 100*time_psi/time << " others=" << 100*(time-time_qp-time_viol-time_psi)/time << endl;
        cerr << "Number of calls to 'find_most_violated_constraint': " << argmax_count << endl;
    } catch (std::exception& e) {
        cerr << endl << "std::exception caught:" << e.what() << endl;
    }

    // free
    if (domain_adaptation) {
        delete[] prior_theta;
    }

    return time;
}


/// SVM-struct + primal optimization + Stochastic Gradient Descent
/// f = 0.5 * lambda * |w|^2 + (1/n)sum{L(x,y;w)}
/// hindge loss: g = lambda*w - (1/n)sum{delta(psi(i,y))}
/// domain_adaptation: 0 = no domain adaptation, 1 = domain adaptation
double SSVM::train_pegasos() {
    int i, j, k;

    int niter = 1, weight_num = 0;
    int correct = 0, total = 0, argmax_count = 0;
    double time = 0, time_qp = 0, time_viol = 0, time_psi = 0;

    // for domain adaptation
    float *prior_theta = NULL;
    if (domain_adaptation) {
        cerr << "[Domain adaptation mode]" << endl;
        prior_theta = new float[n_theta];
        for (int i=0; i < n_theta; i++) {
            prior_theta[i] = theta[i];
            // 0 부터 시작인지 w0부터 시작인지?
            //theta[i] = 0;
        }
    }

#ifdef _OPENMP
    if (buf < omp_get_max_threads()) omp_set_num_threads(buf);
	cerr << "[OpenMP] Number of threads: " << omp_get_max_threads() << endl;
#endif

    if (domain_adaptation) {
        printf("iter primal_cost      |w|  |w-w0|  d(cost)   training acc.  training time\n");
        printf("=========================================================================");
    } else {
        printf("iter primal_cost      |w|   d(cost)   training acc.  training time\n");
        printf("==================================================================");
    }
    fflush(stdout);

    // C/n
    double n = (double) train_data.size();
    // lambda = 1/C
    double lambda = 1.0 / cost;
    //double lambda = 1.0 / cost1;

    double f = 0.0, old_f = 0.0;
    double wscale = 1, old_wscale = 1;
    double obj = 0, diff_obj = 0;
    double dcost = 1;
    double t_i = 0;
    double best_acc = 0, test_acc = 0;
    int best_iter = 0;

    vector<int> train_data_index;
    if (1) {
        for (i=0; i < train_data.size(); i++)
            train_data_index.push_back(i);
    }

    for (; niter <= iter; niter++) {
        timer t;
        total = 0;
        correct = 0;
        old_f = f;
        f = 0;

        if (1) {
            cerr << "r";
            random_shuffle(train_data_index.begin(), train_data_index.end());
            cerr << ".";
        }

        for (size_t sent_i = 0; sent_i < n;) {
            make_M_matrix();
			#pragma omp parallel for private(j)
            for (i = 0; i < buf; i++) {
				int skip = 0;
				#pragma omp critical (sent_i)
				if (sent_i++ >= n) skip = 1;
				//if (sent_i++ >= train_data.size()) skip = 1;
				if (skip) continue;
                // choose random example
				int r;
                if (1) {
					// 다시 한번 체크해 줌
					if (sent_i-1 >= n) continue;
					//if (sent_i-1 >= train_data_index.size()) continue;
                    r = train_data_index[sent_i-1];
                }
                sent_t& sent = train_data[r];
    
                // find most violated contraint
				#pragma omp atomic
                argmax_count++;

                timer t_viol;
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);
                vector<int> y_seq = find_most_violated_constraint(r_vec, sent, wscale);

				#pragma omp atomic
                time_viol += t_viol.elapsed();
    
                // sentence가 맞았는지 검사
                for (j=0; j < sent.size(); j++) {
					#pragma omp atomic
                    total++;
					if (sent[j].outcome == y_seq[j]) {
						#pragma omp atomic
						correct++;
					}
                }
    
                double cur_loss = calculate_loss(sent, y_seq);
    
                timer t_psi;
                vect_t max_vect = make_diff_vector(sent, y_seq);

				#pragma omp atomic
                time_psi += t_psi.elapsed();
    
                double cost_diff = calculate_cost(max_vect);
                // wscale 반영
                cost_diff *= wscale;
    
                double H_y = cur_loss - cost_diff;
    
				#pragma omp critical (work_set)
                if (H_y > 0) {
                    // work set
                    work_set.push_back(max_vect);
                    // loss
                    loss.push_back(H_y);
                    // f: hinge loss
                    f += H_y / n;
                }
            }
    
            // Stochastic Gradient Decent
            timer t_qp;
            // pegasos
            double eta = 1.0 / (lambda * (2 + t_i));
            // check eta
            if (eta * lambda > 0.9) {
                cerr << "e";
                eta = 0.9 / lambda;
            }
    
            double s = 1 - eta * lambda;
            if (s > 0) {
                old_wscale = wscale;
                wscale *= s;
            }
    
            // update w
            for (i=0; i < work_set.size(); i++) {
                vect_t& max_vect = work_set[i];
                for (j=0; j < max_vect.size(); j++) {
                    double factor = 0;
                    // hinge loss: g = lambda*w - (1/n)sum{delta(psi(i,y))}
                    factor = (1.0 / wscale) * eta * max_vect[j].factor / buf;
                    for (k=0; k < max_vect[j].vect.size(); k++) {
                        int fid = max_vect[j].vect[k].first;
                        double val = max_vect[j].vect[k].second;
                        // test
                        //cerr << "fid=" << fid << " theta=" << theta[fid] <<  " val=" << factor * val << endl;
                        double old_theta = theta[fid];
                        // update theta
                        theta[fid] += factor * val;
                        // update obj : obj는 scale을 뺀 값을 저장
                        if (!domain_adaptation) {
                            obj -= old_theta * old_theta;
                            obj += theta[fid] * theta[fid];
                        }
                    }
                }
            }

            // domain adaptation: w = (1-eta*lambda)w + eta*lambda*w0 + gradient
            // eta*lambda*w0 부분 처리
            if (domain_adaptation) {
                double factor = (1.0 / wscale) * eta * lambda;
                #pragma omp parallel for
                for (int i=0; i < n_theta; i++) {
                    theta[i] += factor * prior_theta[i];
                }
            }
    
            // scaling
            if (wscale < 1e-7) {
                cerr << "s";
                //cerr << endl << "wscale=" << wscale;
				#pragma omp parallel for
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) {
                        theta[i] *= wscale;
                    }
                }
                if (!domain_adaptation) {
                    obj *= wscale * wscale;
                }
                wscale = 1;
                cerr << ".";
            }

            work_set.clear();
            loss.clear();
            t_i += 1;
            time_qp += t_qp.elapsed();
        }

        double iter_time = t.elapsed();
        time += iter_time;

        // calculate obj, diff_obj when domain_adaptation model
        if (domain_adaptation) {
            obj = 0;
            diff_obj = 0;
            #pragma omp parallel for
            for (int i=0; i < n_theta; i++) {
                obj += theta[i] * theta[i];
                diff_obj += (theta[i]-prior_theta[i]/wscale) * (theta[i]-prior_theta[i]/wscale);
            }
        }

        // f
        if (domain_adaptation) {
            f += 0.5 * lambda * (wscale*wscale*diff_obj);
        } else {
            f += 0.5 * lambda * (wscale*wscale*obj);
        }
        dcost = (dcost + ABS(old_f - f)/MAX(old_f,f)) / 2.0;

        // continue evaluations
        acc = correct/double(total);

        // test_data 성능
        correct = 0;
        if (!skip_eval || (period == 0 && niter % 10 == 0) || (period > 0 && niter % period == 0) || dcost < threshold || niter == iter) {
            // scaling
            if (wscale < 1) {
                cerr << "s";
				#pragma omp parallel for
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) {
                        theta[i] *= wscale;
                    }
                }
                obj *= wscale * wscale;
                wscale = 1;
            }
            // make M_i matrix
            make_M_matrix();

			#pragma omp parallel for private(j)
            for (int i = 0; i < test_data.size(); i++) {
                sent_t& sent = test_data[i];
                //  make M_i matrix
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);

                double prob;
                vector<int> y_seq = viterbi(r_vec, sent, prob);

                for (j=0; j < sent.size(); j++) {
					if (sent[j].outcome == y_seq[j]) {
						#pragma omp atomic
						correct++;
					}
                }
            }
        }
        test_acc = test_data.size() > 0 ? 100*double(correct)/double(n_test_event) : 0;
        if (test_acc > best_acc) {
            best_acc = test_acc;
            best_iter = niter;
        }

        if (domain_adaptation) {
            printf("\n%4d  %.3e %8.3f %7.3f %9.6f %6.2f%% %6.2f%% %6.2f %7.2f ", niter,
                cost*f, sqrt(wscale*wscale*obj), sqrt(wscale*wscale*diff_obj), dcost, (acc*100), test_acc, iter_time, time);
        } else {
            printf("\n%4d  %.3e %8.3f %9.6f %6.2f%% %6.2f%% %6.2f %7.2f ", niter,
                cost*f, sqrt(wscale*wscale*obj), dcost, (acc*100), test_acc, iter_time, time);
        }
        if (correct > 0) print_status();
        fflush(stdout);

        // 끝
        if (dcost < threshold) {
            printf("\nTraining terminats succesfully in %.2f seconds\n", time);
            break;
        }

        // 중간중간 저장 -- by leeck
        if (period > 0 && niter < iter && (niter == period || (best_acc == test_acc && niter > period))) {
            // scaling
            if (wscale < 1) {
                cerr << "s";
				#pragma omp parallel for
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) {
                        theta[i] *= wscale;
                    }
                }
                obj *= wscale * wscale;
                wscale = 1;
            }

            char temp[200];
            if (model_file != "") {
                timer t_save;
                cerr << "model saving to " << model_file << "." << niter << " ... ";
                sprintf(temp, "%s.%d", model_file.c_str(), niter);
                if (binary) save_bin(string(temp));
				else save(string(temp));
                cerr << "done (" << t_save.elapsed() << ")." << endl;
            }
        } // end of train_data
    }

    // scaling
    if (wscale < 1) {
        cerr << "s";
		#pragma omp parallel for
        for (int i=0; i < n_theta; i++) {
            if (theta[i] != 0) {
                theta[i] *= wscale;
            }
        }
        obj *= wscale * wscale;
        wscale = 1;
    }

    if (niter > iter) {
        printf("\nMaximum numbers of %d iterations reached in %.2f seconds", iter, time);
    }
    
    cout << endl << "Best acc: " << best_acc << " % at " << best_iter;

    cout << endl << "Runtime(%): SGD=" << 100*time_qp/time << " Argmax=" << 100*time_viol/time;
    cout << " psi=" << 100*time_psi/time << " others=" << 100*(time-time_qp-time_viol-time_psi)/time << endl;
    cout << "Number of calls to 'find_most_violated_constraint': " << argmax_count << endl;

    // free
    if (domain_adaptation) {
        delete[] prior_theta;
    }

    return time;
}


/// print start status
void SSVM::print_start_status(string estimate) {
    printf("\nStarting %s iterations...\n", estimate.c_str());
    printf("Number of Sentences : %d\n", (int)train_data.size());
    printf("Number of Words     : %d\n", n_event);
    printf("Number of Predicates: %d\n", n_pred);
    printf("Number of Outcomes  : %d\n", n_outcome);
    printf("Number of Parameters: %d\n", n_theta);
    printf("[Common] Hash Feature   : %d\n", hash_feature);
    printf("[Common] Support Feature: %d\n", support_feature);
    printf("[Common] General Feature: %d\n", general_feature);
    printf("[Common] OWPS format    : %d\n", owps_format);
    printf("[Common] Threshold      : %g\n", threshold);
    printf("[Common] Beam Search    : %d\n", beam);
    printf("[SVM] Cost:%g C/n:%g buf:%d\n", cost, cost/double(train_data.size()), buf);
    if (estimate == "fsmo" || estimate == "fsmo_joint" || estimate == "fsmo_joint2") {
        printf("[SVM] eps:%g rm_inactive:%d\n", cost, cost/double(train_data.size()), eps, buf, rm_inactive);
        printf("[SVM] final_opt_check:%d\n", final_opt_check);
    }
    fflush(stdout);
}


/// kernel function
double SSVM::kernel(vect_t& vect1, vect_t& vect2) {
    register int i, j;
    register double result = 0.0;
    int n1 = vect1.size();
    int n2 = vect2.size();

    for (i=0; i < n1; i++) {
        for (j=0; j < n2; j++) {
            result += vect1[i].factor * vect2[j].factor * dot_product(vect1[i], vect2[j]);
        }
    }

    return result;
}


/// kernel function for joint constraint
double SSVM::kernel4gram(int v1, int v2) {
    int i, j;
    int max = MAX(v1,v2);

    if (max >= gram_size) {
        cerr << endl << "GRAM size is small: " << gram_size << " , extended:" << 2*gram_size << " ... ";
        // 기존 항목 size 늘임
        for (i=0; i < gram_size; i++) {
            for (j=gram_size; j < 2*gram_size; j++) {
                gram[i].push_back(-1);
            }
        }
        // 추가
        for (i=gram_size; i < 2*gram_size; i++) {
            vector<float> gram_i;
            for (j=0; j < 2*gram_size; j++) {
                gram_i.push_back(-1);
            }
            gram.push_back(gram_i);
        }
        gram_size *= 2;
        cerr << "Done." << endl;
    }

    float result = gram[v1][v2];

    if (result == -1) {
        float val = kernel(work_set[v1], work_set[v2]);
        gram[v1][v2] = val;
        gram[v2][v1] = val;
        result = val;
    }
    return result;
}


/// dot_product
double SSVM::dot_product(single_vect_t& svect1, single_vect_t& svect2) {
    register double result = 0.0;
    
    vector<pair<int, float> >::iterator it1 = svect1.vect.begin();
    vector<pair<int, float> >::iterator it1_end = svect1.vect.end();
    vector<pair<int, float> >::iterator it2 = svect2.vect.begin();
    vector<pair<int, float> >::iterator it2_end = svect2.vect.end();
    
    while (it1 != it1_end && it2 != it2_end) {
        if (it1->first == it2->first) {
            result += it1->second * it2->second;
            it1++;
            it2++;
        }
        else if (it1->first > it2->first) {
            it2++;
        }
        else {
            it1++;
        }
    }
    
    return result;
}


/// optimize dual for fsmo: shared_slack 버전 : S_i 마다 적용
/// non-bound와 bound를 분리해서 실행
void SSVM::optimize_dual4fsmo(double cost, double eps1) {
    timer t;
    int changed_num = 0, qiter = 0, i, j, k;
    int smo_count = 0;

    // shrink: svm-light와 비슷(100): qiter - lastiter > 100
    vector<int> shrink(work_set.size(), 0);
    vector<int> lastiter(work_set.size(), 0);
    int shrink_round = 1, shrink_count = 0;
    int final_opt_check1 = final_opt_check;

    do {
        changed_num = 0;

        // S_i 마다 적용
        for (size_t sent_i=0; sent_i < work_set_ids.size(); sent_i++) {
            // non-bound인 경우
            if (sum_alpha[sent_i] < cost - precision) {
                for (size_t work_set_i=0; work_set_i < work_set_ids[sent_i].size(); work_set_i++) {
                    i = work_set_ids[sent_i][work_set_i];
                    // shrink, final_opt_check 발동시에는 shrink 안함
                    if (shrink[i] == shrink_round && (final_opt_check == final_opt_check1)) {
                        shrink_count++;
                        continue;
                    } 
    
                    double x_norm = x_norm_vec[i];
                    if (x_norm == 0) continue;
    
                    double H_y, cost_diff;
                    cost_diff = calculate_cost(work_set[i]);
                    H_y = loss[i] - cost_diff;
    
                    // KKT condition check : 0 <= sum(a[i]) <= C
                    if ( (H_y > 0.5*eps1 && alpha[i] < cost - precision)
                        || (H_y < -0.5*eps1 && alpha[i] > precision)
                       ) {
                        // margin 구하기
                        double org_margin = (loss[i] - cost_diff) / x_norm;
    
                        // lower bound : sum(alpha_i) is clipped to the [0,C]
                        double L = -alpha[i];
                        // upper bound : sum(alpha_i) is clipped to the [0,C]
                        double H = cost - sum_alpha[sent_i];
    
                        double margin = MAX(L, MIN(H, org_margin));
    
                        // w 업데이트
                        if (margin > precision || margin < -precision) {
                            update_weight(work_set[i], margin);
                            // alpha 업데이트
                            alpha[i] += margin;
                            sum_alpha[sent_i] += margin;
                            changed_num++;
                            // for shrink
                            lastiter[i] = qiter;
                        }
                    }
                }
            }
            // bound 되었을 경우, working set selection (SMO 알고리즘 적용)
            // S_i에만 적용
            else {
                double g_max = -1e10, g_min = 1e10, obj_min = 1e10;
                int max_i = -1, min_j = -1;

                // g (= H_y) 저장
                vector<double> g_vec(work_set_ids[sent_i].size(), 0); 

                // i 선택
				#pragma omp parallel for private (i)
                for (int work_set_i=0; work_set_i < (int)work_set_ids[sent_i].size(); work_set_i++) {
                    i = work_set_ids[sent_i][work_set_i];
                    if (x_norm_vec[i] == 0) continue;

                    double H_y, cost_diff;
                    cost_diff = calculate_cost(work_set[i]);
                    H_y = loss[i] - cost_diff;
                    g_vec[work_set_i] = H_y;
                    
					#pragma omp critical (g_max)
                    if (H_y > g_max) {
                        g_max = H_y;
                        max_i = i;
                    }
                }
                if (max_i < 0) continue;
                i = max_i;

                // j 선택
                j = -1;
                double min_H_y = 0;
                for (k=0; k < work_set_ids[sent_i].size(); k++) {
                    j = work_set_ids[sent_i][k];
                    if (x_norm_vec[j] == 0) continue;
                    if (j != i && alpha[j] > 0) {
                        /*
                        double cur_H_y;
                        cur_H_y = loss[j] - calculate_cost(work_set[j]);
                        */
                        double cur_H_y = g_vec[k];
                        // g_min
                        g_min = MIN(g_min, cur_H_y);
                        // first order
                        if (-(g_max - cur_H_y) < obj_min) {
                            obj_min = -(g_max - cur_H_y);
                            min_H_y = cur_H_y;
                            min_j = j;
                        }
                        // second order
                        /*
                        double prod_ij;
                        prod_ij = kernel(work_set[i], work_set[j]);
                        double x_norm = x_norm_vec[i] + x_norm_vec[j] - 2*prod_ij;
                        double cur_obj = -SQUARE(g_max - cur_H_y) / x_norm;
                        if (cur_obj < obj_min) {
                            obj_min = cur_obj;
                            min_H_y = cur_H_y;
                            min_j = j;
                        }
                        */
                    }
                }
                if (j < 0) continue;
                if (g_max - g_min < eps1) continue;
                j = min_j;
    
                double prod_ij = kernel(work_set[i], work_set[j]);
                double x_norm = x_norm_vec[i] + x_norm_vec[j] - 2*prod_ij;
                // a_j margin
                double margin = -(g_max - min_H_y) / x_norm;
                margin = MAX(-alpha[j], margin);
                margin = MIN(alpha[i], margin);

                if (margin > precision || margin < -precision) {
                    update_weight(work_set[i], -margin);
                    update_weight(work_set[j], margin);
                    // alpha 업데이트
                    alpha[i] -= margin;
                    alpha[j] += margin;
                    smo_count++;
                    //if (smo_count % 10000 == 0) cerr << "+";

                    changed_num++;
                    // for shrink
                    lastiter[i] = qiter;
                    lastiter[j] = qiter;
                }
            }
        }
        qiter++;

        // final opt check
        if (final_opt_check1 > 0 && changed_num == 0 && shrink_count > 0) {
            final_opt_check1--;
            cerr << "F";
            changed_num = 1;
        }
    } while (qiter < 100000 && changed_num > 0);

    // shrink 정보
    if (shrink_count > 0) {
        cerr << "s";
    }

    if (smo_count > 0) {
        //cerr << "+" << smo_count;
        cerr << "+";
    }

    // 걸린 시간
    if (qiter >= 10000 || t.elapsed() >= 1000) {
        cerr << "(" << qiter << ":" << t.elapsed() << ")";
    }
}



/// optimize dual for fsmo_joint: shared_slack
/// non-bound와 bound를 분리해서 실행
void SSVM::optimize_dual4fsmo_joint(double cost, double eps1, int use_gram) {
    timer t;
    int changed_num = 0, qiter = 0, i, j, k;
    int smo_count = 0;

    // shrink: svm-light와 비슷(100): qiter - lastiter > 100
    vector<int> shrink(work_set.size(), 0);
    vector<int> lastiter(work_set.size(), 0);
    int shrink_round = 1, shrink_count = 0;
    
    // cost_diff 계산
    if (use_gram) {
        for (i = (int)cost_diff_vec.size(); i < (int)work_set.size(); i++) {
            double cost_diff = calculate_cost(work_set[i]);
            cost_diff_vec.push_back(cost_diff);
        }
    }

    // diff_alpha
    vector<double> diff_alpha(alpha.size(), 0);

    // sum_alpha 계산 : sum(alpha) <= C
    double sum_alpha = 0;
    for (i=0; i < (int)work_set.size(); i++) {
        sum_alpha += alpha[i];
    }

    //double old_b = b;

    int final_opt_check1 = final_opt_check;

    do {
        changed_num = 0;

        // non-bound인 경우
        if (sum_alpha < cost - precision) {
            for (i=0; i < (int)work_set.size(); i++) {
                // shrink, final_opt_check 발동시에는 shrink 안함
                if (shrink[i] == shrink_round && (final_opt_check == final_opt_check1)) {
                    shrink_count++;
                    continue;
                }

                double x_norm = x_norm_vec[i];
                if (x_norm == 0) continue;

                double H_y, cost_diff;
                if (!use_gram) {
                    cost_diff = calculate_cost(work_set[i]);
                } else {
                    cost_diff = cost_diff_vec[i];
                }
                H_y = loss[i] - cost_diff;

                // KKT condition check : 0 <= sum(a[i]) <= C
                if ((H_y > 0.5*eps1 && alpha[i] < cost - precision)
                    || (H_y < -0.5*eps1 && alpha[i] > precision)
                ) {
                    // margin 구하기
                    double org_margin = (loss[i] - cost_diff) / x_norm;

                    // lower bound : sum(alpha) is clipped to the [0,C]
                    double L = -alpha[i];
                    // upper bound : sum(alpha) is clipped to the [0,C]
                    double H = cost - sum_alpha;

                    double margin = MAX(L, MIN(H, org_margin));

                    // w 업데이트
                    if (margin > precision || margin < -precision) {
                        if (!use_gram) {
                            update_weight(work_set[i], margin);
                        }
                        // alpha 업데이트
                        alpha[i] += margin;
                        diff_alpha[i] += margin;
                        sum_alpha += margin;

                        // cost_diff_vec 업데이트
                        if (use_gram) {
                            for (j=0; j < (int)work_set.size(); j++) {
                                double prod = kernel4gram(j, i);
                                cost_diff_vec[j] += margin * prod;
                            }
                        }

                        changed_num++;
                        // for shrink
                        lastiter[i] = qiter;
                    }
                }
            }
        }
        // bound 되었을 경우, working set selection (SMO 알고리즘 적용)
        else {
            double g_max = -1e10, g_min = 1e10, obj_min = 1e10;
            int max_i = -1, min_j = -1;

            // i 선택
            for (i=0; i < (int)work_set.size(); i++) {
                if (x_norm_vec[i] == 0) continue;
                if (alpha[i] > cost - precision) continue;

                double H_y, cost_diff;
                if (!use_gram) {
                    cost_diff = calculate_cost(work_set[i]);
                } else {
                    cost_diff = cost_diff_vec[i];
                }
                H_y = loss[i] - cost_diff;

                if (H_y > g_max) {
                    g_max = H_y;
                    max_i = i;
                }
                // first order
                /*
                if (alpha[i] > 0) {
                    if (H_y < g_min) {
                        g_min = H_y;
                        min_j = i;
                        // 특정 조건 이상이면 중간에 멈춤 - 테스트 중
                        //if (g_max - g_min > 2*eps1) {cerr << "@"; break;}
                    }
                }
                */
            }
            if (max_i < 0) continue;
            i = max_i;

            // second order - j 선택
            double min_H_y = 0;
            for (j=0; j < (int)work_set.size(); j++) {
                if (j != i && alpha[j] > precision) {
                    double cur_H_y;
                    if (!use_gram) {
                        cur_H_y = loss[j] - calculate_cost(work_set[j]);
                    } else {
                        cur_H_y = loss[j] - cost_diff_vec[j];
                    }
                    // g_min
                    g_min = MIN(g_min, cur_H_y);
                    // second order
                    double prod_ij;
                    if (use_gram) {
                        prod_ij = kernel4gram(i, j);
                    } else {
                        prod_ij = kernel(work_set[i], work_set[j]);
                    }
                    double x_norm = x_norm_vec[i] + x_norm_vec[j] - 2*prod_ij;
                    double cur_obj = -SQUARE(g_max - cur_H_y) / x_norm;
                    if (cur_obj < obj_min) {
                        obj_min = cur_obj;
                        min_H_y = cur_H_y;
                        min_j = j;
                    }
                }
            }

            if (min_j < 0) continue;
            if (g_max - g_min < eps1) continue;
            j = min_j;

            double prod_ij;
            if (use_gram) {
                prod_ij = kernel4gram(i, j);
            } else {
                prod_ij = kernel(work_set[i], work_set[j]);
            }
            double x_norm = x_norm_vec[i] + x_norm_vec[j] - 2*prod_ij;
            // a_j margin
            // first order
            //double margin = (g_max - g_min) / x_norm;
            // second order
            double margin = (g_max - min_H_y) / x_norm;
            margin = MIN(alpha[j], margin);

            if (margin > precision || margin < -precision) {
                if (!use_gram) {
                    update_weight(work_set[i], margin);
                    update_weight(work_set[j], -margin);
                }
                // alpha 업데이트
                alpha[i] += margin;
                alpha[j] -= margin;
                diff_alpha[i] += margin;
                diff_alpha[j] -= margin;

                // cost_diff_vec 업데이트
                if (use_gram) {
                    for (k=0; k < (int)work_set.size(); k++) {
                        double prod_i, prod_j;
                        prod_i = kernel4gram(k, i);
                        prod_j = kernel4gram(k, j);
                        cost_diff_vec[k] += margin * prod_i;
                        cost_diff_vec[k] -= margin * prod_j;
                    }
                }

                smo_count++;
                changed_num++;
                // for shrink
                lastiter[i] = qiter;
                lastiter[j] = qiter;
            }
        }
        qiter++;

        // final opt check
        if (final_opt_check1 > 0 && changed_num == 0 && shrink_count > 0) {
            final_opt_check1--;
            cerr << "F";
            changed_num = 1;
        }
    } while (qiter < 100000 && changed_num > 0);

    // diff_alpha 값 w에 반영
    if (use_gram) {
        for (i=0; i < (int)diff_alpha.size(); i++) {
            if (diff_alpha[i] != 0)
                update_weight(work_set[i], diff_alpha[i]);
        }
    }

    // shrink 정보
    if (shrink_count > 0) {
        cerr << "s";
    }

    if (smo_count > 0) {
        cerr << "+";
    }

    // 걸린 시간
    if (qiter >= 10000 || t.elapsed() >= 1000) {
        cerr << "(" << qiter << ":" << t.elapsed() << ")";
    }
}


/// update weight vector
void SSVM::update_weight(vect_t& vect, double d) {
    register size_t i, j;
    register int fid;
    register double factor = 0;
    for (i=0; i < vect.size(); i++) {
        factor = d * vect[i].factor;
        for (j=0; j < vect[i].vect.size(); j++) {
            fid = vect[i].vect[j].first;
            theta[fid] += factor * vect[i].vect[j].second;
        }
    }
}


/// calculate cost : LINEAR: w * f(x,y)
double SSVM::calculate_cost(vect_t& vect) {
    int i;
    register int j, n;
    double result = 0;
    register double cur_result = 0;
    pair<int,float> *vect_p = NULL;
    
    for (i=0; i < vect.size(); i++) {
        single_vect_t& svect = vect[i];
        // sparse vector
        if (!svect.vect.empty()) {
            vect_p = &(svect.vect[0]);
            n = svect.vect.size();
            cur_result = 0;
            for (j=0; j < n; j++) {
                //fid = svect.vect[j].first;
                //cur_result += theta[fid] * svect.vect[j].second;
                cur_result += theta[vect_p[j].first] * vect_p[j].second;
            }
            result += svect.factor * cur_result;
        }
    }
    // w * f(x,y)
    return result;
}


/// save slack for FSMO : 에러 검증에 활용
void SSVM::save_slack(double eps) {
    // model_file이 정해지지 않으면 저장하지 않는다
    if (model_file == "")
        return;

    // slack 값 sorting
    int n = train_data.size();
    vector<pair<double, int> > slack_vec;
    for (int i=0; i < n; i++) {
        if (slacks[i] > eps) {
            slack_vec.push_back(make_pair(slacks[i], i));
        }
    }
    sort(slack_vec.begin(), slack_vec.end());
    reverse(slack_vec.begin(), slack_vec.end());

    // slack 값 파일에 저장
    string file = model_file + ".slack.txt";
    ofstream outfile(file.c_str());
    if (!outfile) {
        cerr << "Can not open data file to write: " << file << endl;
        return;
        //exit(1);
    }

    outfile << "# test answer output feature" << endl;

    for (size_t i=0; i < slack_vec.size(); i++) {
        int id = slack_vec[i].second;
        if (slacks[id] > eps) {
            outfile << "# " << id << "\t" << slacks[id] << endl;
            if (use_comment && id < (int)train_data_comment.size() && train_data_comment[id] != "") {
                outfile << train_data_comment[id] << endl;
            }
            sent_t& sent = train_data[id];
            int sid = slacks_id[id];
            for (size_t j=0; j < sent.size(); j++) {
                // test
                if (sent[j].outcome == y_seq_vec[sid][j]) outfile << "O\t";
                else outfile << "X\t";
                // 정답
                outfile << outcome_vec[sent[j].outcome] << "\t";
                // 추정치
                outfile << outcome_vec[y_seq_vec[sid][j]] << "\t";
                // predicate 출력 ("qid=", "L0=", "0=" 포함한 걸 우선, 없으면 첫번째 것)
                if (!pred_vec.empty()) {
                    size_t k;
                    context_t& cont = sent[j].context;
                    for (k=0; k < cont.size(); k++) {
                        string fi = pred_vec[cont[k].pid];
                        if (fi.find("qid=") < string::npos) {
                            outfile << fi;
                            break;
                        }
                        if (fi.find("0=") < string::npos) {
                            outfile << fi;
                            break;
                        }
                    }
                    if (k == cont.size() && !cont.empty()) {
                        outfile << pred_vec[cont[0].pid];
                    }
                }
                outfile << endl;
            }
            outfile << endl;
        } else {
            break;
        }
    }
    outfile.close();
}


/// length of longest vector
double SSVM::longest_vector() {
    double max_len = 0, len = 0;

    for (size_t i=0; i < alpha.size(); i++) {
        len = sqrt(kernel(work_set[i], work_set[i]));
        if (len > max_len) {
            max_len = len;
        }
    }

    return max_len;
}


/// show feature weight
void SSVM::show_feature() {
    for (size_t pid = 0; pid < pred_vec.size(); pid++) {
        if (support_feature) {
            double obj = 0;
            vector<pair<int, int> >& param = params[pid];
            for (size_t j = 0; j < param.size(); ++j) {
                int oid = param[j].first;
                int fid = param[j].second;
                cout << fid << "\t" << pred_vec[pid] << "\t";
                cout << outcome_vec[oid] << "\t" << theta[fid] << endl;
                obj += SQUARE(theta[fid]);
            }
            cout << "# " << pid << "\t" << pred_vec[pid] << "\t" << sqrt(obj) << endl;
        } else {
            double obj = 0;
            for (size_t oid = 0; oid < n_outcome; oid++) {
                int fid = make_fid(pid, oid);
                cout << fid << "\t" << pred_vec[pid] << "\t";
                cout << outcome_vec[oid] << "\t" << theta[fid] << endl;
                obj += SQUARE(theta[fid]);
            }
            cout << "# " << pid << "\t" << pred_vec[pid] << "\t" << sqrt(obj) << endl;
        }
    }
}


/// remove zero feature
void SSVM::remove_zero_feature(double threshold) {
    // hash crf인 경우 skip
    if (pred_vec.empty()) return;
    
    map<string, int> new_pred_map;
    vector<string> new_pred_vec;
    float *new_theta = new float[n_theta];
    
    int new_fid = 0, remove_count = 0;
    
    if (support_feature) {
        vector<vector<pair<int, int> > > new_params;
        
        for (size_t pid = 0; pid < pred_vec.size(); pid++) {
            if (get_feature_weight(pid) < threshold) {
                remove_count++;
                continue;
            }
            string pred = pred_vec[pid];
            new_pred_map[pred] = new_pred_vec.size();
            new_pred_vec.push_back(pred);
            
            vector<pair<int, int> > new_param;
            vector<pair<int, int> >& param = params[pid];
            for (size_t j = 0; j < param.size(); j++) {
                int oid = param[j].first;
                int fid = param[j].second;
                if (ABS(theta[fid]) > precision) {
                    new_param.push_back(make_pair(oid,new_fid));
                    new_theta[new_fid++] = theta[fid];
                }
            }
            new_params.push_back(new_param);
        }
        
        params.clear();
        params = new_params;
    } else {
        for (size_t pid = 0; pid < pred_vec.size(); pid++) {
            if (get_feature_weight(pid) < threshold) {
                remove_count++;
                continue;
            }
            string pred = pred_vec[pid];
            new_pred_map[pred] = new_pred_vec.size();
            new_pred_vec.push_back(pred);
            for (size_t oid = 0; oid < n_outcome; oid++) {
                int fid = make_fid(pid, oid);
                new_theta[new_fid++] = theta[fid];
            }
        }
    }
    
    cerr << "remove zero feature (threshold=" << threshold << "):" << endl;
    cerr << "\t" << remove_count << " pred (" << 100.0*remove_count/pred_vec.size() << "%): " << pred_vec.size() << " --> " << new_pred_vec.size() << endl;
    cerr << "\t" << n_theta - new_fid << " theta (" << 100.0*(n_theta-new_fid)/n_theta << "%): " << n_theta << " --> " << new_fid << endl;
    
    // theta 정확한 사이즈로
    n_theta = new_fid;
    delete[] theta;
    theta = new float[n_theta];
    for (int i = 0; i < n_theta; i++) {
        theta[i] = new_theta[i];
    }
    delete[] new_theta;
    
    pred_map = new_pred_map;
    pred_vec = new_pred_vec;
    n_pred = pred_vec.size();
}


/// convert all feature for support feature
void SSVM::to_support_feature() {
    if (support_feature) {
        cerr << "support feature mode!" << endl;
        return;
    }
    
    params.clear();
    float *new_theta = new float[n_theta];
    int new_fid = 0;
    
    for (size_t pid = 0; pid < pred_vec.size(); pid++) {
        vector<pair<int, int> > param;
        for (int oid = 0; oid < n_outcome; oid++) {
            int fid = make_fid(pid, oid);
            if (ABS(theta[fid]) > precision) {
                param.push_back(make_pair(oid,new_fid));
                new_theta[new_fid++] = theta[fid];
            }
        }
        params.push_back(param);
    }
    
    cerr << "convert all feature to support feature:" << endl;
    cerr << "\t" << pred_vec.size() << " pred" << endl;
    cerr << "\t" << n_theta - new_fid << " theta (" << 100.0*(n_theta-new_fid)/n_theta << "%): " << n_theta << " --> " << new_fid << endl;
    
    // theta 정확한 사이즈로
    n_theta = new_fid;
    delete[] theta;
    theta = new float[n_theta];
    for (int i = 0; i < n_theta; i++) {
        theta[i] = new_theta[i];
    }
    delete[] new_theta;
    
    // support_feature 변경
    support_feature = 1;
}


/// convert support feature to all feature
void SSVM::to_all_feature() {
    if (!support_feature) {
        cerr << "all feature mode!" << endl;
        return;
    }
    
    int new_n_theta = pred_vec.size() * n_outcome;
    float *new_theta = new float[new_n_theta];
    for (int i = 0; i < new_n_theta; i++) {
        new_theta[i] = 0;
    }
    
    for (size_t i = 0; i < params.size(); i++) {
        vector<pair<int, int> >& param = params[i];
        for (size_t j = 0; j < param.size(); j++) {
            int outcome = param[j].first;
            int fid = param[j].second;
            int new_fid = i * n_outcome + outcome;
            new_theta[new_fid] = theta[fid];
        }
    }
    
    cerr << "convert support feature to all feature:" << endl;
    cerr << "\t" << pred_vec.size() << " pred" << endl;
    cerr << "\t" << new_n_theta << " theta (" << 100.0*new_n_theta/n_theta << "%): " << n_theta << " --> " << new_n_theta << endl;
    
    // theta 정확한 사이즈로
    n_theta = new_n_theta;
    delete[] theta;
    theta = new float[n_theta];
    for (int i = 0; i < n_theta; i++) {
        theta[i] = new_theta[i];
    }
    delete[] new_theta;
    
    // all_feature 변경
    support_feature = 0;
}

