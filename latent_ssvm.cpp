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

#pragma warning(disable: 4786)
#pragma warning(disable: 4996)
#pragma warning(disable: 4267)
#pragma warning(disable: 4244)
#pragma warning(disable: 4018)

#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string.h>

#include "latent_ssvm.hpp"
#include "timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

Latent_SSVM::Latent_SSVM() {
}


/// CCCP + Pagasos.
/// h is the position of node_t in sent_t.
/// use_SPL : Self-Paced learning
double Latent_SSVM::train_latent_ssvm(int use_SPL) {
    int niter, weight_num = 0;
    int correct = 0, total = 0, argmax_count = 0, skip_count = 0;
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

    double f = 0.0, old_f = 0.0;
    double wscale = 1, old_wscale = 1;
    double obj = 0, diff_obj = 0;
    double dcost = 1;
    double best_acc = 0, test_acc = 0;
    int best_iter = 0;
    double t_i = 1;
    double K = 10;

    vector<int> train_data_index;
    for (int i=0; i < train_data.size(); i++) {
        train_data_index.push_back(i);
    }

    for (niter = 1; niter <= iter; niter++) {
        timer t;
        total = 0;
        correct = 0;
        old_f = f;
        f = 0;
        skip_count = 0;
        // self-paced learning
        K = K / 1.3;

        // randome suffle
        random_shuffle(train_data_index.begin(), train_data_index.end());

        for (size_t sent_i = 0; sent_i < train_data.size();) {
            //make_M_matrix();
			#pragma omp parallel for
            for (int i = 0; i < buf; i++) {
				int skip = 0;
				#pragma omp critical (sent_i)
				if (sent_i++ >= train_data.size()) skip = 1;
				if (skip) continue;

                // choose random example
				int r;
				// 다시 한번 체크해 줌
				if (sent_i-1 >= train_data_index.size()) continue;
                r = train_data_index[sent_i-1];
                sent_t& sent = train_data[r];
				// 빈 문장 skip
				if (sent.empty()) continue;

                // curriculum learning: easiness = |H| (i.e. sent.size())
                /*
                //if (niter <= 50) {}
                    //if ((2+niter/5) < sent.size()) {}
                //if (niter <= 10) {}
                    //if ((2+niter) < sent.size()) {}
                if (niter <= 5) {
                    if ((1+2*niter) < sent.size()) {
                        //cerr << "S";
				        #pragma omp atomic
                        skip_count++;
                        continue;
                    }
                }
                */
    
                // find most violated contraint
				#pragma omp atomic
                argmax_count++;

                timer t_viol;
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);

                // hi_star
                int hi_star = -1;
                double max = -1e10;
                // random selection h at initial stage
                if (niter <= 1) {
                    hi_star = rand() % sent.size();
                } else {
                    for (int j=0; j < sent.size(); j++) {
                        int yi = sent[j].outcome;
                        if (r_vec[MAT2(j,yi)] > max) {
                            hi_star = j;
                            max = r_vec[MAT2(j,yi)];
                        }
                        // yi == "O" --> hi_star = 0
                        //if (outcome_vec[yi] == "O") hi_star = 0;
                    }
                }
                // y_hat, h_hat
                int y_hat = -1, h_hat = -1;
                max = -1e10;
                // random selection h at initial stage
                if (niter <= 1) {
                    h_hat = rand() % sent.size();
                    for (int y = 0; y < n_outcome; y++) {
                        // h > 0 && y == "O" --> skip
                        //if (h_hat > 0 && outcome_vec[y] == "O") continue;

                        double score = wscale * r_vec[MAT2(h_hat,y)];
                        if (y != sent[h_hat].outcome) score += 1.0;
                        if (score > max) {
                            y_hat = y;
                            max = score;
                        }
                    }
                } else {
                    for (int j=0; j < sent.size(); j++) {
                        for (int y = 0; y < n_outcome; y++) {
                            // h > 0 && y == "O" --> skip
                            //if (j > 0 && outcome_vec[y] == "O") continue;

                            double score = wscale * r_vec[MAT2(j,y)];
                            if (y != sent[j].outcome) score += 1.0;
                            if (score > max) {
                                h_hat = j;
                                y_hat = y;
                                max = score;
                            }
                        }
                    }
                }

				#pragma omp atomic
                time_viol += t_viol.elapsed();
    
                // 맞았는지 검사
				#pragma omp atomic
                total++;
                if (sent[h_hat].outcome == y_hat) {
					#pragma omp atomic
					correct++;
				}
    
                //calculate loss
                int yi = sent[hi_star].outcome;
                double cur_loss = (yi == y_hat ? 0 : 1);
    
                timer t_psi;
                vect_t max_vect = make_diff_vector(sent, yi, hi_star, y_hat, h_hat);
				#pragma omp atomic
                time_psi += t_psi.elapsed();
    
                double cost_diff = calculate_cost(max_vect);
                // wscale 반영
                cost_diff *= wscale;
    
                double H_y = cur_loss - cost_diff;
    
                // Self-Paced learning
                if (use_SPL) {
				    #pragma omp critical (work_set)
                    if (H_y > 0 && (niter <= 2 || H_y < 1/K)) { // self-paced learning: T0=2, mu=1.3
                        // work set
                        work_set.push_back(max_vect);
                        // loss
                        loss.push_back(H_y);
                        // f: hinge loss
                        f += H_y / n;
                    }
                    if (niter > 2 && H_y >= 1/K) {
                        #pragma omp atomic
                        skip_count++;
                    }
                } else {
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
            }
    
            // Stochastic Gradient Decent
            timer t_qp;
            // pegasos
            double eta = 1.0 / (lambda * (1 + t_i));
            // check eta
            if (eta * lambda > 0.99) {
                cerr << "e";
                eta = 0.99 / lambda;
            }
    
            double s = 1 - eta * lambda;
            if (s > 0) {
                old_wscale = wscale;
                wscale *= s;
            }
    
            // update w
            for (int i=0; i < work_set.size(); i++) {
                vect_t& max_vect = work_set[i];
                for (int j=0; j < max_vect.size(); j++) {
                    double factor = 0;
                    // hinge loss: g = lambda*w - (1/n)sum{delta(psi(i,y))}
                    factor = (1.0 / wscale) * eta * max_vect[j].factor / buf;
                    for (int k=0; k < max_vect[j].vect.size(); k++) {
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
        total = 0;
        correct = 0;
        if (!skip_eval || (period == 0 && niter % 10 == 0) || (period > 0 && niter % period == 0) || dcost < threshold || niter == iter) {
            // scaling
            if (wscale < 1e-7) {
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

			#pragma omp parallel for
            for (int i = 0; i < test_data.size(); i++) {
                sent_t& sent = test_data[i];
				// 빈 문장 skip
				if (sent.empty()) continue;
                //  make R matrix
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);

                // y_hat, h_hat
                int y_hat = -1, h_hat = -1;
                double max = -1e10;
                for (int j=0; j < sent.size(); j++) {
                    for (int y = 0; y < n_outcome; y++) {
                        if (r_vec[MAT2(j,y)] > max) {
                            h_hat = j;
                            y_hat = y;
                            max = r_vec[MAT2(j,y)];
                        }
                    }
                }

                #pragma omp atomic
                total++;
				if (sent[h_hat].outcome == y_hat) {
					#pragma omp atomic
					correct++;
                }
            }
        }
        test_acc = test_data.size() > 0 ? 100*double(correct)/double(total) : 0;
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
        if (skip_count > 0) cerr << "skip=" << skip_count << " ";
        if (correct > 0) print_status();
        fflush(stdout);

        // 끝
        if (dcost < threshold) {
            printf("\nTraining terminats succesfully in %.2f seconds\n", time);
            break;
        }

        // 중간중간 저장 -- by leeck
        if (period > 0 && niter < iter && (niter % period == 0 || (best_acc == test_acc && niter > period))) {
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

            if (model_file != "" && best_acc == test_acc) {
                timer t;
                cerr << "model saving to " << model_file << "." << niter << " ... ";
                char temp[200];
                sprintf(temp, "%s.%d", model_file.c_str(), niter);
                save(string(temp));
                cerr << "done (" << t.elapsed() << ")." << endl;
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

/// make_diff_vector : f(xi,yi,hi*) - f(xi,y,h).
/// h is the position of node_t in sent_t
vect_t Latent_SSVM::make_diff_vector(sent_t& sent, int yi, int hi, int y, int h) {
    vect_t vect;
    single_vect_t svect;
    map<int, float> vect_map;
    
    if (yi != y || hi != h) {
        // (xi, yi, hi*)
        context_t& cont1 = sent[hi].context;
        int outcome = yi;
        context_t::iterator cit = cont1.begin();
        for (; cit != cont1.end(); cit++) {
            int pid = cit->pid;
#ifndef BINARY_FEATURE
            double fval = cit->fval;
#else
            double fval = 1;
#endif
            // f(xi, yi, hi)
            if (support_feature) {
                int fid = make_fid(pid, outcome);
                if (fid >= 0) {
                    if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = fval;
                    else vect_map[fid] += fval;
                }
            } else {
                int fid = pid * n_outcome + outcome;
                if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = fval;
                else vect_map[fid] += fval;
            }
        }
        // (xi, y, h)
        context_t& cont2 = sent[h].context;
        outcome = y;
        cit = cont2.begin();
        for (; cit != cont2.end(); cit++) {
            int pid = cit->pid;
#ifndef BINARY_FEATURE
            double fval = cit->fval;
#else
            double fval = 1;
#endif
            // - f(xi, y, h)
            if (support_feature) {
                int fid = make_fid(pid, outcome);
                if (fid >= 0) {
                    if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -fval;
                    else vect_map[fid] -= fval;
                }
            } else {
                int fid = pid * n_outcome + outcome;
                if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -fval;
                else vect_map[fid] -= fval;
            }
        }
    } else {
        return vect;
    }
    
    double norm = 0.0;
    map<int, float>::iterator it = vect_map.begin();
    for (; it != vect_map.end(); it++) {
        if (it->second != 0) {
            svect.vect.push_back(make_pair(it->first, it->second));
            norm += it->second * it->second;
        }
    }
    
    // |vect|^2 구하기
    svect.twonorm_sq = norm;
    svect.factor = 1.0;
    
    vect.push_back(svect);
    
    return vect;
}


// for prediction
int Latent_SSVM::predict(ostream& f) {
    vector<sent_t>::iterator it = test_data.begin();
    int correct = 0;
    int total = 0;
    int sent_i = 0;

    // 파일에 쓰기
    f << "# output answer latent_var candidate_num score (feature)" << endl;

    timer t;

    for (; it != test_data.end(); ++it) {
        sent_t& sent = *it;
		if (sent.empty()) continue;
        int y, h;
        double score = eval(sent, y, h);

        // comment
        if (use_comment) {
            f << test_data_comment[sent_i++] << endl;
        }

        int ans = sent[h].outcome;
        f << outcome_vec[y] << "\t" << outcome_vec[ans] << "\t" << h << "\t" << sent.size() << "\t" << score;
        // event detect에서 해당 feature 출력
        if (!hash_feature) {
            int is_printed = 0;
            for (int i=0; i < sent[h].context.size(); i++) {
                int pid = sent[h].context[i].pid;
                string& feature = pred_vec[pid];
                if (!strncmp(feature.c_str(), "pred=", 5)) {
                    f << "\t" << feature;
                    is_printed = 1;
                    break;
                }
            }
            if (!is_printed && !sent[h].context.empty()) {
                int pid = sent[h].context[0].pid;
                string& feature = pred_vec[pid];
                f << "\t" << feature;
            }
        }
        f << endl;
        total++;
        if (y == ans) correct++;
    }

    cout << "Accuracy: " << 100.0 * correct / total << "%\t("
        << correct << "/" << total << ")" << endl;

    // 소요 시간 출력
    cout << t.elapsed() << " sec, " << n_test_event / t.elapsed() << " tokens per sec (" << n_test_event << " / " << t.elapsed() << ")" << endl;

    return correct;
}

// for prediction
int Latent_SSVM::predict_nbest(ostream& f, int nbest) {
    vector<sent_t>::iterator it = test_data.begin();
    int correct = 0;
    int total = 0;
    int sent_i = 0;
    vector<double> r_vec; // for openMP

    // 파일에 쓰기
    f << "# output answer latent_var candidate_num score (feature)" << endl;

    timer t;

    for (; it != test_data.end(); ++it) {
        sent_t& sent = *it;
		if (sent.empty()) continue;
        int y, h;
        //double score = eval(sent, y, h);

        r_vec.clear();
        make_R_matrix(r_vec, sent);
        y = h = -1;
        double score = -1e10;
        for (int i=0; i < sent.size(); i++) {
            for (int j = 0; j < n_outcome; j++) {
                if (r_vec[MAT2(i,j)] > score) {
                    h = i;
                    y = j;
                    score = r_vec[MAT2(i,j)];
                }
            }
        }

        // comment
        if (use_comment) {
            f << test_data_comment[sent_i++] << endl;
        }

        // print 1-best
        int ans = sent[h].outcome;
        f << outcome_vec[y] << "\t" << outcome_vec[ans] << "\t" << h << "\t" << sent.size() << "\t" << score;
        // event detect에서 해당 feature 출력
        if (!hash_feature) {
            int is_printed = 0;
            for (int i=0; i < sent[h].context.size(); i++) {
                int pid = sent[h].context[i].pid;
                string& feature = pred_vec[pid];
                if (!strncmp(feature.c_str(), "pred=", 5)) {
                    f << "\t" << feature;
                    is_printed = 1;
                    break;
                }
            }
            if (!is_printed && !sent[h].context.empty()) {
                int pid = sent[h].context[0].pid;
                string& feature = pred_vec[pid];
                f << "\t" << feature;
            }
        }
        f << endl;
        total++;
        if (y == ans) correct++;

        // support only 2-best
        int y2, h2;
        y2 = h2 = -1;
        double score2 = -1e10;
        for (int i=0; i < sent.size(); i++) {
            if (i == h) continue;
            for (int j = 0; j < n_outcome; j++) {
                if (r_vec[MAT2(i,j)] > score2) {
                    h2 = i;
                    y2 = j;
                    score2 = r_vec[MAT2(i,j)];
                }
            }
        }
        // print second best
        if (y2 >= 0 && h2 >= 0) {
            f << "#2 " << outcome_vec[y2] << "\t" << outcome_vec[ans] << "\t" << h2 << "\t" << sent.size() << "\t" << score2;
            // event detect에서 해당 feature 출력
            if (!hash_feature) {
                int is_printed = 0;
                for (int i=0; i < sent[h2].context.size(); i++) {
                    int pid = sent[h2].context[i].pid;
                    string& feature = pred_vec[pid];
                    if (!strncmp(feature.c_str(), "pred=", 5)) {
                        f << "\t" << feature;
                        is_printed = 1;
                        break;
                    }
                }
                if (!is_printed && !sent[h2].context.empty()) {
                    int pid = sent[h2].context[0].pid;
                    string& feature = pred_vec[pid];
                    f << "\t" << feature;
                }
            }
            f << endl;
        }
    }

    cout << "Accuracy: " << 100.0 * correct / total << "%\t("
        << correct << "/" << total << ")" << endl;

    // 소요 시간 출력
    cout << t.elapsed() << " sec, " << n_test_event / t.elapsed() << " tokens per sec (" << n_test_event << " / " << t.elapsed() << ")" << endl;

    return correct;
}

// for prediction
double Latent_SSVM::eval(sent_t& sent, int& y, int& h) {
    vector<double> r_vec; // for openMP
    make_R_matrix(r_vec, sent);

    y = h = -1;
    double max = -1e10;
    for (int i=0; i < sent.size(); i++) {
        for (int j = 0; j < n_outcome; j++) {
            // h > 0 && y == "O" --> skip
            //if (i > 0 && outcome_vec[j] == "O") continue;

            if (r_vec[MAT2(i,j)] > max) {
                h = i;
                y = j;
                max = r_vec[MAT2(i,j)];
            }
        }
    }

    return max;
}

