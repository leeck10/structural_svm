/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
    @file joint_ssvm.cpp
    @brief Joint model using Latent Structural SVMs
        support sequence labeling
        assume F(x,y,z) = F(x) * Y * Z
    @author Changki Lee (leeck@kangwon.ac.kr)
    @date 2013/7/16
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

#include "joint_ssvm.hpp"
#include "timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

Joint_SSVM::Joint_SSVM() {
}

/// first, load latent train data for y or z
void Joint_SSVM::load_latent_event(const string file, bool is_y_train_data) {
    string line, comment;
    int count = 0;
    int old_n_theta = n_theta;
    int old_pred_vec_size = pred_vec.size();

    // to do: support feature
    if (support_feature) {
        cerr << endl << "Error: cannot use support feature mode!" << endl;
        exit(1);
    }

    if (hash_feature && support_feature) {
        cerr << "Hash + Support feature mode is not avalable!" << endl;
        exit(1);
    }

    ifstream f(file.c_str());
    if (!f) {
        cerr << "Can not open data file to read: " << file << endl;
        exit(1);
    }

    sent_t sent;
    context_t cont;

    while (getline(f, line)) {
        // remove newline for windows format file
        string find_str = "\r";
        string::size_type find_pos = line.find(find_str);
        if (string::npos != find_pos) {
            line.replace(find_pos, find_str.size(), "");
        }
        if (line.empty()) {
            if (!owps_format) {
                if (is_y_train_data) {
                    y_train_data.push_back(sent);
                    sent.clear();
                    if (train_num > 0 && train_num <= y_train_data.size()) {
                        break;
                    }
                } else {
                    z_train_data.push_back(sent);
                    sent.clear();
                    if (train_num > 0 && train_num <= z_train_data.size()) {
                        break;
                    }
                }
            }
        } else if (line[0] == '#') {
            // comment 처리
            if (use_comment) {
                if (comment == "") {
                    comment += line;
                } else {
                    comment += "\n";
                    comment += line;
                }
            }
        } else {
            /// Tokenizer
            vector<string> tokens;
            tokenize(line, tokens, " \t");
            vector<string>::iterator it = tokens.begin();

            // outcome
            int oid;
            if (is_y_train_data) {
                if (y_outcome_map.find(*it) == y_outcome_map.end()) {
                    //cerr << *it << " ";
                    oid = y_outcome_map.size();
                    y_outcome_map[*it] = oid;
                    y_outcome_vec.push_back(*it);
                    if (verbose) cerr << "New y outcome: " << *it << endl;
                } else {
                    oid = y_outcome_map[*it];
                    if (y_outcome_vec[oid] != *it) {
                        cerr << "oid=" << oid << " : " << y_outcome_vec[oid] << " != " << *it << endl;
                        exit(1);
                    }
                }
            } else {
                if (z_outcome_map.find(*it) == z_outcome_map.end()) {
                    //cerr << *it << " ";
                    oid = z_outcome_map.size();
                    z_outcome_map[*it] = oid;
                    z_outcome_vec.push_back(*it);
                    if (verbose) cerr << "New z outcome: " << *it << endl;
                } else {
                    oid = z_outcome_map[*it];
                    if (z_outcome_vec[oid] != *it) {
                        cerr << "oid=" << oid << " : " << z_outcome_vec[oid] << " != " << *it << endl;
                        exit(1);
                    }
                }
            }

            node_t node;
            node.outcome = oid;
            node.start = sent.size();
            node.end = sent.size();

            it++;
            cont.clear();
            for (; it != tokens.end(); it++) {
                string fi(it->c_str());
                // test
                //cerr << " " << *it;
                // pred
                int pid;
                bool new_pid = false;

                // qid:n.m 처리
                if (fi.find("qid:") != string::npos) {
                    vector<string> str_vec, str_vec2;
                    split(fi, str_vec, ":");
                    split(str_vec[1], str_vec2, ".");
                    node.start = atoi(str_vec2[0].c_str());
                    node.end = atoi(str_vec2[1].c_str());
                    continue;
                }

                float fval = 1.0;
#ifndef BINARY_FEATURE
                if (general_feature && fi.find(":") != string::npos) {
                    vector<string> str_vec;
                    split(fi, str_vec, ":");
                    fi = str_vec[0];
                    fval = atof(str_vec[1].c_str());
                }
#endif
                
                // hash
                if (hash_feature) {
                    new_pid = false;
                    pid = hash(fi) % n_pred;
                }
                else if (pred_map.find(fi) == pred_map.end()) {
                    new_pid = true;
                    pid = pred_vec.size();
                    pred_map[fi] = pid;
                    pred_vec.push_back(fi);
                }
                else {
                    new_pid = false;
                    pid = pred_map[fi];
                }

                feature_t feature;
                feature.pid = pid;
#ifndef BINARY_FEATURE
                feature.fval = fval;
#endif
                cont.push_back(feature);

            }

            node.context = cont;
            sent.push_back(node);

            if (owps_format) {
                if (is_y_train_data) {
                    y_train_data.push_back(sent);
                    sent.clear();
                    if (train_num > 0 && train_num <= y_train_data.size()) {
            	        ++count;
                        break;
                    }
                } else {
                    z_train_data.push_back(sent);
                    sent.clear();
                    if (train_num > 0 && train_num <= z_train_data.size()) {
            	        ++count;
                        break;
                    }
                }
            }

            ++count;
            if (count % 10000 == 0) {
                cerr << ".";
                if (count % 100000 == 0) cerr << " ";
                if (is_y_train_data) {
                    if (count % 500000 == 0) cerr << "\t" << count << " " << y_train_data.size() << endl;
                } else {
                    if (count % 500000 == 0) cerr << "\t" << count << " " << z_train_data.size() << endl;
                }
            }
        }
    }
    if (!sent.empty()) {
        if (is_y_train_data) {
            y_train_data.push_back(sent);
        } else {
            z_train_data.push_back(sent);
        }
    }

    if (!hash_feature) {
        n_pred = pred_vec.size();
    }
}

/// second, load joint train data: y+h
void Joint_SSVM::load_event(const string joint_file) {
    // second, load joint train data (y+h)
    SSVM::load_event(joint_file);

    // test
    cerr << "Y :";
    for (int i=0; i < y_outcome_vec.size(); i++) cerr << " " << y_outcome_vec[i];
    cerr << endl;
    cerr << "Z :";
    for (int i=0; i < z_outcome_vec.size(); i++) cerr << " " << z_outcome_vec[i];
    cerr << endl;
    cerr << "Y:Z :";
    for (int i=0; i < outcome_vec.size(); i++) cerr << " " << outcome_vec[i];
    cerr << endl;

    // if there is no train data for z,
    // z_outcome_map, z_outcome_vec
    // y+z is represented as "y:z"
    for (int i=0; i < y_outcome_vec.size(); i++) {
        string y = y_outcome_vec[i];
        for (int j=0; j < outcome_vec.size(); j++) {
            string y_z = outcome_vec[j];
            string z;
            if (y_z.find(y+":") != string::npos) {
                size_t idx = y_z.find(y+":");
                y_z.replace(idx, y.size()+1, "");
                z = y_z;
                //cerr << " " << y << " " << z;
            }
            if (z.size() > 0) {
                if (z_outcome_map.find(z) == z_outcome_map.end()) {
                    // z
                    z_outcome_map[z] = z_outcome_vec.size();
                    z_outcome_vec.push_back(z);
                }
                // joint2y_map
                joint2y_map[j] = i;
                // joint2z_map
                joint2z_map[j] = z_outcome_map[z];
                //cerr << " (" << outcome_vec[j] << ", " << y << ", " << z << ")";
            }
        }
    }
    cerr << endl;

    // test
    cerr << "Z :";
    for (int i=0; i < z_outcome_vec.size(); i++) cerr << " " << z_outcome_vec[i];
    cerr << endl;

    // test
    cerr << "joint2y_map :";
    for (int i=0; i < joint2y_map.size(); i++) {
        cerr << " " << outcome_vec[i] << "=" << y_outcome_vec[joint2y_map[i]];
    }
    cerr << endl;

    // test
    cerr << "joint2z_map :";
    for (int i=0; i < joint2z_map.size(); i++) {
        cerr << " " << outcome_vec[i] << "=" << z_outcome_vec[joint2z_map[i]];
    }
    cerr << endl;
}

/// Joint learning using Pegasos
/// use_SPL: Self-Paced learning
double Joint_SSVM::train_joint_ssvm(int use_SPL) {
    int niter, weight_num = 0;
    int correct = 0, correct_y = 0, correct_z = 0, total = 0, argmax_count = 0;
    int skip_count = 0, shrink_count = 0;
    double time = 0, time_qp = 0, time_viol = 0, time_psi = 0;

    // for best model
    float *best_theta = NULL;
    if (1) {
        best_theta = new float[n_theta];
        for (int i=0; i < n_theta; i++) best_theta[i] = theta[i];
    }

    cerr << "[Joint SSVM] y_cost: " << y_cost << " z_cost: " << z_cost << endl;
    cerr << "Number of Sentences in total train data: " << train_data.size() + y_train_data.size() + z_train_data.size() << endl;
    cerr << "Number of Sentences in joint data      : " << train_data.size() << endl;
    cerr << "Number of Sentences in y_data          : " << y_train_data.size() << endl;
    cerr << "Number of Sentences in z_data          : " << z_train_data.size() << endl;
    cerr << "Number of Sentences in test joint data : " << test_data.size() << endl;
    cerr << "Number of y variable                   : " << y_outcome_vec.size() << endl;
    cerr << "Number of z variable                   : " << z_outcome_vec.size() << endl;
    cerr << "Number of y+z variable                 : " << outcome_vec.size() << endl;

#ifdef _OPENMP
    if (buf < omp_get_max_threads()) omp_set_num_threads(buf);
	cerr << "[OpenMP] Number of threads: " << omp_get_max_threads() << endl;
#endif

    printf("iter primal_cost      |w|   d(cost)  Y:Z     Y       Z       training time\n");
    printf("==========================================================================");
    fflush(stdout);

    // C/n
    int n = train_data.size();
    int m = y_train_data.size();
    int n_total = train_data.size() + y_train_data.size() + z_train_data.size();
    // lambda = 1/C
    double lambda = 1.0 / cost;

    double f = 0.0, old_f = 0.0;
    double wscale = 1, old_wscale = 1;
    double obj = 0, diff_obj = 0;
    double dcost = 1;
    double best_acc = 0, test_acc = 0, test_acc_y = 0, test_acc_z = 0;
    int best_iter = 0;
    double t_i = 2;
    double K = 2;

    // init shrink for y_train_data and z_train_data
    opti.resize(y_train_data.size() + z_train_data.size(), 0);

    // initialy, train only joint event
    vector<int> train_data_index;
    for (int i=0; i < train_data.size(); i++) {
        train_data_index.push_back(i);
    }

    for (niter = 1; niter <= iter; niter++) {
        timer t;
        total = correct = skip_count = shrink_count = 0;
        old_f = f;
        f = 0;

        // self-paced learning: mu = 1.2
        if (niter > init_iter+1) K = K / 1.2;

        // after init_iter, train joint event, y_train_data, and z_train_data
        if (niter == init_iter+1) {
            // add y_train_data
            for (int i=0; i < y_train_data.size(); i++) {
                train_data_index.push_back(n+i);
            }
            cerr << " added y_train_data! ";
            // add z_train_data
            for (int i=0; i < z_train_data.size(); i++) {
                train_data_index.push_back(n+m+i);
            }
            cerr << " added z_train_data! ";
        }

        // randome suffle
        random_shuffle(train_data_index.begin(), train_data_index.end());

        for (size_t sent_i = 0; sent_i < train_data_index.size();) {
            make_M_matrix();

			#pragma omp parallel for
            for (int i = 0; i < buf; i++) {
				int skip = 0;
				#pragma omp critical (sent_i)
				if (sent_i++ >= train_data_index.size()) skip = 1;
				if (skip) continue;

                // choose random example
				int r;
				// check one more time
				if (sent_i-1 >= train_data_index.size()) continue;
                r = train_data_index[sent_i-1];
                // initally, train only joint event
                if (niter <= init_iter && r >= n) continue;
                // shrink for y_train_data
                if (r >= n && opti[r-n] >= niter) {
                    #pragma omp atomic
                    shrink_count++;
                    continue;
                }
                sent_t& sent = r < n ? train_data[r] : (r < n+m ? y_train_data[r-n] : z_train_data[r-n-m]);

                // find most violated contraint
				#pragma omp atomic
                argmax_count++;

                timer t_viol;
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);
                vector<int> y_z_seq, y_z_star;
                if (r < n) {
                    // SSVM
                    y_z_seq = SSVM::find_most_violated_constraint(r_vec, sent, wscale);
                } else if (r < n+m) {
                    // latent SSVM for y_train_data
                    y_z_seq = find_most_violated_constraint(r_vec, sent, true, wscale);
                } else {
                    // latent SSVM for z_train_data
                    y_z_seq = find_most_violated_constraint(r_vec, sent, false, wscale);
                }

				#pragma omp atomic
                time_viol += t_viol.elapsed();
    
                // sentence is correct?
                if (r < n) {
                    for (int j=0; j < sent.size(); j++) {
					    #pragma omp atomic
                        total++;
					    if (sent[j].outcome == y_z_seq[j]) {
						    #pragma omp atomic
						    correct++;
					    }
                    }
                }
    
                double cur_loss;
                if (r < n) {
                    cur_loss = SSVM::calculate_loss(sent, y_z_seq);
                } else if (r < n+m) {
                    cur_loss = calculate_latent_loss(sent, y_z_seq, true);
                } else {
                    cur_loss = calculate_latent_loss(sent, y_z_seq, false);
                }
    
                timer t_psi;
                vect_t max_vect;
                if (r < n) {
                    max_vect = SSVM::make_diff_vector(sent, y_z_seq);
                } else if (r < n+m) {
                    y_z_star = find_hidden_variable(r_vec, sent, true);
                    max_vect = make_diff_vector(sent, y_z_star, y_z_seq);
                } else {
                    y_z_star = find_hidden_variable(r_vec, sent, false);
                    max_vect = make_diff_vector(sent, y_z_star, y_z_seq);
                }
				#pragma omp atomic
                time_psi += t_psi.elapsed();
    
                double cost_diff = calculate_cost(max_vect);
                // wscale 반영
                cost_diff *= wscale;
    
                double H_y = cur_loss - cost_diff;

                // self-paced learning for latent train data
                if (use_SPL && r >= n && niter > init_iter) {
				    #pragma omp critical (work_set)
                    if (H_y > 0 && H_y < 1/K) { // self-paced learning
                        // work set
                        work_set.push_back(max_vect);
                        // loss
                        loss.push_back(H_y);
                        // sent_ids
                        sent_ids.push_back(r);
                        // f: hinge loss
                        f += H_y / (double)n_total;
                    }
                    if (H_y >= 1/K) {
                        #pragma omp atomic
                        skip_count++;
                        // shrink
                        if (H_y > 3/K) {
                            opti[r-n] = niter + 5;
                        } else if (H_y > 2/K) {
                            opti[r-n] = niter + 3;
                        }
                        // test
                        /*
				        #pragma omp critical (test)
                        if (H_y > 100) {
                            cerr << endl << " H_y=" << H_y << " latent_sent_i=" << r-n << " sent_size=" << sent.size() << " h_star_size=" << h_star.size() << " y_seq_size=" << y_seq.size() << " (y,{y:h*},{y^:h^})" << endl;
                            for (int j=0; j < sent.size(); j++) {
                                if (joint2visible_map[h_star[j]] != sent[j].outcome) cerr << " *****";
					            cerr << " (" << visible_outcome_vec[sent[j].outcome] << "," << outcome_vec[h_star[j]] << "," << outcome_vec[y_seq[j]] <<")";
                            }
                            cerr << " end" << endl;
                        }
                        */
                    }
                    // shrink for latent_train_data when consecutive correct
                    /*
                    if (cost_diff == 0) {
                        if (opti[r-n] == niter - 1) opti[r-n] = niter + 1;
                        else opti[r-n] = niter;
                    }
                    */
                } else {
                    #pragma omp critical (work_set)
                    if (H_y > 0) {
                        // work set
                        work_set.push_back(max_vect);
                        // loss
                        loss.push_back(H_y);
                        // sent_ids
                        sent_ids.push_back(r);
                        // f: hinge loss
                        f += H_y / (double)n_total;
                    }
                    // shrink for latent_train_data when consecutive correct
                    /*
                    if (r >= n && niter > init_iter) {
                        if (cost_diff == 0) {
                            if (opti[r-n] == niter - 1) opti[r-n] = niter + 1;
                            else opti[r-n] = niter;
                        }
                    }
                    */
                }
            }

            // Stochastic Gradient Decent
            timer t_qp;
            // pegasos
            double s = 1.0 - (1.0 / t_i);
            if (s > 0) {
                old_wscale = wscale;
                wscale *= s;
            }
    
            // update w
            for (int i=0; i < work_set.size(); i++) {
                vect_t& max_vect = work_set[i];
                int sent_id = sent_ids[i];
                for (int j=0; j < max_vect.size(); j++) {
                    double factor = 0;
                    // hinge loss: g = lambda*w - (1/n)sum{delta(psi(i,y))}
                    //if (sent_id < n) factor = (1.0 / wscale) * eta * max_vect[j].factor / buf;
                    //else factor = (1.0 / wscale) * latent_eta * max_vect[j].factor / buf;
                    if (sent_id < n) {
                        factor = (1.0 / wscale) * (cost / t_i) * max_vect[j].factor / buf;
                    } else if (sent_id < n+m) {
                        factor = (1.0 / wscale) * (y_cost / t_i) * max_vect[j].factor / buf;
                    } else {
                        factor = (1.0 / wscale) * (z_cost / t_i) * max_vect[j].factor / buf;
                    }
                    for (int k=0; k < max_vect[j].vect.size(); k++) {
                        int fid = max_vect[j].vect[k].first;
                        double val = max_vect[j].vect[k].second;
                        // test
                        //cerr << "fid=" << fid << " theta=" << theta[fid] <<  " val=" << factor * val << endl;
                        double old_theta = theta[fid];
                        // update theta
                        theta[fid] += factor * val;
                        // update obj : obj는 scale을 뺀 값을 저장
                        obj -= old_theta * old_theta;
                        obj += theta[fid] * theta[fid];
                    }
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
                obj *= wscale * wscale;
                wscale = 1;
                cerr << ".";
            }

            work_set.clear();
            loss.clear();
            sent_ids.clear();
            t_i += 1;
            time_qp += t_qp.elapsed();
        }

        double iter_time = t.elapsed();
        time += iter_time;

        // f
        //f += 0.5 * lambda * (wscale*wscale*obj);
        f += 0.5 * (1.0 / cost) * (wscale*wscale*obj);
        //dcost = (dcost + ABS(old_f - f)/MAX(old_f,f)) / 2.0;
        dcost = ABS(old_f - f)/MAX(old_f,f);

        // continue evaluations
        acc = correct/double(total);

        // test_data accuracy
        correct = correct_y = correct_z = 0;
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
            make_M_matrix();

			#pragma omp parallel for
            for (int i = 0; i < test_data.size(); i++) {
                sent_t& sent = test_data[i];
                //  make R matrix
				vector<double> r_vec;
                make_R_matrix(r_vec, sent);

                double prob;
                vector<int> y_z_seq = viterbi(r_vec, sent, prob);

                for (int j=0; j < sent.size(); j++) {
                    int answer = sent[j].outcome;
                    int output = y_z_seq[j];
					if (output == answer) {
						#pragma omp atomic
						correct++;
					}
					if (joint2y_map[output] == joint2y_map[answer]) {
						#pragma omp atomic
						correct_y++;
					}
					if (joint2z_map[output] == joint2z_map[answer]) {
						#pragma omp atomic
						correct_z++;
					}
                }
            }
        }
        test_acc = test_data.size() > 0 ? 100*double(correct)/double(n_test_event) : 0;
        test_acc_y = test_data.size() > 0 ? 100*double(correct_y)/double(n_test_event) : 0;
        test_acc_z = test_data.size() > 0 ? 100*double(correct_z)/double(n_test_event) : 0;
        if (test_acc > best_acc) {
            best_acc = test_acc;
            best_iter = niter;
            // for best model
            if (niter > init_iter) {
                // scaling
                if (wscale < 1) {
                    cerr << "s";
				    #pragma omp parallel for
                    for (int i=0; i < n_theta; i++) {
                        if (theta[i] != 0) theta[i] *= wscale;
                    }
                    obj *= wscale * wscale;
                    wscale = 1;
                }
                for (int i=0; i < n_theta; i++) best_theta[i] = theta[i];
            }
        }

        printf("\n%4d  %.3e %8.3f %9.6f %6.2f%% %6.2f%% %6.2f%% %6.2f %7.2f ", niter,
            cost*f, sqrt(wscale*wscale*obj), dcost, test_acc, test_acc_y,
            test_acc_z, iter_time, time);

        if (skip_count > 0 || shrink_count > 0) {
            cout << endl << "\t";
            if (skip_count > 0) cout << "1/K=" << 1.0/K << " skip=" << skip_count << " ";
            if (shrink_count > 0) cout << "shrink=" << shrink_count << " ";
        }
        if (correct > 0) print_status();
        fflush(stdout);


        // terminate
        if (niter > init_iter && skip_count == 0 && dcost < threshold) {
            printf("\nTraining terminats succesfully in %.2f seconds\n", time);
            break;
        }

        // save model periodically
        if (period > 0 && niter < iter && (niter % period == 0 || (best_acc == test_acc && niter > period))) {
            // scaling
            if (wscale < 1) {
                cerr << "s";
				#pragma omp parallel for
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) theta[i] *= wscale;
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
        }
    }

    // scaling
    if (wscale < 1) {
        cerr << "s";
		#pragma omp parallel for
        for (int i=0; i < n_theta; i++) {
            if (theta[i] != 0) theta[i] *= wscale;
        }
        obj *= wscale * wscale;
        wscale = 1;
    }

    if (niter > iter) {
        printf("\nMaximum numbers of %d iterations reached in %.2f seconds", iter, time);
    }
    
    cout << endl << "Best acc: " << best_acc << " % at " << best_iter;
    // copy best_theta to theta
    if (best_iter > init_iter) {
        cout << endl << "copy best_theta to theta ... ";
        for (int i=0; i < n_theta; i++) theta[i] = best_theta[i];
        cout << "done.";
    }

    cout << endl << "Runtime(%): SGD=" << 100*time_qp/time << " Argmax=" << 100*time_viol/time;
    cout << " psi=" << 100*time_psi/time << " others=" << 100*(time-time_qp-time_viol-time_psi)/time << endl;
    cout << "Number of calls to 'find_most_violated_constraint': " << argmax_count << endl;

    return time;
}

/// find most violated constraint for latent SSVM.
/// sent is an example of y_train_data or z_train_data
/// argmax_{y,z} {w*F(x_i,y,z} + L(y_i,y)} --> return {y,z} or
/// argmax_{y,z} {w*F(x_i,y,z} + L(z_i,z)} --> return {y,z}
vector<int> Joint_SSVM::find_most_violated_constraint(vector<double>& r_vec, sent_t& sent, bool is_y_train_data, double wscale) {
    vector<vector<int> > psi;
    vector<vector<double> > delta;
    int i, j, k, l, answer;
    double cur, cur_loss = 0;
	int m_vec_size = sent.size() + 1;
    
    for (i=0; i < m_vec_size; i++) {
        if (i < m_vec_size - 1) {
            // answer = y or z
            answer = sent[i].outcome;
        } else {
            answer = default_oid;
        }
        double one_loss = (wscale == 1 ? 1 : 1 / wscale);
        
        vector<int> psi_i;
        vector<double> delta_i;
        
        // for beam search
        vector<int> beam_vec;
        vector<pair<double, int> > prev_delta;
        if (i > 0 && beam > 0) {
            // sorting delta[i-1]
            for (k=0; k < n_outcome; k++) {
                prev_delta.push_back(make_pair(delta[i-1][k], k));
            }
            sort(prev_delta.begin(), prev_delta.end(), greater<pair<double,int> >());
            
            for (l=0; l < n_outcome; l++) {
                beam_vec.push_back(prev_delta[l].second);
                if (l >= beam) {
                    if (verbose && l > beam) {
                        cerr << " beam=" << l << endl;
                    }
                    break;
                }
            }
        }
        
        for (j=0; j < n_outcome; j++) {
            double max = -1e15;
            int max_k = -1;
            
            // loss
            if (i < m_vec_size - 1) {
                //cur_loss = (j != answer ? one_loss : 0);
                if (is_y_train_data) {
                    cur_loss = (joint2y_map[j] != answer ? one_loss : 0);
                } else {
                    cur_loss = (joint2z_map[j] != answer ? one_loss : 0);
                }
            } else {
                cur_loss = 0;
            }
            
            if (i == 0) {
                max = r_vec[MAT2(0,j)] + cur_loss;
                max_k = default_oid;
            } else {
                if (beam == 0) {
                    // dense graph = fully-connected
                    for (k=0; k < n_outcome; k++) {
                        if (i < m_vec_size - 1) {
                            cur = delta[i-1][k] + cur_loss + r_vec[MAT2(i,j)] + m_vec[MAT2(k,j)];
                        } else {
                            cur = delta[i-1][k];
                        }
                        
                        if (k == 0 || cur > max) {
                            max = cur;
                            max_k = k;
                        }
                    }
                } else {
                    // beam search in dense graph
                    for (l=0; l < beam_vec.size(); l++) {
                        k = beam_vec[l];
                        if (i < m_vec_size - 1) {
                            cur = delta[i-1][k] + cur_loss + r_vec[MAT2(i,j)] + m_vec[MAT2(k,j)];
                        } else {
                            cur = delta[i-1][k];
                        }
                        
                        if (l == 0 || cur > max) {
                            max = cur;
                            max_k = k;
                        }
                    }
                }
            }
            
            if (max_k < 0) {
                cerr << "Error: find_most_violated_constraint" << endl;
                exit(1);
            }
            
            delta_i.push_back(max);
            psi_i.push_back(max_k);
        }
        delta.push_back(delta_i);
        psi.push_back(psi_i);
    }
    
    vector<int> y_z_seq;
    int prev_y = default_oid;
    for (i = m_vec_size-1; i >= 1; i--) {
        int y = psi[i][prev_y];
        y_z_seq.push_back(y);
        prev_y = y;
    }
    reverse(y_z_seq.begin(), y_z_seq.end());
    
    // test
    /*
     cerr << endl;
     for (i=0; i < y_z_seq.size(); i++)
     cerr << (*outcome_vec)[y_z_seq[i]] << " ";
     */
    
    return y_z_seq;
}

/// find hidden variable.
/// sent is an example of y_train_data or z_train_data.
/// argmax_{z} {w*f(x_i,y_i,z)} --> return {y_i,z*}
/// argmax_{y} {w*f(x_i,y,z_i)} --> return {y*,z_i}
vector<int> Joint_SSVM::find_hidden_variable(vector<double>& r_vec, sent_t& sent, bool is_y_train_data) {
    vector<vector<int> > psi;
    vector<vector<double> > delta;
    int i, j, k, answer;
    double cur;
	int m_vec_size = sent.size() + 1;
    
    for (i=0; i < m_vec_size; i++) {
        if (i < m_vec_size - 1) {
            // answer = y or z
            answer = sent[i].outcome;
        } else {
            answer = default_oid;
        }
        
        vector<int> psi_i;
        vector<double> delta_i;
        
        for (j=0; j < n_outcome; j++) {
            double max = -1e20;
            int max_k = -1;
            
            if (i == 0) {
                max_k = default_oid;
                max = -1e20;
                if (is_y_train_data) {
                    // j = {y_i,z}, answer = y_i
                    if (joint2y_map[j] == answer) max = r_vec[MAT2(0,j)];
                } else {
                    // j = {y,z_i}, answer = z_i
                    if (joint2z_map[j] == answer) max = r_vec[MAT2(0,j)];
                }
            } else {
                // dense graph = fully-connected
                if (is_y_train_data) {
                    // j = {y_i,z}, answer = y_i
                    if (joint2y_map.find(j) == joint2y_map.end()) {
                        cerr << " find_hidden_variable: joint2y_map: " << outcome_vec[j] << endl;
                        exit(1);
                    }
                } else {
                    // j = {y,z_i}, answer = z_i
                    if (joint2z_map.find(j) == joint2z_map.end()) {
                        cerr << " find_hidden_variable: joint2z_map: " << outcome_vec[j] << endl;
                        exit(1);
                    }
                }
                if (is_y_train_data && joint2y_map[j] != answer) {
                    max = -1e20;
                    max_k = default_oid;
                } else if (!is_y_train_data && joint2z_map[j] != answer) {
                    max = -1e20;
                    max_k = default_oid;
                } else {
                    max_k = default_oid;
				    for (k=0; k < n_outcome; k++) {
                        if (is_y_train_data) {
					        // k = {y,z}, answer = y
					        if (joint2y_map[k] != sent[i-1].outcome) continue;
                        } else {
					        // k = {y,z}, answer = z
					        if (joint2z_map[k] != sent[i-1].outcome) continue;
                        }

					    if (i < m_vec_size - 1) {
						    delta[i-1][k];
						    r_vec[MAT2(i,j)];
						    m_vec[MAT2(k,j)];
						    cur = delta[i-1][k] + r_vec[MAT2(i,j)] + m_vec[MAT2(k,j)];
					    } else {
						    cur = delta[i-1][k];
					    }
                        
					    if (k == 0 || cur > max) {
						    max = cur;
						    max_k = k;
					    }
					}
				}
            }
            
            if (max_k < 0) {
                cerr << "Error: find_hidden_variable" << endl;
                exit(1);
            }
            
            delta_i.push_back(max);
            psi_i.push_back(max_k);
        }
        delta.push_back(delta_i);
        psi.push_back(psi_i);
    }
    
    vector<int> y_z_seq;
    int prev_y = default_oid;
    for (i = m_vec_size-1; i >= 1; i--) {
        int y = psi[i][prev_y];
        y_z_seq.push_back(y);
        prev_y = y;
    }
    reverse(y_z_seq.begin(), y_z_seq.end());
    
    // test
    /*
     cerr << endl;
     for (i=0; i < y_z_seq.size(); i++)
     cerr << (*outcome_vec)[y_z_seq[i]] << " ";
     */
    
    return y_z_seq;
}

/// calculate loss for latent SSVM.
/// L(y_i,y) or L(z_i,z)) instead of L(y_i,z_i,y,z)
double Joint_SSVM::calculate_latent_loss(sent_t& sent, vector<int>& y_z_seq, bool is_y_train_data) {
    int i, answer = 0;
    double incorrect = 0;
    
    for (i=0; i < sent.size(); i++) {
        answer = sent[i].outcome;
        if (is_y_train_data) {
            // answer = y
            if (answer != joint2y_map[y_z_seq[i]]) incorrect += 1;
        } else {
            // answer = z
            if (answer != joint2z_map[y_z_seq[i]]) incorrect += 1;
        }
    }
    return incorrect;
}

/// make_diff_vector: return {f(x_i,y_i,z_i*) - f(x_i,y,z)} or
///                   return {f(x_i,y_i*,z_i) - f(x_i,y,z)}.
/// y_z_star : {y_i,z_i*} or {y_i*,z_i}.
/// y_z_seq : {y,z}
vect_t Joint_SSVM::make_diff_vector(sent_t& sent, vector<int>& y_z_star, vector<int>& y_z_seq) {
    vect_t vect;
    single_vect_t svect;
    map<int, float> vect_map;
    bool same = true;
    
    for (size_t i=0; i < sent.size(); i++) {
        context_t& cont = sent[i].context;
        //int outcome = sent[i].outcome;
        int outcome = y_z_star[i];
        
        if (outcome != y_z_seq[i]) {
            same = false;
        }
        
        // g
        if (outcome != y_z_seq[i]) {
            context_t::iterator cit = cont.begin();
            for (; cit != cont.end(); cit++) {
                int pid = cit->pid;
#ifndef BINARY_FEATURE
                double fval = cit->fval;
#else
                double fval = 1;
#endif

                if (support_feature) {
                    // f(x_i, y_i, z_i*)
                    int fid = make_fid(pid, outcome);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = fval;
                        else vect_map[fid] += fval;
                    }
                    // - f(x_i, y, z)
                    fid = make_fid(pid, y_z_seq[i]);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -fval;
                        else vect_map[fid] -= fval;
                    }
                } else {
                    // f(x_i, y_i, z_i*)
                    int fid = pid * n_outcome + outcome;
                    if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = fval;
                    else vect_map[fid] += fval;
                    // - f(x_i, y, z)
                    fid = fid - outcome + y_z_seq[i];
                    if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -fval;
                    else vect_map[fid] -= fval;
                }
            }
        }
        
        // f
        if (i > 0 && !edge_pid.empty()) {
            if (support_feature) {
                // f(x_i, y_i, z_i*)
                //int y1 = sent[i-1].outcome;
                int y1 = y_z_star[i-1];
                if (y1 < edge_pid.size()) {
                    int pid = edge_pid[y1];
                    int fid = make_fid(pid, outcome);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = 1;
                        else vect_map[fid] += 1;
                    }
                }
                // - f(x_i, y, z)
                y1 = y_z_seq[i-1];
                if (y1 < edge_pid.size()) {
                    int pid = edge_pid[y1];
                    int fid = make_fid(pid, y_z_seq[i]);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -1;
                        else vect_map[fid] -= 1;
                    }
                }
            } else {
                // f(x_i, y_i, z_i*)
                //int y1 = sent[i-1].outcome;
                int y1 = y_z_star[i-1];
                int pid = edge_pid[y1];
                int fid = pid * n_outcome + outcome;
                if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = 1;
                else vect_map[fid] += 1;
                // - f(x_i, y, z)
                y1 = y_z_seq[i-1];
                pid = edge_pid[y1];
                fid = pid * n_outcome + y_z_seq[i];
                if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -1;
                else vect_map[fid] -= 1;
            }
        } // end context
    } // end sentence
    
    if (same) return vect;
    
    double norm = 0.0;
    map<int, float>::iterator it = vect_map.begin();
    for (; it != vect_map.end(); it++) {
        if (it->second != 0) {
            svect.vect.push_back(make_pair(it->first, it->second));
            norm += it->second * it->second;
        }
    }
    
    // |vect|^2
    svect.twonorm_sq = norm;
    svect.factor = 1.0;
    vect.push_back(svect);
    
    return vect;
}


