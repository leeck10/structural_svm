/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
 	@file ssvm_tool.cpp
	@brief (linear chain) Structural SVMs
    @author Changki Lee (leeck@kangwon.ac.kr)
	@date 2013/3/4
*/

#include <cassert>

#if HAVE_GETTIMEOFDAY
    #include <sys/time.h> // for gettimeofday()
#endif

#include <cstdlib>
#include <cassert>
#include <stdexcept> //for std::runtime_error
#include <memory>    //for std::bad_alloc
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "ssvm_cmdline.h"

#include "ssvm.hpp"
#include "latent_ssvm.hpp"
#include "joint_ssvm.hpp"

using namespace std;

int main(int argc,char* argv[]) {
    SSVM *svm;
    string model_file, source_file;

    // time check
    struct tm *t;
    time_t timer;
    time(&timer);
    t = localtime(&timer);
    int year = t->tm_year + 1900;
    int month = t->tm_mon + 1;
    int day = t->tm_mday;
    //printf("%d/%d/%d\n", year, month, day);
    if (year >= 2020 || year <= 2012) {
        printf("%s\n", "본 프로그램의 사용기간이 만료 되었습니다.");
        printf("%s\n", "leeck@kangwon.ac.kr (이창기, 010-3308-0337)로 문의해주시기 바랍니다.");
        exit(1);
    }

    try {
        gengetopt_args_info args_info;

        /* let's call our CMDLINE Parser */
        if (cmdline_parser (argc, argv, &args_info) != 0)
            return EXIT_FAILURE;

        string in_file, out_file, test_file;
        int random = 0;
        // for Joint SSVM
        string y_data, z_data;

        // latent, joint, SSVM
        if (args_info.latent_given || args_info.latent_SPL_given) {
            svm = new Latent_SSVM();
        } else if (args_info.joint_given || args_info.joint_SPL_given) {
            svm = new Joint_SSVM();
        } else {
            svm = new SSVM();
        }

        // hash
        if (args_info.hash_given) {
            svm->hash_feature = 2;
            svm->set_n_pred(args_info.hash_arg);
        }
        // model
        if (args_info.model_given) {
            model_file = args_info.model_arg;
            svm->model_file = model_file;
        }
        // output
        if (args_info.output_given) {
            out_file = args_info.output_arg;
        }
        // succeed
        if (args_info.source_given) {
            source_file = args_info.source_arg;
        }
        // period
        if (args_info.period_given) {
            svm->period = args_info.period_arg;
        }
        // rm_inactive
        if (args_info.rm_inactive_given) {
            svm->rm_inactive = args_info.rm_inactive_arg;
        }
        // buf
        if (args_info.buf_given) {
            svm->buf = args_info.buf_arg;
        }
        // iter
        if (args_info.iter_given) {
            svm->iter = args_info.iter_arg;
        }
        // cost
        if (args_info.cost_given) {
            svm->cost = args_info.cost_arg;
        }
        // threshold
        if (args_info.threshold_given) {
            svm->threshold = args_info.threshold_arg;
        }
        // eps
        if (args_info.epsilon_given) {
            svm->eps = args_info.epsilon_arg;
        }
        // beam
        if (args_info.beam_given) {
            svm->beam = args_info.beam_arg;
        }
        // random
        if (args_info.random_given) {
            random = args_info.random_arg;
        }
        // train_num
        if (args_info.train_num_given) {
            svm->train_num = args_info.train_num_arg;
        }
        // for Joint SSVM
        // y_data
        if (args_info.y_data_given) {
            y_data = args_info.y_data_arg;
        }
        // z_data
        if (args_info.z_data_given) {
            z_data = args_info.z_data_arg;
        }
        // y_cost
        if (args_info.y_cost_given) {
            svm->y_cost = args_info.y_cost_arg;
        }
        // z_cost
        if (args_info.z_cost_given) {
            svm->z_cost = args_info.z_cost_arg;
        }
        // init_iter
        if (args_info.init_iter_given) {
            svm->init_iter = args_info.init_iter_arg;
        }

        svm->verbose = args_info.verbose_flag;
        svm->binary = args_info.binary_flag;
        svm->use_comment = args_info.comment_flag;
        svm->owps_format = args_info.owps_format_flag;
        svm->support_feature = args_info.support_flag;
        svm->general_feature = args_info.general_flag;
        svm->final_opt_check = args_info.final_opt_flag;
        svm->skip_eval = args_info.skip_eval_flag;

        if (args_info.inputs_num > 0) {
            in_file = args_info.inputs[0];
            if (args_info.inputs_num > 1) {
                test_file = args_info.inputs[1];
            }
        } else if (!args_info.show_given && 
                !args_info.convert_given &&
                !args_info.convert2_given &&
                !args_info.convert3_given &&
                !args_info.modify_given) {
            cmdline_parser_print_help();
            return EXIT_FAILURE;
        }

        string estimate = "pegasos";
        if (args_info.fsmo_given)
            estimate = "fsmo";
        if (args_info.fsmo_joint_given)
            estimate = "fsmo_joint";
        if (args_info.fsmo_joint2_given)
            estimate = "fsmo_joint2";
        if (args_info.pegasos_given)
            estimate = "pegasos";
        if (args_info.latent_given)
            estimate = "latent_SSVM";
        if (args_info.latent_SPL_given)
            estimate = "latent_SPL";
        if (args_info.joint_given)
            estimate = "joint_SSVM";
        if (args_info.joint_SPL_given)
            estimate = "joint_SPL";

		if (args_info.predict_given) {
			// predict mode
			if (model_file == "") throw runtime_error("model name not given");

			if (svm->binary) svm->load_bin(model_file);
            else svm->load(model_file);

			cerr << "Loading predicting events from " << in_file << endl;
            svm->load_test_event(in_file);

            // ostream
            ofstream f(out_file.c_str());

			if (svm->owps_format) {
                if (args_info.nbest_arg > 1) {
                    if (f) svm->predict_nbest(f, args_info.nbest_arg);
                    else svm->predict_nbest(cout, args_info.nbest_arg);
                } else {
                    if (f) svm->predict_owps(f);
                    else svm->predict_owps(cout);
                }
			} else {
                if (args_info.nbest_arg > 1) {
                    if (f) svm->predict_nbest(f, args_info.nbest_arg);
                    else svm->predict_nbest(cout, args_info.nbest_arg);
                } else {
                    if (f) svm->predict(f);
                    else svm->predict(cout);
                }
            }
		} else if (args_info.show_given) {
			// show-feature mode
			if (model_file == "") throw runtime_error("model name not given");

			if (svm->binary) svm->load_bin(model_file);
            else svm->load(model_file);

			svm->show_feature();
        } else if (args_info.convert_given) {
			// convert mode: txt to bin or bin to txt (with -b)
			if (model_file == "") throw runtime_error("model name not given");

            if (svm->binary) svm->load_bin(model_file);
            else svm->load(model_file);

            svm->remove_zero_feature(svm->threshold);

			if (svm->binary) svm->save(model_file + ".txt");
            else svm->save_bin(model_file + ".bin");
        } else if (args_info.convert2_given) {
			// convert mode: all feature to support feature
			if (model_file == "") throw runtime_error("model name not given");

            if (svm->binary) svm->load_bin(model_file);
            else svm->load(model_file);

            svm->to_support_feature();
            svm->remove_zero_feature(svm->threshold);

            if (svm->binary) svm->save_bin(model_file + ".support_f.bin");
            else svm->save(model_file + ".support_f.txt");
        } else if (args_info.convert3_given) {
			// convert mode: support feature to all feature
			if (model_file == "") throw runtime_error("model name not given");

            if (svm->binary) svm->load_bin(model_file);
            else svm->load(model_file);

            svm->to_all_feature();
            svm->remove_zero_feature(svm->threshold);

			if (svm->binary) svm->save_bin(model_file + ".all_f.bin");
            else svm->save(model_file + ".all_f.txt");
        } else if (args_info.modify_given) {
			// modify_weight mode: modify feature weights
			if (model_file == "") throw runtime_error("model name not given");

            if (svm->binary) svm->load_bin(model_file);
            else svm->load(model_file);

            // modidfy feature weight
            ifstream f(args_info.modify_arg);
            if (!f) {
                cerr << "Fail to open file: " << args_info.modify_arg << endl;
                exit(1);
            }
            string line;
            while (getline(f, line)) {
                vector<string> tokens;
                svm->tokenize(line, tokens, " \t");
                if (tokens.size() == 3) {
                    string pred = tokens[0];
                    string outcome = tokens[1];
                    float weight = (float)atof(tokens[2].c_str());
                    int success = svm->set_feature_weight(pred, outcome, weight);
                    if (!success) {
                        cerr << "Error: " << line << endl;
                    }
                } else {
                    cerr << "Error: " << line << endl;
                }
            }

			if (svm->binary) svm->save_bin(model_file + ".modified.bin");
            else svm->save(model_file + ".modified.txt");
        } else if (args_info.domain_given) {
            // domain adaptation mode
            if (source_file != "") {
				if (svm->binary) svm->load_bin(source_file);
                else svm->load(source_file);
                svm->incremental = 1;
                svm->domain_adaptation = 1;
			} else {
                cerr << "Prior model not given!" << endl;
                exit(1);
            }

			printf("\nLoading training events from %s\n", in_file.c_str());
            svm->load_event(in_file);

            cerr << "load_event is done!" << endl;

            if (random != 0) {
                srand(random);
                svm->random_shuffle_train_data();
                cerr << "random_shuffle is done!" << endl;
            }

            // test while training
			if (test_file != "") {
			    printf("\nLoading testing events from %s\n", test_file.c_str());
				svm->load_test_event(test_file);
			}

			svm->train(estimate);

			if (model_file != "") {
				cerr << "model saving ... ";
				if (svm->binary) svm->save_bin(model_file);
                else svm->save(model_file);
				cerr << "done." << endl;
			} else {
				cerr << "Warning: model name not given, no model saved" << endl;
			}
        } else if (args_info.joint_given || args_info.joint_SPL_given) {
            // joint SSVM
            int y_train_num = 0, z_train_num = 0, train_num = svm->train_num;
            if (args_info.y_train_num_given) {
                y_train_num = args_info.y_train_num_arg;
            }
            if (args_info.z_train_num_given) {
                z_train_num = args_info.z_train_num_arg;
            }

			printf("\nLoading y events from %s\n", y_data.c_str());
            svm->train_num = y_train_num;
            svm->load_latent_event(y_data, true);
            cerr << "load_latent_event is done!" << endl;

            if (z_data != "") {
			    printf("\nLoading z events from %s\n", z_data.c_str());
                svm->train_num = z_train_num;
                svm->load_latent_event(z_data, false);
                cerr << "load_latent_event is done!" << endl;
            }

			printf("\nLoading joint events from %s\n", in_file.c_str());
            svm->train_num = train_num;
            svm->load_event(in_file);
            cerr << "load_event is done!" << endl;

            if (random != 0) {
                srand(random);
                svm->random_shuffle_train_data();
                cerr << "random_shuffle is done!" << endl;
            }

            // test while training
			if (test_file != "") {
			    printf("\nLoading testing events from %s\n", test_file.c_str());
				svm->load_test_event(test_file);
			}

			svm->train(estimate);

			if (model_file != "") {
				cerr << "model saving ... ";
				if (svm->binary) svm->save_bin(model_file);
                else svm->save(model_file);
				cerr << "done." << endl;
			} else {
				cerr << "Warning: model name not given, no model saved" << endl;
			}
		} else {
            // train mode
			printf("\nLoading training events from %s\n", in_file.c_str());
            svm->load_event(in_file);

            cerr << "load_event is done!" << endl;

            if (random != 0) {
                srand(random);
                svm->random_shuffle_train_data();
                cerr << "random_shuffle is done!" << endl;
            }

            // test while training
			if (test_file != "") {
			    printf("\nLoading testing events from %s\n", test_file.c_str());
				svm->load_test_event(test_file);
			}

			svm->train(estimate);

			if (model_file != "") {
				cerr << "model saving ... ";
				if (svm->binary) svm->save_bin(model_file);
                else svm->save(model_file);
				cerr << "done." << endl;
			} else {
				cerr << "Warning: model name not given, no model saved" << endl;
			}
		}
    } catch (std::bad_alloc& e) {
        cerr << "std::bad_alloc caught: out of memory" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (std::runtime_error& e) {
        cerr << "std::runtime_error caught:" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (std::exception& e) {
        cerr << "std::exception caught:" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "unknown exception caught!" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

