/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
    @file ssvm.cpp
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
#include <string.h>
#include <stdlib.h>

#include "ssvm.hpp"
#include "timer.hpp"
#include "pqueue.hpp"

using namespace std;

SSVM::SSVM() {
#ifdef BINARY_FEATURE
    cerr << "# You cannot use general feature mode!" << endl;
#endif
    // pointer
    theta = NULL;

    edge = "@@@edge="; // 내부적으로만 쓰인다
    n_pred = 0;
    n_theta = 0;
    n_event = 0;
    n_test_event = 0;
    default_oid = 0;

    // make_pid4train
    is_train = 0;

    // parameter
    use_comment = 0;
    owps_format = 0;
    hash_feature = 0;
    support_feature = 0;
    general_feature = 0;
    incremental = 0;
    beam = 0;
    verbose = 0;
    binary = 0;
    skip_eval = 0;
    train_num = 0;
    threshold = 1e-04;

    // pegasos
    iter = 100;
    period = 0;

    // svm
    cost = 1;
    rm_inactive = 50;      // inactive const 제거 조건 (iteration)
    buf = 100;
    eps = 0.01;
    // linear이면 1e-25
    precision = 1e-25;
    // final optimality check for shrinking
    final_opt_check = 0;

    // GRAM size
    gram_size = 100;

    // domain adaptation
    domain_adaptation = 0;

    // Joint SSVM
    y_cost = 1;
    z_cost = 1;
    init_iter = 10;
}

SSVM::~SSVM() {
    clear();
}

void SSVM::clear() {
    n_theta = n_outcome = n_event = n_test_event = 0;

    if (theta != NULL) {
        delete[] theta;
        theta = NULL;
    }

    edge_pid.clear();

    train_data.clear();
    test_data.clear();
    train_data_comment.clear();
    test_data_comment.clear();

    m_vec.clear();

    // ssvm
    alpha.clear();
    alpha_history.clear();
    work_set.clear();
    loss.clear();
    x_norm_vec.clear();
    sent_ids.clear();
    work_set_ids.clear();
    y_seq_vec.clear();
    opti.clear();
    slacks.clear();
    slacks_id.clear();
    cost_diff_vec.clear();

    // gram matrix
    for (size_t i=0; i < gram.size(); i++) {
        gram[i].clear();
    }
    gram.clear();
    gram_size = 100;
}


void SSVM::load(const string model) {
    ifstream f(model.c_str());
    if (!f) {
        cerr << "Fail to open file: " << model << endl;
        exit(1);
    }

    cerr << "loading " << model << " ... ";
    timer t;

    n_theta = 0;

    int count, fid;
    int i;
    string line;

    // check model format
    getline(f, line);
    if (!strncmp(line.c_str(), "#txt,shash", 10)) {
        cerr << "Model format: sparse hash txt model" << endl;
        hash_feature = 2;
    } else if (!strncmp(line.c_str(), "#txt,hash", 9)) {
        cerr << "Model format: hash txt model" << endl;
        hash_feature = 1;
    } else if (!strncmp(line.c_str(), "#txt,ssvm", 9)) {
        cerr << "Model format: ssvm txt model" << endl;
        hash_feature = 0;
    } else if (!strncmp(line.c_str(), "#txt,maxent", 11)) { // 하위호환
        cerr << "Model format: crf txt model" << endl;
        hash_feature = 0;
    } else {
        cerr << "Model format error: not txt model!" << endl;
        char temp[100];
        strncpy(temp, line.c_str(), 9);
        temp[9] = 0;
        cerr << temp << endl;
        exit(1);
    }

    // read context predicates
    getline(f, line);
    n_pred = atoi(line.c_str());
    if (!hash_feature) {
        for (i = 0; i < n_pred; ++i) {
            getline(f, line);
            pred_map[line] = i;
            pred_vec.push_back(line);
        }
        cerr << "(pred_map:" << t.elapsed() << ") ";
    }

    // read outcomes
    getline(f, line);
    count = atoi(line.c_str());
    for (i = 0; i < count; ++i) {
        getline(f, line);
        outcome_map[line] = i;
        outcome_vec.push_back(line);
    }

    // read paramaters (count가 0 이면 all_feature, 아니면 support_feature)
    getline(f, line);
    count = atoi(line.c_str());
    if (count == 0) {
        support_feature = false;
    } else {
        support_feature = true;
        vector<string> tokens;
        fid = 0;
        vector<pair<int, int> > param;
        for (i = 0; i < count; ++i) {
            param.clear();
            getline(f, line);
            int oid;
            tokenize(line, tokens, " \t");

            vector<string>::iterator it = tokens.begin();
            ++it; // skip count which is only used in binary format
            for (; it != tokens.end(); it++) {
                oid = atoi(it->c_str());
                param.push_back(make_pair(oid,fid++));
            }
            params.push_back(param);
        }
    }

    // load theta
    getline(f, line);
    n_theta = atoi(line.c_str());

    if (theta != NULL) {
        delete[] theta;
    }
    theta = new float[n_theta];
    // sparse hash: set zero
    if (hash_feature == 2) {
        for (i = 0; i < n_theta; ++i) theta[i] = 0;
    }

    i = 0;
    unsigned int uint;
    while (getline(f, line)) {
        assert(!line.empty());
        // sparse hash
        if (hash_feature == 2) {
            uint = atoi(line.c_str());
            if (uint < 0 || uint >= n_theta) {
                cerr << "Error: i=" << i << " index=" << uint << " n_theta=" << n_theta << endl;
            }
            getline(f, line);
            theta[uint] = atof(line.c_str());
            i++;
        } else {
            // hash, support, all
            theta[i] = atof(line.c_str());
            i++;
        }
    }
    if (hash_feature != 2) assert(i == n_theta);

    n_outcome = outcome_vec.size();

    make_edge_pid();

    // 소요 시간 출력
    cerr << "(" << t.elapsed() << ") done."  << endl;
}

void SSVM::load_bin(const string model) {
    FILE *f;
    f = fopen(model.c_str(), "rb");

    if (!f) {
        cerr << "Fail to open file: " << model << endl;
        exit(1);
    }

    cerr << "loading " << model << " ... ";
    timer t;

    n_theta = 0;

    int count, fid, len, i, j;
    // 너무 많이 잡으면 win32에서 에러가 남
    char buffer[1024*16];

    // check model format
    fread((void*)&buffer, sizeof("#bin,hash"), 1, f);
    if (!strncmp(buffer, "#bin,shash", sizeof("#bin,hash"))) {
        cerr << "Model format: sparse hash binary model!" << endl;
        hash_feature = 2;
    } else if (!strncmp(buffer, "#bin,hash", sizeof("#bin,hash"))) {
        cerr << "Model format: hash binary model!" << endl;
        hash_feature = 1;
    } else if (!strncmp(buffer, "#bin,ssvm", sizeof("#bin,ssvm"))) {
        cerr << "Model format: ssvm binary model!" << endl;
        hash_feature = 0;
    } else if (!strncmp(buffer, "#bin,maxen", sizeof("#bin,maxe"))) { // 하위호환
        cerr << "Model format: crf binary model!" << endl;
        hash_feature = 0;
    } else {
        cerr << "Model format error: not binary model!" << endl;
        buffer[sizeof("#bin,hash")] = 0;
        cerr << buffer << endl;
        exit(1);
    }

    // read context predicates
    fread((void*)&n_pred, sizeof(n_pred), 1, f);
    if (!hash_feature) {
        for (i = 0; i < n_pred; ++i) {
            fread((void*)&len, sizeof(len), 1, f);
            fread((void*)&buffer, len, 1, f);
            string line(buffer, len);
            pred_map[line] = i;
            pred_vec.push_back(line);
        }
        cerr << "(pred_map:" << t.elapsed() << ") ";
    }

    // read outcomes
    fread((void*)&count, sizeof(count), 1u, f);
    for (i = 0; i < count; ++i) {
        fread((void*)&len, sizeof(len), 1u, f);
        fread((void*)&buffer, len, 1u, f);
        string line(buffer, len);
        outcome_map[line] = i;
        outcome_vec.push_back(line);
    }

    // read paramaters (count가 0 이면 all_feature, 아니면 support_feature)
    fread((void*)&count, sizeof(count), 1u, f);
    if (count == 0) {
        support_feature = false;
    } else {
        support_feature = true;
        fid = 0;
        vector<pair<int, int> > param;
        for (i=0; i < count; i++) {
            param.clear();
            // 현재는 이전 포맷과의 호환성 문제로 oid를 int로 저장
            int count2, oid;
            fread((void*)&count2, sizeof(count2), 1u, f);
            for (j=0; j < count2; j++) {
                fread((void*)&oid, sizeof(oid), 1u, f);
                param.push_back(make_pair(oid,fid++));
            }
            params.push_back(param);
        }
    }

    // load theta
    fread((void*)&n_theta, sizeof(n_theta), 1u, f);

    if (theta != NULL) {
        delete[] theta;
    }
    theta = new float[n_theta];

    float theta_i;
    // sparse hash
    if (hash_feature == 2) {
        // set zero
        for (i = 0; i < n_theta; ++i) theta[i] = 0;
        // read data
        int nonzero_num, index;
        fread((void*)&nonzero_num, sizeof(int), 1u, f);
        for (i = 0; i < nonzero_num; ++i) {
            fread((void*)&index, sizeof(int), 1u, f);
            fread((void*)&theta_i, sizeof(float), 1u, f);
            theta[index] = theta_i;
        }
    } else {
        // hash, support, all
        for (i = 0; i < n_theta; ++i) {
            fread((void*)&theta_i, sizeof(float), 1u, f);
            theta[i] = theta_i;
        }
    }

    fclose(f);

    n_outcome = outcome_vec.size();

    make_edge_pid();

    // 소요 시간 출력
    cerr << "(" << t.elapsed() << ") done."  << endl;
}

void SSVM::save(const string model) {
    FILE *f;
    f = fopen(model.c_str(), "w");

    if (!f) {
        cerr << "Unable to open model file to write: " << model << endl;
        exit(1);
    }

    // todo: write a header section here
    if (hash_feature == 2) {
        fprintf(f, "#txt,shash\n");
    } else if (hash_feature) {
        fprintf(f, "#txt,hash\n");
    } else {
        fprintf(f, "#txt,ssvm\n");
    }

    // n_pred
    fprintf(f, "%d\n", n_pred);
    if (!hash_feature) {
        for (int i = 0; i < n_pred; ++i) {
            fprintf(f, "%s\n", pred_vec[i].c_str());
        }
    }

    fprintf(f, "%d\n", outcome_vec.size());
    for (int i = 0; i < outcome_vec.size(); ++i) {
        fprintf(f, "%s\n", outcome_vec[i].c_str());
    }

    if (support_feature) {
        fprintf(f, "%d\n", params.size());
        for (int i = 0; i < params.size(); ++i) {
            vector<pair<int, int> >& param = params[i];
            fprintf(f, "%d", param.size());
            for (int j = 0; j < param.size(); ++j) {
                fprintf(f, " %d", param[j].first);
            }
            fprintf(f, "\n");
        }
    } else {
        fprintf(f, "0\n");
    }

    // write theta
    fprintf(f, "%d\n", n_theta);
    for (int i = 0; i < n_theta; ++i) {
        // sparse hash
        if (hash_feature == 2) {
            if (theta[i] != 0) {
                fprintf(f, "%d\n", i);
                fprintf(f, "%g\n", theta[i]);
            }
        } else {
            // hash, support, all
            fprintf(f, "%g\n", theta[i]);
        }
    }

    fclose(f);
}

void SSVM::save_bin(const string model) {
    FILE *f;
    f = fopen(model.c_str(), "wb");

    if (!f) {
        cerr << "Unable to open model file to write: " << model << endl;
        exit(1);
    }

    int i, j, uint;

    // todo: write a header section here
    if (hash_feature == 2) {
        // sparse hash
        fwrite((void*)"#bin,shash", sizeof("#bin,hash"), 1u, f);
    } else if (hash_feature) {
        fwrite((void*)"#bin,hash", sizeof("#bin,hash"), 1u, f);
    } else {
        fwrite((void*)"#bin,ssvm", sizeof("#bin,ssvm"), 1u, f);
    }

    // n_pred
    uint = n_pred;
    fwrite((void*)&uint, sizeof(uint), 1u, f);
    if (!hash_feature) {
        for (i = 0; i < n_pred; ++i) {
            uint = pred_vec[i].size();
            fwrite((void*)&uint, sizeof(uint), 1u, f);
            fwrite((void*)pred_vec[i].c_str(), pred_vec[i].size(), 1u, f);
        }
    }

    uint = outcome_vec.size();
    fwrite((void*)&uint, sizeof(uint), 1u, f);
    for (i = 0; i < outcome_vec.size(); ++i) {
        uint = outcome_vec[i].size();
        fwrite((void*)&uint, sizeof(uint), 1u, f);
        fwrite((void*)outcome_vec[i].c_str(), outcome_vec[i].size(), 1u, f);
    }

    if (support_feature) {
        uint = params.size();
        fwrite((void*)&uint, sizeof(uint), 1u, f);
        for (i = 0; i < params.size(); ++i) {
            vector<pair<int, int> >& param = params[i];
            uint = param.size();
            fwrite((void*)&uint, sizeof(uint), 1u, f);
            for (j = 0; j < param.size(); ++j) {
                // 현재는 이전 포맷과의 호환성 문제로 oid를 int로 저장
                uint = param[j].first;
                fwrite((void*)&uint, sizeof(uint), 1u, f);
            }
        }
    } else {
        uint = 0;
        fwrite((void*)&uint, sizeof(uint), 1u, f);
    }

    // write theta
    uint = n_theta;
    fwrite((void*)&uint, sizeof(uint), 1u, f);
    float theta_i;
    // sparse hash
    if (hash_feature == 2) {
        uint = 0;
        for (i = 0; i < n_theta; ++i) if (theta[i] != 0) uint++;
        fwrite((void*)&uint, sizeof(uint), 1u, f);
        for (i = 0; i < n_theta; ++i) {
            if (theta[i] != 0) {
                uint = i;
                fwrite((void*)&uint, sizeof(uint), 1u, f);
                theta_i = theta[i];
                fwrite((void*)&theta_i, sizeof(float), 1u, f);
            }
        }
    } else {
        // hash, support, all
        for (i = 0; i < n_theta; ++i) {
            theta_i = theta[i];
            fwrite((void*)&theta_i, sizeof(float), 1u, f);
        }
    }

    fclose(f);
}


// load event and make param
void SSVM::load_event(const string file) {
    string line, comment;
    int i, j, count = 0;
    int old_n_theta = n_theta;
    int old_pred_vec_size = pred_vec.size();

    ifstream f(file.c_str());
    if (!f) {
        cerr << "Can not open data file to read: " << file << endl;
        exit(1);
    }

    if (hash_feature && support_feature) {
        cerr << "Hash + Support feature mode is not avalable!" << endl;
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
            // comment 처리
            if (use_comment) {
                train_data_comment.push_back(comment);
                comment = "";
            }
            if (!owps_format) {
                train_data.push_back(sent);
                sent.clear();
                if (train_num > 0 && train_num <= train_data.size()) {
                    break;
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
            if (outcome_map.find(*it) == outcome_map.end()) {
                //cerr << *it << " ";
                oid = outcome_vec.size();
                outcome_map[*it] = oid;
                outcome_vec.push_back(*it);
                if (verbose) cerr << "New outcome: " << *it << endl;
            } else {
                oid = outcome_map[*it];
                if (outcome_vec[oid] != *it) {
                    cerr << "oid=" << oid << " : " << outcome_vec[oid] << " != " << *it << endl;
                    exit(1);
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

                // support_feature인 경우 params 업데이트
                if (support_feature) {
                    if (new_pid) {
                        vector<pair<int, int> > param;
                        int fid = n_theta;
                        n_theta++;
                        param.push_back(make_pair(oid, fid));
                        params.push_back(param);
                    } else {
                        vector<pair<int, int> >& param = params[pid];
                        for (i=0; i < param.size(); i++) {
                            if (param[i].first == oid) {
                                break;
                            }
                        }
                        if (i == param.size()) {
                            int fid = n_theta;
                            n_theta++;
                            param.push_back(make_pair(oid, fid));
                            // sort
                            sort(param.begin(), param.end());
                        }
                    }
                }
            }

            node.context = cont;
            sent.push_back(node);

            if (owps_format) {
                train_data.push_back(sent);
                sent.clear();
                // comment 처리
                if (use_comment) {
                    train_data_comment.push_back(comment);
                    comment = "";
                }
                if (train_num > 0 && train_num <= train_data.size()) {
            	    ++count;
                    break;
                }
            }

            ++count;
            if (count % 10000 == 0) {
                cerr << ".";
                if (count % 100000 == 0) cerr << " ";
                if (count % 500000 == 0) cerr << "\t" << count << " " << train_data.size() << endl;
            }
        }
    }
    if (!sent.empty()) {
        train_data.push_back(sent);
        // comment 처리
        if (use_comment) {
            train_data_comment.push_back(comment);
            comment = "";
        }
    }

    n_event += count;

    if (support_feature && !incremental) {
        // fid sorting
        int fid = 0;
        for (i = 0; i < params.size(); ++i) {
            vector<pair<int, int> >& param = params[i];
            for (j = 0; j < param.size(); ++j) {
                param[j].second = fid;
                fid++;
            }
        }
    }

    int old_n_outcome = n_outcome;
    n_outcome = outcome_vec.size();

    // hash
    if (hash_feature) {
        // support only all feature
        n_theta = n_pred * n_outcome;
        if (theta != NULL) delete[] theta;
        // alloc memory
        cerr << endl << "theta allocated: " << n_theta << " ... ";
        theta = new float[n_theta];
        cerr << "Done." << endl;
        // init
        for (int i=0; i < n_theta; i++) {
            theta[i] = 0.0;
        }
        return;
    }

    n_pred = pred_vec.size();

    // not incremental mode
    if (!incremental) {
        if (!support_feature) {
            n_theta = pred_vec.size() * n_outcome;
        }

        if (theta != NULL) delete[] theta;
        // alloc memory
        cerr << endl << "theta allocated: " << n_theta << " ... ";
        theta = new float[n_theta];
        cerr << "Done." << endl;
        // init
        for (i=0; i < n_theta; i++) {
            theta[i] = 0.0;
        }
    } else {
        // incremental인 경우
        if (support_feature) {
            // 새로 outcome이 추가된 경우 --> add_edge에서 처리
            // alloc memory
            cerr << endl << "theta allocated: " << n_theta << " ... ";
            float *new_theta = new float[n_theta];
            cerr << "Done." << endl;

            // copy previous data & fid sorting
            for (i=0; i < n_theta; i++) {
                new_theta[i] = 0;
            }
            int fid = 0;
            for (i = 0; i < old_pred_vec_size; ++i) {
                vector<pair<int, int> >& param = params[i];
                for (j = 0; j < param.size(); ++j) {
                    int outcome = param[j].first;
                    int old_fid = param[j].second;
                    param[j].second = fid;
                    // copy previous data
                    if (old_fid < old_n_theta) {
                        new_theta[fid] = theta[old_fid];
                    } else {
                        //cerr << " new fid:" << old_fid;
                    }
                    fid++;
                }
            }

            if (theta != NULL) delete[] theta;
            theta = new_theta;
        } else {
            // all_feature인 경우
            int new_n_theta = pred_vec.size() * n_outcome;

            // alloc memory
            cerr << endl << "theta allocated: " << new_n_theta << " ... ";
            float *new_theta = new float[new_n_theta];
            cerr << "Done." << endl;

            // copy previous data
            if (n_outcome == old_n_outcome) {
                for (i=0; i < n_theta; i++) {
                    new_theta[i] = theta[i];
                }
                for (i=n_theta; i < new_n_theta; i++) {
                    new_theta[i] = 0;
                }
            } else {
                // 새로 outcome이 추가된 경우 --> add_edge에서 처리
                for (i=0; i < new_n_theta; i++) {
                    new_theta[i] = 0;
                }
                for (i=0; i < old_pred_vec_size; i++) {
                    int old_fid = i * old_n_outcome;
                    int fid = i * n_outcome;
                    for (j = 0; j < old_n_outcome; j++) {
                        new_theta[fid + j] = theta[old_fid + j];
                    }
                }
            }
            n_theta = new_n_theta;

            if (theta != NULL) delete[] theta;
            theta = new_theta;
        }
    }
}


void SSVM::load_test_event(const string file) {
    string line, comment;
    int count = 0;

    ifstream f(file.c_str());
    if (!f) {
        cerr << "Can not open data file to read: " << file << endl;
        exit(1);
    }

    sent_t sent;
    context_t cont;

    test_data.clear();

    while (getline(f, line)) {
        // remove newline for windows format file
        string find_str = "\r";
        string::size_type find_pos = line.find(find_str);
        if (string::npos != find_pos) {
            line.replace(find_pos, find_str.size(), "");
        }
        if (line.empty()) {
            if (!owps_format) {
                test_data.push_back(sent);
                sent.clear();
            }
            // comment 처리
            if (use_comment) {
                test_data_comment.push_back(comment);
                comment = "";
            }
        } else if (line[0] == '#') {
            // comment 처리 안함
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

            int oid;
            if (outcome_map.find(*it) == outcome_map.end()) {
                //cerr << *it << " ";
                oid = outcome_vec.size();
                //exit(1);
            } else {
                oid = outcome_map[*it];
                if (outcome_vec[oid] != *it) {
                    cerr << "oid=" << oid << " : " << outcome_vec[oid] << " != " << *it << endl;
                }
            }

            node_t node;
            node.outcome = oid;
            node.start = sent.size();
            node.end = sent.size();

            ++it;
            cont.clear();
            for (; it != tokens.end();) {
                string fi(it->c_str()); ++it;

                // qid:n.m 처리
                if (fi.find("qid:") != string::npos) {
                    vector<string> str_vec, str_vec2;
                    split(fi, str_vec, ":");
                    split(str_vec[1], str_vec2, ".");
                    node.start = atoi(str_vec2[0].c_str());
                    node.end = atoi(str_vec2[1].c_str());
                    continue;
                }

                feature_t feature;
#ifndef BINARY_FEATURE
                feature.fval = 1.0;
                if (general_feature && fi.find(":") != string::npos) {
                    vector<string> str_vec;
                    split(fi, str_vec, ":");
                    fi = str_vec[0];
                    feature.fval = atof(str_vec[1].c_str());
                }
#endif

                // hash
                if (hash_feature) {
                    feature.pid = hash(fi) % n_pred;
                    cont.push_back(feature);
                }
                else if (pred_map.find(fi) != pred_map.end()) {
                    feature.pid = pred_map[fi];
                    cont.push_back(feature);
                }
            }

            node.context = cont;
            sent.push_back(node);

            if (owps_format) {
                test_data.push_back(sent);
                sent.clear();
                // comment 처리
                if (use_comment) {
                    test_data_comment.push_back(comment);
                    comment = "";
                }
            }

            ++count;
            if (count % 10000 == 0) {
                cerr << ".";
                if (count % 100000 == 0)
                    cerr << " ";
                if (count % 500000 == 0)
                    cerr << "\t" << count << endl;
            }
        }
    }
    if (!sent.empty()) {
        test_data.push_back(sent);
        // comment 처리
        if (use_comment) {
            test_data_comment.push_back(comment);
            comment = "";
        }
    }

    n_test_event = count;
}


// log scale 사용
int SSVM::predict(ostream& f) {
    vector<sent_t>::iterator it = test_data.begin();
    int correct = 0;
    int total = 0;
    int sent_i = 0;

    // 처음에 반드시 실행
    make_M_matrix();

    // 파일에 쓰기
    f << "# output answer" << endl;

    timer t;

    for (; it != test_data.end(); ++it) {
        sent_t& sent = *it;
        vector<string> label;
        double y_seq_prob = eval(sent, label);

        // comment
        if (use_comment) {
            f << test_data_comment[sent_i++] << endl;
        }

        for (size_t i=0; i < sent.size(); i++) {
            int ans = sent[i].outcome;
            f << label[i] << "\t" << outcome_vec[ans] << endl;
            total++;
            if (outcome_vec[ans] == label[i]) {
                correct++;
            }
        }
        f << endl;
    }

    cout << "Accuracy: " << 100.0 * correct / total << "%\t("
        << correct << "/" << total << ")" << endl;

    // 소요 시간 출력
    cout << t.elapsed() << " sec, " << n_test_event / t.elapsed() << " tokens per sec (" << n_test_event << " / " << t.elapsed() << ")" << endl;

    return correct;
}

///< n-best
int SSVM::predict_nbest(ostream& f, int nbest) {
    f << "# 문장정보, 정답열, N-best" << endl;
    f << "# loss prob label ..." << endl;

    vector<sent_t>::iterator it = test_data.begin();
    int i, j, k;
    int correct = 0;
    int total = 0;
    int sent_i = 0;

    make_M_matrix();

    for (; it != test_data.end(); ++it) {
        sent_t& sent = *it;
        vector<vector<string> > label;
        vector<double> y_seq_prob = eval_nbest(sent, label);

        // comment
        if (use_comment) {
            f << test_data_comment[sent_i++] << endl;
        }

        // 정답열 출력
        int outcome = 0;
        f << "0";
        for (k=0; k < sent.size(); k++) {
            outcome = sent[k].outcome;
            if (outcome < n_outcome) {
                f << " " << outcome_vec[outcome];
            } else {
                f << " **UNK**";
            }
        }
        f << endl;

        // N-best 출력
        for (i=0; i < nbest && i < label.size(); i++) {
            int loss = 0;
            for (j=0; j < sent.size(); j++) {
                outcome = sent[j].outcome;
                if (outcome_vec[outcome] != label[i][j]) loss++;
            }
            f << loss << " " << y_seq_prob[i];

            for (j=0; j < sent.size(); j++) {
                if (i == 0) {
                    outcome = sent[j].outcome;
                    total++;
                    if (outcome_vec[outcome] == label[i][j]) correct++;
                }
                f << " " << label[i][j];
            }
            f << endl;
        }
        f << endl;
    }
    cout << "Accuracy: " << 100.0 * correct / total << "%\t("
        << correct << "/" << total << ")" << endl;
    return correct;
}

// for OWPS format
int SSVM::predict_owps(ostream& f) {
    vector<sent_t>::iterator it = test_data.begin();
    int correct = 0;
    int total = 0;
    int sent_i = 0;

    timer t;

    for (; it != test_data.end(); ++it) {
        sent_t& sent = *it;
        string label;
        double y_seq_prob = eval_owps(sent, label);

        // comment
        if (use_comment) {
            f << test_data_comment[sent_i++] << endl;
        }
        for (size_t i=0; i < sent.size(); i++) {
            if (i >= 1) {
                cerr << "OWPS Error: sent_size != 1 : " << sent.size() << endl;
                exit(1);
            }
            int ans = sent[i].outcome;
            if (ans >= 0 && ans < n_outcome) {
                f << label << "\t" << outcome_vec[ans] << endl;
            } else {
                f << label << "\t" << "**UNK**" << endl;
            }
            total++;
            if (outcome_vec[ans] == label) {
                correct++;
            }
        }
        f << endl;
    }
    cout << "Accuracy: " << 100.0 * correct / total << "%\t(" << correct << "/" << total << ")" << endl;
    cout << t.elapsed() << " sec, " << n_test_event / t.elapsed() << " tokens per sec (" << n_test_event << " / " << t.elapsed() << ")" << endl;

    return correct;
}


// make_sent: string feature
void SSVM::make_sent(const vector<vector<string> >& cont_seq, sent_t &sent) {
    int i, j;

    sent.clear();
    for (i=0; i < cont_seq.size(); i++) {
        context_t cont;
        const vector<string>& context = cont_seq[i];
        for (j=0; j < context.size(); j++) {
            string fi = context[j];
            feature_t feature;
#ifndef BINARY_FEATURE
            feature.fval = 1.0;
            if (general_feature && fi.find(":") != string::npos) {
                vector<string> str_vec;
                split(fi, str_vec, ":");
                fi = str_vec[0];
                feature.fval = atof(str_vec[1].c_str());
            }
#endif

            // hash
            if (hash_feature) {
                feature.pid = hash(fi) % n_pred;
                cont.push_back(feature);
            }
            else if (pred_map.find(fi) != pred_map.end()) {
                feature.pid = pred_map[fi];
                cont.push_back(feature);
            }
        }

        node_t node;
        node.outcome = default_oid;
        node.start = sent.size();
        node.end = sent.size();
        node.context = cont;
        sent.push_back(node);
    }
}

// make_sent: pid 사용, binary feature
void SSVM::make_sent(const vector<vector<int> >& cont_seq, sent_t &sent) {
    int i, j;
    feature_t feature;
#ifndef BINARY_FEATURE
    feature.fval = 1.0;
#endif
    context_t cont;

    sent.clear();
    for (i=0; i < cont_seq.size(); i++) {
        cont.clear();
        const vector<int>& context = cont_seq[i];
        for (j=0; j < context.size(); j++) {
            int pid = context[j];
            if (pid >= 0) {
                feature.pid = pid;
                cont.push_back(feature);
            }
        }

        node_t node;
        node.outcome = default_oid;
        node.start = sent.size();
        node.end = sent.size();
        node.context = cont;
        sent.push_back(node);
    }
}

// make_sent: pid 사용, general feature
void SSVM::make_sent(const vector<vector<pair<int,float> > >& cont_seq, sent_t &sent) {
    int i, j;
    feature_t feature;
#ifndef BINARY_FEATURE
    feature.fval = 1.0;
#endif
    context_t cont;

    sent.clear();
    for (i=0; i < cont_seq.size(); i++) {
        cont.clear();
        const vector<pair<int,float> >& context = cont_seq[i];
        for (j=0; j < context.size(); j++) {
            int pid = context[j].first;
            float fval = context[j].second;
            if (pid >= 0) {
                feature.pid = pid;
#ifndef BINARY_FEATURE
                feature.fval = fval;
#endif
                cont.push_back(feature);
            }
        }

        node_t node;
        node.outcome = default_oid;
        node.start = sent.size();
        node.end = sent.size();
        node.context = cont;
        sent.push_back(node);
    }
}


// for prediction
double SSVM::eval(sent_t& sent, vector<string>& label) {
    int i;
    vector<double> r_vec; // for openMP

    // make M_i matrix : log scale
    if (m_vec.empty()) make_M_matrix();
    make_R_matrix(r_vec, sent);
    double y_seq_prob;
    // log scale
    vector<int> y_seq = viterbi(r_vec, sent, y_seq_prob);

    // make label
    label.clear();
    for (i=0; i < y_seq.size(); i++) {
        int y = y_seq[i];
        label.push_back(outcome_vec[y]);
    }
    return y_seq_prob;
}

// for prediction with constraint
double SSVM::eval_with_constraint(sent_t& sent, const vector<string>& constraint, vector<string>& label) {
    int i;
    vector<double> r_vec; // for openMP

    // make M_i matrix : log scale
	if (m_vec.empty()) make_M_matrix();
    make_R_matrix(r_vec, sent);
    // constraint 처리
    constrain_R_matrix(r_vec, constraint);

    double y_seq_prob;
    // log scale
    vector<int> y_seq = viterbi(r_vec, sent, y_seq_prob);

    // make label
    label.clear();
    for (i=0; i < y_seq.size(); i++) {
        int y = y_seq[i];
        label.push_back(outcome_vec[y]);
    }
    return y_seq_prob;
}

// for Korean word segmentation: score - weight*loss
double SSVM::eval_with_loss(sent_t& sent, double weight, vector<string>& input_label, vector<string>& label) {
    int i;
    // sent[i].output에 입력 label을 복원함
    vector<int> input_y_seq;
    string cur_label, prev_label="I";
    for (i=0; i < sent.size(); i++) {
        cur_label = prev_label + ":" + input_label[i];
        if (outcome_map.find(cur_label) != outcome_map.end()) {
            sent[i].outcome = outcome_map[cur_label];
        } else {
            cerr << "outcome_map(unknown_key): " << cur_label << endl;
        }
        input_y_seq.push_back(sent[i].outcome);
        prev_label = input_label[i];
    }

    vector<double> r_vec; // for openMP

    // make M_i matrix : log scale
	if (m_vec.empty()) make_M_matrix();
    make_R_matrix(r_vec, sent);
    double wscale = -1.0 / weight;
    vector<int> y_seq = find_most_violated_constraint(r_vec, sent, wscale);

    // make label
    label.clear();
    for (i=0; i < y_seq.size(); i++) {
        int y = y_seq[i];
        label.push_back(outcome_vec[y]);
    }

    // prob
    double score = 0;
    int y, prev_y = default_oid;
    for (int i=0; i < sent.size(); i++) {
        y = y_seq[i];
        if (i == 0) {
            score += r_vec[MAT2(i,y)];
        } else {
            score += r_vec[MAT2(i,y)] + m_vec[MAT2(prev_y,y)];
        }
        prev_y = y;
        // score - weight*loss
        if (input_y_seq[i] != y) score -= weight;
    }
    //cerr << score << endl;
    return score;
}

// n-best prediction
vector<double> SSVM::eval_nbest(sent_t& sent, vector<vector<string> >& label, int n)
{
    int i, j;
    vector<double> r_vec; // for openMP

    // make M_i matrix : log scale
    if (m_vec.empty()) make_M_matrix();
    make_R_matrix(r_vec, sent);
    vector<double> y_seq_prob;
    // log scale
    vector<vector<int> > y_seq = viterbi_nbest(r_vec, sent, y_seq_prob);

    // make label
    label.clear();
    for (i=0; i < y_seq.size(); i++) {
        vector<string> label_i;
        for (j=0; j < y_seq[i].size(); j++) {
            label_i.push_back(outcome_vec[y_seq[i][j]]);
        }
        label.push_back(label_i);
    }
    return y_seq_prob;
}

///< for OWPS
double SSVM::eval_owps(sent_t& sent, string& label) {
    vector<double> r_vec; // for openMP

    // make M_i matrix
    make_M_matrix4owps(r_vec, sent);
    double y_seq_prob;
    vector<int> y_seq = viterbi4owps(r_vec, sent, y_seq_prob);

    // make label
    label = outcome_vec[y_seq[0]];
    return (double)y_seq_prob;
}

///< for OWPS: 모든 output의 score를 구함 (확률값이 아님)
vector<double> SSVM::eval_owps_all(sent_t& sent) {
    vector<double> r_vec; // for openMP

    // r_vec은 반드시 필요
    r_vec.resize(n_outcome);
	for (int i=0; i < n_outcome; i++) {
        r_vec[i] = 0.0;
	}
    context_t& cont = sent[0].context;
    // g
    for (int j=0; j < cont.size(); j++) {
        int pid = cont[j].pid;
#ifndef BINARY_FEATURE
        double fval = cont[j].fval;
#else
        double fval = 1;
#endif
        if (support_feature) {
            vector<pair<int, int> >& param = params[pid];
            for (int k=0; k < param.size(); ++k) {
                int y2 = param[k].first;
                int fid = param[k].second;
                r_vec[y2] += theta[fid] * fval;
            }
        } else {
            for (int y2=0; y2 < n_outcome; y2++) {
                int fid = make_fid(pid, y2);
                r_vec[y2] += theta[fid] * fval;
            }
        }
    }

    return r_vec;
}

///< for NHN parser: cont is the vector of pid
vector<double> SSVM::eval_owps_all(const vector<int>& cont) {
    vector<double> r_vec; // for openMP

    // r_vec은 반드시 필요
    r_vec.resize(n_outcome);
	for (int i=0; i < n_outcome; i++) {
        r_vec[i] = 0.0;
	}
    // g
    for (int j=0; j < cont.size(); j++) {
        int pid = cont[j];
        if (pid < 0) continue;
        if (support_feature) {
            vector<pair<int, int> >& param = params[pid];
            for (int k=0; k < param.size(); ++k) {
                int y2 = param[k].first;
                int fid = param[k].second;
                if (fid < 0) continue;
                r_vec[y2] += theta[fid];
            }
        } else {
            for (int y2=0; y2 < n_outcome; y2++) {
                int fid = make_fid(pid, y2);
                if (fid < 0) continue;
                r_vec[y2] += theta[fid];
            }
        }
    }

    return r_vec;
}


///< for previous version CRF tool 
double SSVM::eval(const vector<vector<string> >& cont_seq, vector<string>& label) {
    // make sent
    sent_t sent;
    make_sent(cont_seq, sent);

    return eval(sent, label);
}

vector<double> SSVM::eval_nbest(const vector<vector<string> >& cont_seq, vector<vector<string> >& label, int n) {
    // make sent
    sent_t sent;
    make_sent(cont_seq, sent);

    return eval_nbest(sent, label, n);
}

double SSVM::eval_owps(const vector<string>& cont, string& label) {
    vector<vector<string> > cont_seq;
    cont_seq.push_back(cont);
    // make sent
    sent_t sent;
    make_sent(cont_seq, sent);

    return eval_owps(sent, label);
}

vector<double> SSVM::eval_owps_all(const vector<string>& cont) {
    vector<vector<string> > cont_seq;
    cont_seq.push_back(cont);
    // make sent
    sent_t sent;
    make_sent(cont_seq, sent);

    return eval_owps_all(sent);
}


// 최초에 호출
void SSVM::add_edge() {
    if (!hash_feature) {
        int old_n_theta = n_theta;
        int i, j;
        
        if (verbose) cerr << endl;
        for (i=0; i < outcome_vec.size(); i++) {
            if (verbose) cerr << outcome_vec[i] << " ";
            // update pred_map, pred_vec
            string fi = edge + outcome_vec[i];
            if (pred_map.find(fi) != pred_map.end()) continue;
            int pid = pred_vec.size();
            pred_map[fi] = pid;
            pred_vec.push_back(fi);
            n_pred = pred_vec.size();
            // update n_theta for all_feature
            if (!support_feature) {
                n_theta += n_outcome;
            } else {
                // support feature
                vector<pair<int, int> > new_param;
                for (j=0; j < outcome_vec.size(); j++) {
                    // update param and n_theta for support_feature
                    // incremental시 기존 param과 통합 필요
                    int fid = n_theta;
                    n_theta++;
                    new_param.push_back(make_pair(j, fid));
                }
                params.push_back(new_param);
            }
            if (verbose) cerr << endl;
        }
        
        // update n_theta, theta
        float *new_theta = new float[n_theta];
        if (new_theta == NULL) {
            cerr << "Error: add_edge: memory alloc fail!" << endl;
            exit(1);
        }
        
        for (i=0; i < old_n_theta ; i++) {
            new_theta[i] = theta[i];
        }
        for (i=old_n_theta; i < n_theta ; i++) {
            new_theta[i] = 0.0;
        }
        
        if (theta != NULL) {
            delete[] theta;
        }
        theta = new_theta;
    }
    
    make_edge_pid();
}


// make edge_pid
void SSVM::make_edge_pid() {
    edge_pid.clear();
    
    if (verbose) {
        cerr << "make_edge_pid:" << endl;
    }
    
    for (int y1=0; y1 < n_outcome; y1++) {
        string fi = edge + outcome_vec[y1];
        // hash
        if (hash_feature) {
            int pid = hash(fi) % n_pred;
            edge_pid.push_back(pid);
            if (verbose) cerr << fi << ":" << pid << endl;
        }
        else if (pred_map.find(fi) != pred_map.end()) {
            int pid = pred_map[fi];
            edge_pid.push_back(pid);
            if (verbose) cerr << fi << ":" << pid << endl;
        }
        else if (verbose) {
            cerr << "Warning: make_edge_pid: " << fi << " does not exist in pred_map!" << endl;
        }
    }
}


// hash function
unsigned int SSVM::hash(const string key) {
    unsigned int i,len, ch, result = 5381;
	//result = 0;
    
	len = key.length();
	for (i=0; i < len; i++) {
        ch = (unsigned int)key[i];
        //result = ((result<< 5) + result) + ch; // hash * 33 + ch --> collision rate 4.12% : djb2
        //result = (result*31) + ch; // hash * 31 + ch --> collision rate 4.98%
        //result = ((result<< 5) + result) ^ ch; // collision rate 2.81%
        result = ch + (result<< 6) + (result<<16) - result; // collision rate 0.0286% : sdbm
        //result += ch; //  collision rate ? % : lose lose --> too slow
	}
    
	return (unsigned int)result;
}


// fid로 convert
int SSVM::make_fid(int pid, int oid) {
    if (pid < 0) {
        cerr << "Error: make_fid: pid=" << pid << " oid=" << oid << endl;
        return -1;
    }
    
    if (hash_feature || !support_feature) {
        // hash or all feature
        return pid * n_outcome + oid;
    } else {
        // support feature
        vector<pair<int, int> >& param = params[pid];
        // binary search
        for (size_t i=0, j=param.size(), k=0; i < j;) {
            k = (i + j) / 2;
            if (param[k].first == oid) {
                return param[k].second;
            } else if (param[k].first > oid) {
                j = k;
            } else {
                i = k + 1;
            }
        }
        // sequential search
        /*
         for (int i=0; i < param.size(); i++) {
         if (param[i].first == oid) {
         return param[i].second;
         }
         else if (param[i].first > oid) {
         return -1;
         }
         }
         */
        // 존재하지 않을 경우
        //cerr << "Error: make_fid: pid=" << pid << " oid=" << oid << endl;
        return -1;
    }
}


// fully-connected
void SSVM::make_M_matrix() {
    m_vec.clear();
	m_vec.resize(n_outcome * n_outcome);
	for (int i=0; i < n_outcome * n_outcome; i++) {
        m_vec[i] = 0.0;
	}

    // f
    for (int y1=0; y1 < n_outcome; y1++) {
        // for ME
        if (edge_pid.empty() || edge_pid.size() < n_outcome)
            continue;

        int pid = edge_pid[y1];

        if (support_feature) {
            vector<pair<int, int> >& param = params[pid];
            for (size_t j=0; j < param.size(); ++j) { 
                int y2 = param[j].first;
                int fid = param[j].second;
                m_vec[MAT2(y1,y2)] += theta[fid];
                // test
                //cerr << fid << ":" << (*outcome_vec)[y1] << ":" << (*outcome_vec)[y2] << "=" << m_vec[MAT2(y1,y2)] << endl;
            }
        } else {
            // all_feature
            int fid = pid * n_outcome;
            for (int y2=0; y2 < n_outcome; y2++) { 
                //int fid = make_fid(pid, y2);
                //int fid = pid * n_outcome + y2;
                m_vec[MAT2(y1,y2)] += theta[fid + y2];
                // test
                //cerr << fid << ":" << (*outcome_vec)[y1] << ":" << (*outcome_vec)[y2] << "=" << m_vec[MAT2(y1,y2)] << endl;
            }
        }
    }
}


void SSVM::make_R_matrix(vector<double>& r_vec, sent_t& sent) {
    register int i, j, k;
    int m_vec_size = sent.size() + 1;

	r_vec.clear();
    r_vec.resize(m_vec_size * n_outcome);
	for (i=0; i < m_vec_size * n_outcome; i++) {r_vec[i] = 0.0;}

    for (i=0; i < m_vec_size - 1; i++) {
        context_t& cont = sent[i].context;
        // g
        for (j=0; j < cont.size(); j++) {
            int pid = cont[j].pid;

            if (support_feature) {
                vector<pair<int, int> >& param = params[pid];
                for (k=0; k < param.size(); ++k) {
                    int y2 = param[k].first;
                    int fid = param[k].second;
#ifndef BINARY_FEATURE
                    r_vec[MAT2(i,y2)] += theta[fid] * cont[j].fval;
#else
                    r_vec[MAT2(i,y2)] += theta[fid];
#endif
                    // test
                    //cerr << fid << "=" << theta[fid] * fval << " ";
                }
            } else {
                // all feature
                int fid = pid * n_outcome;
                for (k=0; k < n_outcome; k++) {
                    //int fid = make_fid(pid, k);
                    //int fid = pid * n_outcome + k;
#ifndef BINARY_FEATURE
                    r_vec[MAT2(i,k)] += theta[fid + k] * cont[j].fval;
#else
                    r_vec[MAT2(i,k)] += theta[fid + k];
#endif
                    // test
                    //cerr << fid << "=" << theta[fid] * fval << " ";
                }
            }
        }
    }
}


// for OWPS
void SSVM::make_M_matrix4owps(vector<double>& r_vec, sent_t& sent) {
    int i, j, k;
    int m_vec_size = sent.size() + 1;

    // m_vec은 n-best search때에 viterbi_nbest 함수를 사용하기 위해서
	if (m_vec.empty()) {
		m_vec.clear();
		m_vec.resize(n_outcome * n_outcome);
		for (i=0; i < n_outcome * n_outcome; i++) {
			m_vec[i] =1;
		}
	}

    // r_vec은 반드시 필요
    r_vec.clear();
    r_vec.resize(m_vec_size * n_outcome);
	for (i=0; i < m_vec_size * n_outcome; i++) {
        r_vec[i] = 0.0;
	}

    for (i=0; i < m_vec_size - 1; ++i) {
        context_t& cont = sent[i].context;
        // g
        for (j=0; j < cont.size(); j++) {
            int pid = cont[j].pid;
#ifndef BINARY_FEATURE
            double fval = cont[j].fval;
#else
            double fval = 1;
#endif
            if (support_feature) {
                vector<pair<int, int> >& param = params[pid];
                for (k=0; k < param.size(); ++k) {
                    int y2 = param[k].first;
                    int fid = param[k].second;
                    r_vec[MAT2(i,y2)] += theta[fid] * fval;
                }
            } else {
                for (int y2=0; y2 < n_outcome; y2++) {
                    int fid = make_fid(pid, y2);
                    r_vec[MAT2(i,y2)] += theta[fid] * fval;
                }
            }
        }
        // normalize
        double sum = 0.0;
        for (j=0; j < n_outcome; j++) {
            r_vec[MAT2(i,j)] = exp(r_vec[MAT2(i,j)]);
            sum += r_vec[MAT2(i,j)];
        }
        for (j=0; j < n_outcome; j++) {
            r_vec[MAT2(i,j)] /= sum;
        }
    } // end for
}


// for eval_with_constraint
// ""인 경우는 constrain이 없는 것임
void SSVM::constrain_R_matrix(vector<double>& r_vec, const vector<string>& constraint) {
    register int i, j;

    for (i=0; i < constraint.size(); i++) {
        if (constraint[i] == "") continue;
        if (outcome_map.find(constraint[i]) == outcome_map.end()) {
            if (verbose) cerr << "CRF: constrain_R_matrix: unknown constraint: " << constraint[i] << endl;
            continue;
        }
        int outcome = outcome_map[constraint[i]];
        for (j=0; j < n_outcome; j++) {
            if (j != outcome) {
                r_vec[MAT2(i,j)] = -1e10;
            }
        }
    } 
}


// viterbi search
vector<int> SSVM::viterbi(vector<double>& r_vec, sent_t& sent, double& prob) {
    vector<vector<int> > psi;
    vector<vector<double> > delta;
    int i, j, k, l;
    double cur;
	int m_vec_size = sent.size() + 1;
    // reserve
    psi.reserve(m_vec_size);
    delta.reserve(m_vec_size);

    vector<int> psi_i;
    vector<double> delta_i;
    vector<int> beam_vec;

    for (i=0; i < m_vec_size; i++) {
        psi_i.clear();
        delta_i.clear();

        if (i > 0 && beam > 0) {
            // for beam search
            beam_vec.clear();
            PQueue<int> pqueue(beam);
            // sorting delta[i-1]
            for (k=0; k < n_outcome; k++) {
               pqueue.insert(make_pair(delta[i-1][k], k));
            }
            // test
            //cerr << n_outcome << endl;
            //pqueue.print();

            for (l=0; l < pqueue.n && l < pqueue.size(); l++) {
                beam_vec.push_back(pqueue.queue[l].second);
            }
        }

        for (j=0; j < n_outcome; j++) {
            double max = -1e15;
            int max_k = 0;
            if (i == 0) {
                max = r_vec[MAT2(i,j)];
                max_k = default_oid;
            } else if (i < m_vec_size - 1) {
                if (beam == 0) {
                    // dense graph = fully-connected
                    for (k=0; k < n_outcome; k++) {
                        cur = delta[i-1][k] + r_vec[MAT2(i,j)] + m_vec[MAT2(k,j)];
                        if (k == 0 || cur > max) {
                            max = cur;
                            max_k = k;
                        }
                    }
                } else {
                    // beam search in dense graph
                    for (l=0; l < beam_vec.size(); l++) {
                        k = beam_vec[l];
                        cur = delta[i-1][k] + r_vec[MAT2(i,j)] + m_vec[MAT2(k,j)];
                        if (l == 0 || cur > max) {
                            max = cur;
                            max_k = k;
                        }
                    }
                }
            } else {
                for (k=0; k < n_outcome; k++) {
                    cur = delta[i-1][k];
                    if (k == 0 || cur > max) {
                        max = cur;
                        max_k = k;
                    }
                }
            }
            delta_i.push_back(max);
            psi_i.push_back(max_k);
        }
        delta.push_back(delta_i);
        psi.push_back(psi_i);
    }

    vector<int> y_seq;
    y_seq.reserve(m_vec_size);
    int prev_y = default_oid;
    for (i = m_vec_size-1; i >= 1; i--) {
        int y = psi[i][prev_y];
        y_seq.push_back(y);
        prev_y = y;
    }
    reverse(y_seq.begin(), y_seq.end());
    prob = delta[m_vec_size-1][default_oid];

    // test
    /*
    cerr << endl << "v: ";
    for (i=0; i < y_seq.size(); i++) {
        cerr << outcome_vec[y_seq[i]] << " ";
    }
    cerr << ": " << delta[m_vec_size-1][0];
    */

    return y_seq;
}


// n-best search
vector<vector<int> > SSVM::viterbi_nbest(vector<double>& r_vec, sent_t& sent, vector<double>& prob, int n)
{
    vector<vector<vector<pair<double, vector<int> > > > > state;
    int i, j, k, l;
	int m_vec_size = sent.size() + 1;

    for (i=0; i < m_vec_size; i++) {
        vector<vector<pair<double, vector<int> > > > state_i;
        for (j=0; j < n_outcome; j++) {
            if (i == 0) {
                k = 0;
                double cur = r_vec[MAT2(i,j)];
                vector<pair<double, vector<int> > > state_ij;
                vector<int> path;
                state_ij.push_back(make_pair(cur, path));
                state_i.push_back(state_ij);
                // test
                //cerr << "[" << i << "," << (*outcome_vec)[j] << "] " << cur << endl;
            } else {
                // 각 state 마다 n-best를 저장
                PQueue<vector<int> > pqueue(n);
                for (k=0; k < n_outcome; k++) {
                    for (l=0; l < state[i-1][k].size(); l++) {
                        double cur = state[i-1][k][l].first;
                        if (i < m_vec_size - 1) {
                            cur += r_vec[MAT2(i,j)] + m_vec[MAT2(k,j)];
                        }
                        vector<int> path = state[i-1][k][l].second;
                        path.push_back(k);
                        pqueue.insert(make_pair(cur, path));
                    }
                }
                vector<pair<double, vector<int> > > state_ij;
                for (l=0; l < pqueue.n && l < pqueue.size(); l++) {
                    state_ij.push_back(pqueue.queue[l]);
                    // test
                    //cerr << "[" << i << "," << (*outcome_vec)[j] << "] " << pqueue.queue[l].second[pqueue.queue[l].second.size()-1] << " " << pqueue.queue[l].first << endl;
                }
                // 마지막에 sorting 해준다
                if (i == m_vec_size - 1) {
                    sort(state_ij.begin(), state_ij.end());
                    reverse(state_ij.begin(), state_ij.end());
                }
                state_i.push_back(state_ij);
            }
        }
        state.push_back(state_i);
    }

    vector<vector<int> > nbest;
    prob.clear();
    for (l=0; l < state[m_vec_size-1][default_oid].size(); l++) { 
        prob.push_back(state[m_vec_size-1][default_oid][l].first);
        nbest.push_back(state[m_vec_size-1][default_oid][l].second);
    }

    // test
    /*
    if (verbose) {
        for (i=0; i < prob.size(); i++) {
            cerr << i << "=" << prob[i] << " ";
        }
        cerr << endl;
        for (i=0; i < nbest.size(); i++) {
            cerr << "[" << i << "] ";
            for (j=0; j < nbest[i].size(); j++) {
                cerr << (*outcome_vec)[nbest[i][j]] << " ";
            }
            cerr << endl;
        }
    }
    */

    return nbest;
}

// 1-best search for MEMM
vector<int> SSVM::viterbi4owps(vector<double>& r_vec, sent_t& sent, double& prob) {
    vector<int> y_seq;
    int i, j;
    prob = 1.0;
	int m_vec_size = sent.size() + 1;

    for (i=0; i < m_vec_size-1; i++) {
        double max = -1e15;
        int max_k = 0;
        for (j=0; j < n_outcome; j++) {
            if (r_vec[MAT2(i,j)] > max) {
                max = r_vec[MAT2(i,j)];
                max_k = j;
            }
        }
        prob += max;
        y_seq.push_back(max_k);
    }
    return y_seq;
}


// find most violated constraint : viterbi 함수와 유사 (사용법도 같음), log_scale 임
vector<int> SSVM::find_most_violated_constraint(vector<double>& r_vec, sent_t& sent, double wscale) {
    vector<vector<int> > psi;
    vector<vector<double> > delta;
    int i, j, k, l, answer;
    double cur, cur_loss = 0;
	int m_vec_size = sent.size() + 1;
    
    for (i=0; i < m_vec_size; i++) {
        if (i < m_vec_size - 1) {
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
                cur_loss = (j != answer ? one_loss : 0);
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
    
    vector<int> y_seq;
    int prev_y = default_oid;
    for (i = m_vec_size-1; i >= 1; i--) {
        int y = psi[i][prev_y];
        y_seq.push_back(y);
        prev_y = y;
    }
    reverse(y_seq.begin(), y_seq.end());
    
    // test
    /*
     cerr << endl;
     for (i=0; i < y_seq.size(); i++)
     cerr << (*outcome_vec)[y_seq[i]] << " ";
     */
    
    return y_seq;
}


// make_diff_vector : f(xi,yi) - f(xi,y)
vect_t SSVM::make_diff_vector(sent_t& sent, vector<int>& y_seq) {
    vect_t vect;
    single_vect_t svect;
    map<int, float> vect_map;
    bool same = true;
    
    for (size_t i=0; i < sent.size(); i++) {
        context_t& cont = sent[i].context;
        int outcome = sent[i].outcome;
        
        if (outcome != y_seq[i]) {
            same = false;
        }
        
        // g
        if (outcome != y_seq[i]) {
            context_t::iterator cit = cont.begin();
            for (; cit != cont.end(); cit++) {
                int pid = cit->pid;
#ifndef BINARY_FEATURE
                double fval = cit->fval;
#else
                double fval = 1;
#endif
                if (support_feature) {
                    // f(xi, yi)
                    int fid = make_fid(pid, outcome);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = fval;
                        else vect_map[fid] += fval;
                    }
                    // - f(xi, y)
                    fid = make_fid(pid, y_seq[i]);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -fval;
                        else vect_map[fid] -= fval;
                    }
                } else {
                    // f(xi, yi)
                    int fid = pid * n_outcome + outcome;
                    if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = fval;
                    else vect_map[fid] += fval;
                    // - f(xi, y)
                    fid = fid - outcome + y_seq[i];
                    if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -fval;
                    else vect_map[fid] -= fval;
                }
            }
        }
        
        // f
        if (i > 0 && !edge_pid.empty()) {
            if (support_feature) {
                // f(xi, yi)
                int y1 = sent[i-1].outcome;
                if (y1 < edge_pid.size()) {
                    int pid = edge_pid[y1];
                    int fid = make_fid(pid, outcome);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = 1;
                        else vect_map[fid] += 1;
                    }
                }
                // - f(xi, y)
                y1 = y_seq[i-1];
                if (y1 < edge_pid.size()) {
                    int pid = edge_pid[y1];
                    int fid = make_fid(pid, y_seq[i]);
                    if (fid >= 0) {
                        if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -1;
                        else vect_map[fid] -= 1;
                    }
                }
            } else {
                // f(xi, yi)
                int y1 = sent[i-1].outcome;
                int pid = edge_pid[y1];
                int fid = pid * n_outcome + outcome;
                if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = 1;
                else vect_map[fid] += 1;
                // - f(xi, y)
                y1 = y_seq[i-1];
                pid = edge_pid[y1];
                fid = pid * n_outcome + y_seq[i];
                if (vect_map.find(fid) == vect_map.end()) vect_map[fid] = -1;
                else vect_map[fid] -= 1;
            }
        } // end context
    } // end sentence
    
    // vect : 정답이면 empty vect 리턴
    if (same) {
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


// append_diff_vector : s_vect에 {f(xi,yi) - f(xi,y)}을 더함
// s_vect는 dense vector를 사용함
void SSVM::append_diff_vector(vector<float>& dense_vect, vect_t& vect) {
    size_t i, j, n;
    int fid;
    
    for (i=0; i < vect.size(); i++) {
        single_vect_t& svect = vect[i];
        // sparse vector
        n = svect.vect.size();
        if (svect.factor == 1) {
            for (j=0; j < n; j++) {
                fid = svect.vect[j].first;
                dense_vect[fid] += svect.vect[j].second;
            }
        } else {
            for (j=0; j < n; j++) {
                fid = svect.vect[j].first;
                dense_vect[fid] += svect.factor * svect.vect[j].second;
            }
        }
    }
}


// calculate loss
double SSVM::calculate_loss(sent_t& sent, vector<int>& y_seq) {
    int i, answer = 0;
    double incorrect = 0;
    
    for (i=0; i < sent.size(); i++) {
        answer = sent[i].outcome;
        if (answer != y_seq[i]) {
            incorrect += 1;
        }
    }
    return incorrect;
}


/** Tokenize string to words.
 Tokenization of string and assignment to word vector.
 Delimiters are set of char.
 @param str string
 @param tokens token vector
 @param delimiters delimiters to divide string
 @return none
 */
void SSVM::tokenize(const string& str, vector<string>& tokens, const string& delimiters) {
    tokens.clear();
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos  = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
} 

/** Split string to words.
 Tokenization of string and assignment to word vector.
 Delimiter is string.
 @param str string
 @param tokens token vector
 @param delimiter delimiter to divide string
 @return none
 */
void SSVM::split(const string& str, vector<string>& tokens, const string& delimiter) {
    tokens.clear();
    string::size_type pos = str.find(delimiter, 0);
    string::size_type lastPos = 0;
    while (0 <= pos && str.size() > pos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos + delimiter.size();
        pos = str.find(delimiter, lastPos);
    }
    tokens.push_back(str.substr(lastPos, str.size() - lastPos));
}

