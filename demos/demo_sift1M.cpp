/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "faiss/IndexHNSW.h"

#include <sys/stat.h>

#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/
//读取fvecs文件，移除维度信息，返回float向量部分
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

//读取ivecs文件：唯一区别是数据类型是 int 不是 float， 直接将返回的 x 从原本的 float* 强转为 int*
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    double t0 = elapsed();

    // this is typically the fastest one.
    //const char* index_key = "IVF4096,Flat";    //"IVF4096" 表示使用 IVF（Inverted File）索引，nlist=4096，也就是分成 4096 个簇
      const char* index_key = "HNSW16,Flat";

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    faiss::Index* index;
    
    size_t d;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        //d 被写入向量的维度； nt 被写入向量的数量； 返回值 xt 是一个指向大小为 nt × d 的 float 数组的指针
        float* xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);
        //根据描述将 index 赋值给对应的 Index 子类,这里对应的为 IndexIVFFlat 子类
        index = faiss::index_factory(d, index_key); 
        index->verbose = true;
        faiss::index_factory_verbose = true;
        ((faiss::IndexHNSW*)index)->hnsw.efConstruction = 50;

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
        
        //构建簇和质心
        index->train(nt, xt);
        delete[] xt;
    }

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float* xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);

        //把 base 里的向量加载到已经训练好的索引中(这里才开始构图但是没有聚类)
        index->add(nb, xb);

        delete[] xb;
    }

    size_t nq;  //查询向量的数量
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;         // nb of results per query in the GT
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    // Result of the auto-tuning
    std::string selected_params;

    { // run auto-tuning

        printf("[%.3f s] Preparing auto-tune criterion 1-recall at 1 "
               "criterion, with k=%ld nq=%ld\n",
               elapsed() - t0,
               k,
               nq);

        //衡量 1-recall@1 的召回率
        faiss::OneRecallAtRCriterion crit(nq, 1);
        crit.set_groundtruth(k, nullptr, gt);
        crit.nnn = k; // by default, the criterion will request only 1 NN

        printf("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);
        
        //针对当前索引 index 生成可调节参数空间
        faiss::ParameterSpace params;
        params.initialize(index);

        printf("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
               elapsed() - t0,
               params.parameter_ranges.size(),
               params.n_combinations());

        faiss::OperatingPoints ops;

        //尝试不同的参数组合，评估每组参数下的搜索效果
        params.explore(index, nq, xq, crit, &ops);

        printf("[%.3f s] Found the following operating points: \n",
               elapsed() - t0);

        ops.display();

        // 从自动调参得到的最优点中，选择第一个 recall@1 大于 0.5 的参数组合，并将其保存到 selected_params 中
        for (int i = 0; i < ops.optimal_pts.size(); i++) {
            if (ops.optimal_pts[i].perf > 0.5) {
                selected_params = ops.optimal_pts[i].key;
                break;
            }
        }
        assert(selected_params.size() >= 0 ||
               !"could not find good enough op point");
    }

    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;

        printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
               elapsed() - t0,
               selected_params.c_str());

        params.set_index_parameters(index, selected_params.c_str());

        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               nq);
        
        //计算搜索结果的 Recall@1、Recall@10、Recall@100
        faiss::idx_t* I = new faiss::idx_t[nq * k];
        float* D = new float[nq * k];

        index->search(nq, xq, k, D, I);

        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
