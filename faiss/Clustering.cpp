/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/Clustering.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/kmeans1d.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

Clustering::Clustering(int d, int k) : d(d), k(k) {}

Clustering::Clustering(int d, int k, const ClusteringParameters& cp)
        : ClusteringParameters(cp), d(d), k(k) {}

void Clustering::post_process_centroids() {     //在每次更新聚类中心后对其进行后处理
    if (spherical) {
        fvec_renorm_L2(d, k, centroids.data());
    }

    if (int_centroids) {
        for (size_t i = 0; i < centroids.size(); i++)
            centroids[i] = roundf(centroids[i]);
    }
}

void Clustering::train(     //调用 train_encoded 函数来执行训练
        idx_t nx,
        const float* x_in,
        Index& index,
        const float* weights) {
    train_encoded(
            nx,
            reinterpret_cast<const uint8_t*>(x_in),
            nullptr,
            index,
            weights);
}

namespace {

uint64_t get_actual_rng_seed(const int seed) {
    return (seed >= 0)
            ? seed
            : static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                            .time_since_epoch()
                                            .count());
}

idx_t subsample_training_set(
        const Clustering& clus,
        idx_t nx,
        const uint8_t* x,
        size_t line_size,
        const float* weights,
        uint8_t** x_out,
        float** weights_out) {
    if (clus.verbose) {
        printf("Sampling a subset of %zd / %" PRId64 " for training\n",
               clus.k * clus.max_points_per_centroid,
               nx);
    }

    const uint64_t actual_seed = get_actual_rng_seed(clus.seed);

    std::vector<int> perm;
    if (clus.use_faster_subsampling) {
        // use subsampling with splitmix64 rng
        SplitMix64RandomGenerator rng(actual_seed);

        const idx_t new_nx = clus.k * clus.max_points_per_centroid;
        perm.resize(new_nx);
        for (idx_t i = 0; i < new_nx; i++) {
            perm[i] = rng.rand_int(nx);
        }
    } else {
        // use subsampling with a default std rng
        perm.resize(nx);
        rand_perm(perm.data(), nx, actual_seed);
    }

    nx = clus.k * clus.max_points_per_centroid;
    uint8_t* x_new = new uint8_t[nx * line_size];
    *x_out = x_new;

    // might be worth omp-ing as well
    for (idx_t i = 0; i < nx; i++) {
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    }
    if (weights) {
        float* weights_new = new float[nx];
        for (idx_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }
    return nx;
}

/** compute centroids as (weighted) sum of training points
 *
 * @param x            训练向量数组, size n * code_size (from codec)
 * @param codec        how to decode the vectors (if NULL then cast to float*)
 * @param weights      每个训练样本的权重，大小为 n
 * @param assign       表示第 i 个训练样本被分配到的最近质心编号（聚类标签）
 * @param k_frozen     前 k_frozen 个质心不进行更新，用于冻结已有质心
 * @param centroids    数组，每一行是一个聚类质心向量   
 * @param hassign      每个质心的分配直方图，大小为 k ,第 ci 个质心被分配了多少样本
 *                     
 *
 */

void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(*centroids) * d * k);   //把质心初始化为 0

    size_t line_size = codec ? codec->sa_code_size() : d * sizeof(float);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i];     //获取该样本所属的聚类编号
            assert(ci >= 0 && ci < k + k_frozen);
            ci -= k_frozen;
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;  //找到当前样本所属的质心在 centroids 数组中的地址
                const float* xi;
                if (!codec) {
                    xi = reinterpret_cast<const float*>(x + i * line_size); //把原始字节转换为 float* 指针
                } else {
                    float* xif = decode_buffer.data();
                    codec->sa_decode(1, x + i * line_size, xif);
                    xi = xif;
                }
                if (weights) {  //每个样本对其聚类中心的贡献按权重加权
                    float w = weights[i];
                    hassign[ci] += w;
                    for (size_t j = 0; j < d; j++) {    //按维度加权累加样本向量
                        c[j] += xi[j] * w;
                    }
                } else {
                    hassign[ci] += 1.0;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j];
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (idx_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        //hassign[ci] 表示聚到第 ci 个质心的训练向量数量
        float norm = 1 / hassign[ci];       //将每个质心向量取平均，完成 KMeans 聚类中的“更新质心”
        float* c = centroids + ci * d;      //第 ci 个质心向量的起始地址
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;       //遍历每一维 j，将之前累加的值除以 hassign[ci]
        }
    }
}

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)

/** Handle empty clusters by splitting larger ones.
 *
 * It works by slightly changing the centroids to make 2 clusters from
 * a single one. Takes the same arguments as compute_centroids.
 *
 * @return           nb of spliting operations (larger is worse)
 */
int split_clusters(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng(1234);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            size_t cj;
            for (cj = 0; true; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            memcpy(centroids + ci * d,
                   centroids + cj * d,
                   sizeof(*centroids) * d);

            /* small symmetric pertubation */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}

} // namespace

void Clustering::train_encoded(     // k-means 聚类训练的完整实现
        idx_t nx,
        const uint8_t* x_in,
        const Index* codec,
        Index& index,
        const float* weights) {
    FAISS_THROW_IF_NOT_FMT(
            nx >= k,
            "Number of training points (%" PRId64
            ") should be at least "
            "as large as number of clusters (%zd)",
            nx,
            k);

    FAISS_THROW_IF_NOT_FMT(
            (!codec || codec->d == d),
            "Codec dimension %d not the same as data dimension %d",
            int(codec->d),
            int(d));

    FAISS_THROW_IF_NOT_FMT(
            index.d == d,
            "Index dimension %d not the same as data dimension %d",
            int(index.d),
            int(d));

    double t0 = getmillisecs();

    if (!codec && check_input_data_for_NaNs) {
        // Check for NaNs in input data. Normally it is the user's
        // responsibility, but it may spare us some hard-to-debug
        // reports.
        const float* x = reinterpret_cast<const float*>(x_in);
        for (size_t i = 0; i < nx * d; i++) {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(x[i]), "input contains NaN's or Inf's");
        }
    }

    const uint8_t* x = x_in;        //样本数据指针，连续存储了 nx 个样本，每个样本占 line_size 字节
    std::unique_ptr<uint8_t[]> del1;    //声明一个智能指针 del1，用来管理动态分配的 uint8_t 数组内存
    std::unique_ptr<float[]> del3;      //声明另一个智能指针 del3，管理动态分配的 float 数组内存
    size_t line_size = codec ? codec->sa_code_size() : sizeof(float) * d;   //计算每个训练样本（向量）占用的字节数

    if (nx > k * max_points_per_centroid) {     //如果训练点数远多于每个簇的最大允许点数乘以簇数，说明数据太多了，可能训练太慢或者内存压力大
        uint8_t* x_new;
        float* weights_new;
        nx = subsample_training_set(        //调用 subsample_training_set() 对训练数据进行下采样，减少训练点数
                *this, nx, x, line_size, weights, &x_new, &weights_new);
        del1.reset(x_new);
        x = x_new;
        del3.reset(weights_new);
        weights = weights_new;
    } else if (nx < k * min_points_per_centroid) {      //训练点数太少，不足以满足每个簇所需的最少点数，打印一个警告，提示训练数据可能不够
        fprintf(stderr,
                "WARNING clustering %" PRId64
                " points to %zd centroids: "
                "please provide at least %" PRId64 " training points\n",
                nx,
                k,
                idx_t(k) * min_points_per_centroid);
    }

    if (nx == k) {      //当训练样本数 nx 正好等于聚类中心数 k 时，直接用样本当质心，不用真正聚类
        // this is a corner case, just copy training set to clusters
        if (verbose) {
            printf("Number of training points (%" PRId64
                   ") same as number of "
                   "clusters, just copying\n",
                   nx);
        }
        centroids.resize(d * k);
        if (!codec) {
            memcpy(centroids.data(), x_in, sizeof(float) * d * k);
        } else {
            codec->sa_decode(nx, x_in, centroids.data());
        }

        // one fake iteration...
        ClusteringIterationStats stats = {0.0, 0.0, 0.0, 1.0, 0};
        iteration_stats.push_back(stats);

        index.reset();
        index.add(k, centroids.data());
        return;
    }

    if (verbose) {
        printf("Clustering %" PRId64
               " points in %zdD to %zd clusters, "
               "redo %d times, %d iterations\n",
               nx,
               d,
               k,
               nredo,
               niter);
        if (codec) {
            printf("Input data encoded in %zd bytes per vector\n",
                   codec->sa_code_size());
        }
    }

    std::unique_ptr<idx_t[]> assign(new idx_t[nx]);     //assign：分配一个大小为 nx 的数组，用来存储每个样本点所属的簇索引  
    std::unique_ptr<float[]> dis(new float[nx]);        //dis：分配一个大小为 nx 的数组，用来存储每个样本点到其簇中心的距离

    // remember best iteration for redo
    bool lower_is_better = !is_similarity_metric(index.metric_type);    //判断当前的距离度量是不是相似度度量（metric_type 是距离还是相似度），如果是距离，目标是越小越好
    float best_obj = lower_is_better ? HUGE_VALF : -HUGE_VALF;      //best_obj：记录当前最优目标函数值（比如总距离和）
    std::vector<ClusteringIterationStats> best_iteration_stats;     //best_iteration_stats 和 best_centroids：保存每次训练的统计信息和最优的簇中心。
    std::vector<float> best_centroids;

    // support input centroids

    FAISS_THROW_IF_NOT_MSG(
            centroids.size() % d == 0,
            "size of provided input centroids not a multiple of dimension");

    size_t n_input_centroids = centroids.size() / d;        //计算当前已经给出的初始质心（centroids）的个数

    if (verbose && n_input_centroids > 0) {
        printf("  Using %zd centroids provided as input (%sfrozen)\n",
               n_input_centroids,
               frozen_centroids ? "" : "not ");
    }

    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n", (getmillisecs() - t0) / 1000.);
    }
    t0 = getmillisecs();

    // initialize seed
    const uint64_t actual_seed = get_actual_rng_seed(seed);     //获取真正用于随机数生成器的种子，保证后续随机操作（比如质心初始化）可重复或有确定性

    // temporary buffer to decode vectors during the optimization
    std::vector<float> decode_buffer(codec ? d * decode_block_size : 0);        //如果用到了编码器（codec 非空），就申请一个临时浮点数组，大小是 d * decode_block_size，用于解码编码后的向量，方便后续计算

    for (int redo = 0; redo < nredo; redo++) {      //redo 表示为了避免局部最优，进行多次聚类重试，每次随机初始化。
        if (verbose && nredo > 1) {
            printf("Outer iteration %d / %d\n", redo, nredo);
        }

        // initialize (remaining) centroids with random points from the dataset
        centroids.resize(d * k);    //分配 k 个 d 维质心的存储空间
        std::vector<int> perm(nx);

        rand_perm(perm.data(), nx, actual_seed + 1 + redo * 15486557L); //用 actual_seed 派生一个新种子生成随机排列 perm，确保每次 redo 初始化的点不同

        //选择k个聚类中心
        if (!codec) {   //如果没有编码器（!codec），就直接 memcpy 向量
            for (int i = n_input_centroids; i < k; i++) {   //从已有质心数量 n_input_centroids 继续初始化剩余的质心
                memcpy(&centroids[i * d], x + perm[i] * line_size, line_size);  //指针偏移操作:取第i个被随机选中的样本复制line_size字节长度到质心的储存空间中,即将随机选中的样本向量作为第 i 个初始化质心
            }
        } else {
            for (int i = n_input_centroids; i < k; i++) {   //有编码器就先解码（sa_decode）再存入质心
                codec->sa_decode(1, x + perm[i] * line_size, &centroids[i * d]);
            }
        }

        post_process_centroids();   //在每次更新聚类中心后对其进行后处理

        // prepare the index

        if (index.ntotal != 0) {    //清空已有数据（如果 index 中已经有向量）
            index.reset();
        }

        if (!index.is_trained) {        //如果 index 还没有训练，先用 centroids 对其进行训练
            index.train(k, centroids.data());
        }

        index.add(k, centroids.data());     //将 k 个聚类中心加入到 IVF 索引的倒排文件结构中

        // k-means iterations
        float obj = 0;
        for (int i = 0; i < niter; i++) {
            double t0s = getmillisecs();

            if (!codec) {
                index.search(       //执行向量搜索，返回查询向量最近的一个质心的距离和索引
                        nx,
                        reinterpret_cast<const float*>(x),
                        1,
                        dis.get(),
                        assign.get());
            } else {
                // search by blocks of decode_block_size vectors
                size_t code_size = codec->sa_code_size();
                for (size_t i0 = 0; i0 < nx; i0 += decode_block_size) {
                    size_t i1 = i0 + decode_block_size;
                    if (i1 > nx) {
                        i1 = nx;
                    }
                    codec->sa_decode(
                            i1 - i0, x + code_size * i0, decode_buffer.data());
                    index.search(
                            i1 - i0,
                            decode_buffer.data(),
                            1,
                            dis.get() + i0,
                            assign.get() + i0);
                }
            }

            InterruptCallback::check();
            t_search_tot += getmillisecs() - t0s;

            // accumulate objective
            obj = 0;    //把所有向量到其对应最近质心的距离累加，得到当前的聚类误差obj
            for (int j = 0; j < nx; j++) {
                obj += dis[j];
            }

            // update the centroids
            std::vector<float> hassign(k);

            size_t k_frozen = frozen_centroids ? n_input_centroids : 0;
            compute_centroids(      //重新分配簇中心位置
                    d,
                    k,
                    nx,
                    k_frozen,
                    x,
                    codec,
                    assign.get(),
                    weights,
                    hassign.data(),
                    centroids.data());

            int nsplit = split_clusters(    //对大簇进行分裂，防止某些簇过大导致效果不佳
                    d, k, nx, k_frozen, hassign.data(), centroids.data());

            // collect statistics
            ClusteringIterationStats stats = {      //收集本次迭代的时间、搜索时间、目标值、不平衡度和分裂簇数，存入iteration_stats
                    obj,
                    (getmillisecs() - t0) / 1000.0,
                    t_search_tot / 1000,
                    imbalance_factor(nx, k, assign.get()),
                    nsplit};
            iteration_stats.push_back(stats);

            if (verbose) {
                printf("  Iteration %d (%.2f s, search %.2f s): "
                       "objective=%g imbalance=%.3f nsplit=%d       \r",
                       i,
                       stats.time,
                       stats.time_search,
                       stats.obj,
                       stats.imbalance_factor,
                       nsplit);
                fflush(stdout);
            }

            post_process_centroids();

            // add centroids to index for the next iteration (or for output)

            index.reset();
            if (update_index) {
                index.train(k, centroids.data());
            }

            index.add(k, centroids.data());
            InterruptCallback::check();
        }

        if (verbose)
            printf("\n");
        if (nredo > 1) {
            if ((lower_is_better && obj < best_obj) ||
                (!lower_is_better && obj > best_obj)) {
                if (verbose) {
                    printf("Objective improved: keep new clusters\n");
                }
                best_centroids = centroids;
                best_iteration_stats = iteration_stats;
                best_obj = obj;
            }
            index.reset();
        }
    }
    if (nredo > 1) {        //保存和恢复效果最好的聚类结果
        centroids = best_centroids;
        iteration_stats = best_iteration_stats;
        index.reset();
        index.add(k, best_centroids.data());    //在所有重试聚类完成后，把质心和统计信息恢复成最好的那个，并重新用最好的质心更新索引
    }
}

Clustering1D::Clustering1D(int k) : Clustering(1, k) {}

Clustering1D::Clustering1D(int k, const ClusteringParameters& cp)
        : Clustering(1, k, cp) {}

void Clustering1D::train_exact(idx_t n, const float* x) {
    const float* xt = x;

    std::unique_ptr<uint8_t[]> del;
    if (n > k * max_points_per_centroid) {
        uint8_t* x_new;
        float* weights_new;
        n = subsample_training_set(
                *this,
                n,
                (uint8_t*)x,
                sizeof(float) * d,
                nullptr,
                &x_new,
                &weights_new);
        del.reset(x_new);
        xt = (float*)x_new;
    }

    centroids.resize(k);
    double uf = kmeans1d(xt, n, k, centroids.data());

    ClusteringIterationStats stats = {0.0, 0.0, 0.0, uf, 0};
    iteration_stats.push_back(stats);
}

float kmeans_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids) {
    Clustering clus(d, k);
    clus.verbose = d * n * k > (size_t(1) << 30);
    // display logs if > 1Gflop per iteration
    IndexFlatL2 index(d);
    clus.train(n, x, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

/******************************************************************************
 * ProgressiveDimClustering implementation
 ******************************************************************************/

ProgressiveDimClusteringParameters::ProgressiveDimClusteringParameters() {
    progressive_dim_steps = 10;
    apply_pca = true; // seems a good idea to do this by default
    niter = 10;       // reduce nb of iterations per step
}

Index* ProgressiveDimIndexFactory::operator()(int dim) {
    return new IndexFlatL2(dim);
}

ProgressiveDimClustering::ProgressiveDimClustering(int d, int k) : d(d), k(k) {}

ProgressiveDimClustering::ProgressiveDimClustering(
        int d,
        int k,
        const ProgressiveDimClusteringParameters& cp)
        : ProgressiveDimClusteringParameters(cp), d(d), k(k) {}

namespace {

void copy_columns(idx_t n, idx_t d1, const float* src, idx_t d2, float* dest) {
    idx_t d = std::min(d1, d2);
    for (idx_t i = 0; i < n; i++) {
        memcpy(dest, src, sizeof(float) * d);
        src += d1;
        dest += d2;
    }
}

} // namespace

void ProgressiveDimClustering::train(
        idx_t n,
        const float* x,
        ProgressiveDimIndexFactory& factory) {
    int d_prev = 0;

    PCAMatrix pca(d, d);

    std::vector<float> xbuf;
    if (apply_pca) {
        if (verbose) {
            printf("Training PCA transform\n");
        }
        pca.train(n, x);
        if (verbose) {
            printf("Apply PCA\n");
        }
        xbuf.resize(n * d);
        pca.apply_noalloc(n, x, xbuf.data());
        x = xbuf.data();
    }

    for (int iter = 0; iter < progressive_dim_steps; iter++) {
        int di = int(pow(d, (1. + iter) / progressive_dim_steps));
        if (verbose) {
            printf("Progressive dim step %d: cluster in dimension %d\n",
                   iter,
                   di);
        }
        std::unique_ptr<Index> clustering_index(factory(di));

        Clustering clus(di, k, *this);
        if (d_prev > 0) {
            // copy warm-start centroids (padded with 0s)
            clus.centroids.resize(k * di);
            copy_columns(
                    k, d_prev, centroids.data(), di, clus.centroids.data());
        }
        std::vector<float> xsub(n * di);
        copy_columns(n, d, x, di, xsub.data());

        clus.train(n, xsub.data(), *clustering_index.get());

        centroids = clus.centroids;
        iteration_stats.insert(
                iteration_stats.end(),
                clus.iteration_stats.begin(),
                clus.iteration_stats.end());

        d_prev = di;
    }

    if (apply_pca) {
        if (verbose) {
            printf("Revert PCA transform on centroids\n");
        }
        std::vector<float> cent_transformed(d * k);
        pca.reverse_transform(k, centroids.data(), cent_transformed.data());
        cent_transformed.swap(centroids);
    }
}

} // namespace faiss
