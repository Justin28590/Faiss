//每个倒排桶用的是 IndexFlatL2（原始向量),适合小数据量,精度高

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

using idx_t = faiss::idx_t;

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size: 0.1M
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.; //在向量的第一个维度增加一个线性偏移量，避免随机重合
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 100;    //聚类中心数
    int k = 4;          //查询时返回的邻居数

    faiss::IndexFlatL2 quantizer(d);    //创建了一个用于划分倒排桶的“量化器”：平面的，没有创建 HNSW 结构
    faiss::IndexIVFFlat index(&quantizer, d, nlist);        //创建了一个倒排文件索引（IVF） 平面的Flat
    index.verbose = true;   // IndexIVF 有 verbose 成员，可以直接设置                                                                                                                                                                                                                   
    assert(!index.is_trained);  //确保在调用 train() 之前，索引是未训练状态,否则抛出异常
    index.train(nb, xb);        //训练倒排索引（IVF）的量化器和编码器                                                                                                                                                     
    assert(index.is_trained);
    index.add(nb, xb);      //添加到 index 的索引中去

    { //对查询向量 xq 进行 k 近邻搜索，并打印最后 5 个查询结果的邻居索引和距离
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("I=\n");                                                                     
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
