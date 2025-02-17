#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);

    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        printf("testtest alloc size = %zu\n", size);
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        for (auto it = free_blocks.begin(); it != free_blocks.end(); it ++) {
          printf("size = %zu \n", it->second);
          if (it->second >= size) {
            printf("size = %zu \n", it->second);
            if (it->second > size)
              free_blocks[it->first + size] = it->second - size;
            size_t rst = it->first; 
            free_blocks.erase(it->first);
            return rst;
          }
        }

        size_t rst = peak;
        peak += size;
        printf("alloc rst = %zu \n", rst);
        used += size;
        return rst;
    }

    void Allocator::update_free() {
        vector<int> addrs;
        vector<int> sizes;

        for (const auto & pair : free_blocks) {
          addrs.push_back(pair.first);
          sizes.push_back(pair.second);
        }

        int n = addrs.size();
        for (int i = 0; i < n - 1; i ++) {
          int j = i + 1;
          while (j < n && addrs[i] + sizes[i] == addrs[j]) {
            free_blocks.erase(addrs[j]);
            sizes[i] += sizes[j];
            j ++;
          }
          free_blocks[addrs[i]] = sizes[i];
          i = j - 1;
        }

        if(free_blocks.size()) {
          auto last_elem = *free_blocks.rbegin();
          if (last_elem.first + last_elem.second == peak) {
            peak -= last_elem.second;
            free_blocks.erase(last_elem.first);
          }
        }
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        printf("free[%zu] = %zu \n", addr, size);
        free_blocks[addr] = size;

        used -= size;
        update_free();
        // update  TODO
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
