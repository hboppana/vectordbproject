#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    // Wait for a task or shutdown signal
                    condition_.wait(lock, [this] { return !tasks_.empty() || stop_; });
                    
                    if (stop_ && tasks_.empty()) {
                        break;
                    }
                    
                    if (tasks_.empty()) {
                        continue;
                    }
                    
                    auto task = std::move(tasks_.front());
                    tasks_.pop();
                    lock.unlock();
                    
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    template<typename F>
    void enqueue(F&& func) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks_.emplace(std::forward<F>(func));
        }
        condition_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return tasks_.empty(); });
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};
