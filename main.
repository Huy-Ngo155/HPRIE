#pragma once

#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <expected>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <tuple>
#include <variant>
#include <optional>
#include <functional>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <concurrentqueue.h>
#include <readerwriterqueue.h>
#include <linux/limits.h>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#include <numa.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

namespace MedicalCore {

template<typename T>
struct TypeTraits {
    static constexpr bool is_numeric = std::is_arithmetic_v<T>;
    static constexpr bool is_floating = std::is_floating_point_v<T>;
    static constexpr bool is_integer = std::is_integral_v<T>;
};

template<typename T, typename Enable = void>
struct AllocatorSelector {
    using type = std::allocator<T>;
};

template<typename T>
struct AllocatorSelector<T, std::enable_if_t<TypeTraits<T>::is_numeric>> {
#ifdef __linux__
    class NumaAlignedAllocator : public std::allocator<T> {
        int numa_node_;
        size_t alignment_;
    public:
        NumaAlignedAllocator(int node = 0, size_t align = 64) 
            : numa_node_(node), alignment_(align) {}
        
        template<typename U>
        NumaAlignedAllocator(const NumaAlignedAllocator<U>& other) 
            : numa_node_(other.numa_node_), alignment_(other.alignment_) {}
        
        T* allocate(size_t n) {
            void* ptr = nullptr;
#ifdef __linux__
            if (numa_node_ >= 0 && alignment_ > 0) {
                ptr = numa_alloc_onnode(n * sizeof(T), numa_node_);
                if (ptr && (reinterpret_cast<uintptr_t>(ptr) % alignment_ != 0)) {
                    numa_free(ptr, n * sizeof(T));
                    ptr = nullptr;
                }
            }
#endif
            if (!ptr) {
                if (posix_memalign(&ptr, alignment_, n * sizeof(T)) != 0) {
                    throw std::bad_alloc();
                }
            }
            return static_cast<T*>(ptr);
        }
        
        void deallocate(T* p, size_t n) {
#ifdef __linux__
            if (numa_node_ >= 0) {
                numa_free(p, n * sizeof(T));
                return;
            }
#endif
            free(p);
        }
        
        int numa_node() const { return numa_node_; }
        size_t alignment() const { return alignment_; }
    };
    using type = NumaAlignedAllocator;
#else
    class AlignedAllocator : public std::allocator<T> {
        size_t alignment_;
    public:
        AlignedAllocator(size_t align = 64) : alignment_(align) {}
        
        template<typename U>
        AlignedAllocator(const AlignedAllocator<U>& other) 
            : alignment_(other.alignment_) {}
        
        T* allocate(size_t n) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment_, n * sizeof(T)) != 0) {
                throw std::bad_alloc();
            }
            return static_cast<T*>(ptr);
        }
        
        void deallocate(T* p, size_t n) {
            free(p);
        }
        
        size_t alignment() const { return alignment_; }
    };
    using type = AlignedAllocator;
#endif
};

template<typename T>
using OptimizedAllocator = typename AllocatorSelector<T>::type;

template<typename T, size_t Capacity>
class RingBuffer {
    std::array<T, Capacity> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<size_t> count_{0};
    
public:
    bool try_push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        count_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    bool try_pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        
        item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        count_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    
    size_t size() const {
        return count_.load(std::memory_order_relaxed);
    }
    
    bool empty() const {
        return size() == 0;
    }
    
    bool full() const {
        return size() == Capacity;
    }
};

template<typename T>
class LockFreeMPMCQueue {
    moodycamel::ConcurrentQueue<T> queue_;
    
public:
    bool try_enqueue(const T& item) {
        return queue_.try_enqueue(item);
    }
    
    bool try_enqueue(T&& item) {
        return queue_.try_enqueue(std::move(item));
    }
    
    bool try_dequeue(T& item) {
        return queue_.try_dequeue(item);
    }
    
    size_t size_approx() const {
        return queue_.size_approx();
    }
};

class HyperThreadAwareExecutor {
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{true};
    std::function<void()> work_fn_;
    
#ifdef __linux__
    void set_thread_affinity(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#endif
    
public:
    HyperThreadAwareExecutor(size_t worker_count, std::function<void()> work) 
        : work_fn_(std::move(work)) {
        
        for (size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this, i]() {
#ifdef __linux__
                set_thread_affinity(static_cast<int>(i));
#endif
                while (running_.load(std::memory_order_relaxed)) {
                    work_fn_();
                }
            });
        }
    }
    
    ~HyperThreadAwareExecutor() {
        running_.store(false);
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }
};

template<typename T, size_t N>
class Histogram {
    std::array<std::atomic<int64_t>, N> buckets_;
    std::atomic<int64_t> total_{0};
    
    static constexpr double log_base = 1.1;
    
    size_t bucket_index(T value) const {
        if (value <= 0) return 0;
        double log_val = std::log(static_cast<double>(value)) / std::log(log_base);
        return std::min(static_cast<size_t>(log_val), N - 1);
    }
    
public:
    void record(T value) {
        buckets_[bucket_index(value)].fetch_add(1, std::memory_order_relaxed);
        total_.fetch_add(1, std::memory_order_relaxed);
    }
    
    T percentile(double p) const {
        int64_t target = static_cast<int64_t>(total_.load() * p / 100.0);
        int64_t accumulated = 0;
        
        for (size_t i = 0; i < N; ++i) {
            accumulated += buckets_[i].load();
            if (accumulated >= target) {
                return static_cast<T>(std::pow(log_base, i));
            }
        }
        return static_cast<T>(std::pow(log_base, N - 1));
    }
};

class SessionManager {
    struct SessionMetadata {
        std::string model_version;
        std::chrono::steady_clock::time_point created_at;
        int64_t inference_count;
        size_t memory_usage;
        bool is_healthy;
    };
    
    std::unordered_map<std::string, SessionMetadata> metadata_;
    mutable std::shared_mutex metadata_mutex_;
    
public:
    void update_metadata(const std::string& session_id, 
                        const std::string& version,
                        size_t memory_usage) {
        std::unique_lock lock(metadata_mutex_);
        auto& meta = metadata_[session_id];
        meta.model_version = version;
        meta.created_at = std::chrono::steady_clock::now();
        meta.memory_usage = memory_usage;
        meta.is_healthy = true;
    }
    
    void record_inference(const std::string& session_id) {
        std::shared_lock lock(metadata_mutex_);
        if (auto it = metadata_.find(session_id); it != metadata_.end()) {
            it->second.inference_count++;
        }
    }
    
    void mark_unhealthy(const std::string& session_id) {
        std::unique_lock lock(metadata_mutex_);
        if (auto it = metadata_.find(session_id); it != metadata_.end()) {
            it->second.is_healthy = false;
        }
    }
    
    std::vector<std::string> get_unhealthy_sessions() const {
        std::shared_lock lock(metadata_mutex_);
        std::vector<std::string> unhealthy;
        for (const auto& [id, meta] : metadata_) {
            if (!meta.is_healthy) {
                unhealthy.push_back(id);
            }
        }
        return unhealthy;
    }
};

class ConnectionPool {
    struct Connection {
        int fd{-1};
        std::chrono::steady_clock::time_point last_used;
        bool in_use{false};
        
        bool is_valid() const { return fd >= 0; }
    };
    
    std::vector<Connection> connections_;
    std::queue<size_t> available_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::string host_;
    int port_;
    size_t max_pool_size_;
    std::chrono::milliseconds timeout_;
    
    bool connect_socket(int& fd) {
#ifdef __linux__
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return false;
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port_);
        inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr);
        
        if (connect(fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(fd);
            return false;
        }
        return true;
#else
        return false;
#endif
    }
    
public:
    ConnectionPool(const std::string& host, int port, size_t max_size,
                  std::chrono::milliseconds timeout)
        : host_(host), port_(port), max_pool_size_(max_size), timeout_(timeout) {}
    
    class ScopedConnection {
        ConnectionPool* pool_{nullptr};
        size_t index_{0};
        
    public:
        ScopedConnection(ConnectionPool* pool, size_t idx) 
            : pool_(pool), index_(idx) {}
        
        ~ScopedConnection() {
            if (pool_) {
                std::lock_guard lock(pool_->mutex_);
                pool_->connections_[index_].in_use = false;
                pool_->connections_[index_].last_used = std::chrono::steady_clock::now();
                pool_->available_.push(index_);
                pool_->cv_.notify_one();
            }
        }
        
        int fd() const { 
            return pool_ ? pool_->connections_[index_].fd : -1; 
        }
    };
    
    std::optional<ScopedConnection> acquire() {
        std::unique_lock lock(mutex_);
        
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        
        while (true) {
            if (!available_.empty()) {
                size_t idx = available_.front();
                available_.pop();
                if (!connections_[idx].in_use) {
                    connections_[idx].in_use = true;
                    connections_[idx].last_used = std::chrono::steady_clock::now();
                    return ScopedConnection(this, idx);
                }
            }
            
            if (connections_.size() < max_pool_size_) {
                Connection conn;
                if (connect_socket(conn.fd)) {
                    conn.last_used = std::chrono::steady_clock::now();
                    conn.in_use = true;
                    size_t idx = connections_.size();
                    connections_.push_back(conn);
                    return ScopedConnection(this, idx);
                }
            }
            
            if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                return std::nullopt;
            }
        }
    }
    
    void cleanup_idle(std::chrono::seconds max_idle) {
        std::lock_guard lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < connections_.size(); ++i) {
            auto& conn = connections_[i];
            if (!conn.in_use && (now - conn.last_used) > max_idle) {
#ifdef __linux__
                if (conn.fd >= 0) {
                    close(conn.fd);
                }
#endif
                conn.fd = -1;
            }
        }
    }
};

class DistributedCoordinator {
    struct NodeInfo {
        std::string id;
        std::string address;
        int port;
        std::atomic<int64_t> load{0};
        std::atomic<int64_t> capacity{100};
        std::atomic<bool> healthy{true};
        std::chrono::steady_clock::time_point last_heartbeat;
    };
    
    std::unordered_map<std::string, NodeInfo> nodes_;
    mutable std::shared_mutex nodes_mutex_;
    std::thread heartbeat_thread_;
    std::atomic<bool> running_{true};
    
    void heartbeat_loop() {
        while (running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            auto now = std::chrono::steady_clock::now();
            std::vector<std::string> to_remove;
            
            {
                std::shared_lock lock(nodes_mutex_);
                for (auto& [id, node] : nodes_) {
                    if (now - node.last_heartbeat > std::chrono::seconds(10)) {
                        node.healthy.store(false);
                        to_remove.push_back(id);
                    }
                }
            }
            
            if (!to_remove.empty()) {
                std::unique_lock lock(nodes_mutex_);
                for (const auto& id : to_remove) {
                    nodes_.erase(id);
                }
            }
        }
    }
    
public:
    DistributedCoordinator() {
        heartbeat_thread_ = std::thread(&DistributedCoordinator::heartbeat_loop, this);
    }
    
    ~DistributedCoordinator() {
        running_.store(false);
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }
    }
    
    void register_node(const std::string& id, const std::string& addr, int port) {
        std::unique_lock lock(nodes_mutex_);
        auto& node = nodes_[id];
        node.id = id;
        node.address = addr;
        node.port = port;
        node.last_heartbeat = std::chrono::steady_clock::now();
        node.healthy.store(true);
    }
    
    void update_load(const std::string& id, int64_t load) {
        std::shared_lock lock(nodes_mutex_);
        if (auto it = nodes_.find(id); it != nodes_.end()) {
            it->second.load.store(load);
            it->second.last_heartbeat = std::chrono::steady_clock::now();
        }
    }
    
    std::optional<std::pair<std::string, int>> select_node() {
        std::shared_lock lock(nodes_mutex_);
        
        std::string selected_id;
        double best_score = -1.0;
        
        for (const auto& [id, node] : nodes_) {
            if (!node.healthy.load()) continue;
            
            double load_factor = static_cast<double>(node.load.load()) / node.capacity.load();
            double score = 1.0 - load_factor;
            
            if (score > best_score) {
                best_score = score;
                selected_id = id;
            }
        }
        
        if (auto it = nodes_.find(selected_id); it != nodes_.end()) {
            return std::make_pair(it->second.address, it->second.port);
        }
        return std::nullopt;
    }
    
    std::vector<std::string> get_healthy_nodes() const {
        std::shared_lock lock(nodes_mutex_);
        std::vector<std::string> healthy;
        for (const auto& [id, node] : nodes_) {
            if (node.healthy.load()) {
                healthy.push_back(id);
            }
        }
        return healthy;
    }
};

class RateLimiter {
    struct TokenBucket {
        std::atomic<int64_t> tokens{0};
        std::chrono::steady_clock::time_point last_refill;
        int64_t capacity;
        int64_t refill_rate;
        std::chrono::milliseconds refill_interval;
    };
    
    std::unordered_map<std::string, TokenBucket> buckets_;
    mutable std::shared_mutex buckets_mutex_;
    
    void refill_bucket(TokenBucket& bucket) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - bucket.last_refill);
        
        int64_t new_tokens = (elapsed.count() * bucket.refill_rate) / 
                           bucket.refill_interval.count();
        
        if (new_tokens > 0) {
            int64_t current = bucket.tokens.load();
            int64_t desired = std::min(current + new_tokens, bucket.capacity);
            while (!bucket.tokens.compare_exchange_weak(current, desired)) {
                desired = std::min(current + new_tokens, bucket.capacity);
            }
            bucket.last_refill = now;
        }
    }
    
public:
    bool try_acquire(const std::string& key, int64_t tokens = 1) {
        std::unique_lock lock(buckets_mutex_);
        auto& bucket = buckets_[key];
        
        refill_bucket(bucket);
        
        int64_t current = bucket.tokens.load();
        if (current < tokens) {
            return false;
        }
        
        while (!bucket.tokens.compare_exchange_weak(current, current - tokens)) {
            if (current < tokens) {
                return false;
            }
        }
        return true;
    }
    
    void configure_bucket(const std::string& key, int64_t capacity, 
                         int64_t refill_rate, std::chrono::milliseconds interval) {
        std::unique_lock lock(buckets_mutex_);
        auto& bucket = buckets_[key];
        bucket.capacity = capacity;
        bucket.refill_rate = refill_rate;
        bucket.refill_interval = interval;
        bucket.tokens.store(capacity);
        bucket.last_refill = std::chrono::steady_clock::now();
    }
};

template<typename T>
class ObjectPool {
    struct PooledObject {
        T object;
        std::chrono::steady_clock::time_point created_at;
        std::atomic<bool> in_use{false};
        
        bool try_acquire() {
            bool expected = false;
            return in_use.compare_exchange_strong(expected, true);
        }
        
        void release() {
            in_use.store(false);
        }
    };
    
    std::vector<std::unique_ptr<PooledObject>> objects_;
    moodycamel::ConcurrentQueue<PooledObject*> available_;
    std::function<std::unique_ptr<T>()> factory_;
    std::atomic<size_t> total_objects_{0};
    
public:
    explicit ObjectPool(std::function<std::unique_ptr<T>()> factory) 
        : factory_(std::move(factory)) {}
    
    class ScopedObject {
        PooledObject* obj_{nullptr};
        ObjectPool* pool_{nullptr};
        
    public:
        ScopedObject(PooledObject* obj, ObjectPool* pool) 
            : obj_(obj), pool_(pool) {}
        
        ~ScopedObject() {
            if (obj_ && pool_) {
                obj_->release();
                pool_->available_.enqueue(obj_);
            }
        }
        
        T& operator*() { return obj_->object; }
        T* operator->() { return &obj_->object; }
    };
    
    std::optional<ScopedObject> acquire() {
        PooledObject* obj = nullptr;
        if (available_.try_dequeue(obj)) {
            if (obj->try_acquire()) {
                return ScopedObject(obj, this);
            }
            available_.enqueue(obj);
        }
        
        std::unique_ptr<PooledObject> new_obj = std::make_unique<PooledObject>();
        new_obj->object = std::move(*factory_());
        new_obj->created_at = std::chrono::steady_clock::now();
        new_obj->in_use.store(true);
        
        obj = new_obj.get();
        objects_.push_back(std::move(new_obj));
        total_objects_.fetch_add(1);
        
        return ScopedObject(obj, this);
    }
    
    size_t size() const {
        return total_objects_.load();
    }
    
    size_t available_count() const {
        return available_.size_approx();
    }
};

class MemoryWatcher {
    std::atomic<size_t> high_water_mark_{0};
    std::atomic<size_t> current_usage_{0};
    std::atomic<bool> memory_pressure_{false};
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};
    
    size_t get_process_memory() const {
#ifdef __linux__
        std::ifstream statm("/proc/self/statm");
        if (statm.is_open()) {
            size_t pages;
            statm >> pages;
            return pages * sysconf(_SC_PAGESIZE);
        }
#endif
        return 0;
    }
    
    void monitor_loop() {
        while (running_.load(std::memory_order_relaxed)) {
            size_t usage = get_process_memory();
            current_usage_.store(usage);
            
            if (usage > high_water_mark_.load()) {
                high_water_mark_.store(usage);
            }
            
            size_t hwm = high_water_mark_.load();
            if (hwm > 0 && usage > hwm * 0.9) {
                memory_pressure_.store(true);
            } else {
                memory_pressure_.store(false);
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
public:
    MemoryWatcher() {
        monitor_thread_ = std::thread(&MemoryWatcher::monitor_loop, this);
    }
    
    ~MemoryWatcher() {
        running_.store(false);
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    
    bool under_pressure() const {
        return memory_pressure_.load();
    }
    
    size_t current_usage() const {
        return current_usage_.load();
    }
    
    size_t high_water_mark() const {
        return high_water_mark_.load();
    }
    
    void reset_high_water_mark() {
        high_water_mark_.store(current_usage_.load());
    }
};

template<typename Request, typename Response>
class RequestPipeline {
    struct PipelineStage {
        std::function<std::expected<Response, std::string>(const Request&)> process;
        std::string name;
        size_t max_concurrent;
        std::atomic<size_t> current_concurrent{0};
    };
    
    std::vector<PipelineStage> stages_;
    std::vector<std::thread> workers_;
    moodycamel::BlockingConcurrentQueue<std::pair<Request, size_t>> queue_;
    std::atomic<bool> running_{true};
    std::function<void(const Request&, const Response&)> completion_callback_;
    
    void worker_loop(size_t stage_index) {
        auto& stage = stages_[stage_index];
        
        while (running_.load(std::memory_order_relaxed)) {
            std::pair<Request, size_t> item;
            if (!queue_.wait_dequeue_timed(item, 100)) {
                continue;
            }
            
            auto& [request, next_stage] = item;
            
            if (stage.current_concurrent.load() >= stage.max_concurrent) {
                queue_.enqueue(std::move(item));
                std::this_thread::yield();
                continue;
            }
            
            stage.current_concurrent.fetch_add(1);
            auto result = stage.process(request);
            stage.current_concurrent.fetch_sub(1);
            
            if (result) {
                if (next_stage < stages_.size()) {
                    queue_.enqueue(std::make_pair(request, next_stage + 1));
                } else if (completion_callback_) {
                    completion_callback_(request, result.value());
                }
            }
        }
    }
    
public:
    void add_stage(std::function<std::expected<Response, std::string>(const Request&)> process,
                  const std::string& name, size_t max_concurrent) {
        stages_.push_back({std::move(process), name, max_concurrent});
    }
    
    void start(size_t worker_per_stage = 2) {
        for (size_t i = 0; i < stages_.size(); ++i) {
            for (size_t j = 0; j < worker_per_stage; ++j) {
                workers_.emplace_back(&RequestPipeline::worker_loop, this, i);
            }
        }
    }
    
    void stop() {
        running_.store(false);
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }
    
    std::future<std::expected<Response, std::string>> submit(Request request) {
        auto promise = std::make_shared<std::promise<std::expected<Response, std::string>>>();
        auto future = promise->get_future();
        
        queue_.enqueue(std::make_pair(std::move(request), 0));
        
        return future;
    }
    
    void set_completion_callback(std::function<void(const Request&, const Response&)> callback) {
        completion_callback_ = std::move(callback);
    }
    
    std::vector<size_t> get_stage_loads() const {
        std::vector<size_t> loads;
        for (const auto& stage : stages_) {
            loads.push_back(stage.current_concurrent.load());
        }
        return loads;
    }
};

class CircuitBreaker {
    enum class State { CLOSED, OPEN, HALF_OPEN };
    
    std::atomic<State> state_{State::CLOSED};
    std::atomic<int64_t> failure_count_{0};
    std::atomic<int64_t> success_count_{0};
    std::chrono::steady_clock::time_point last_failure_time_;
    std::chrono::milliseconds reset_timeout_;
    int64_t failure_threshold_;
    int64_t success_threshold_;
    mutable std::mutex state_mutex_;
    
public:
    CircuitBreaker(int64_t failure_thresh, int64_t success_thresh,
                  std::chrono::milliseconds reset_timeout)
        : reset_timeout_(reset_timeout)
        , failure_threshold_(failure_thresh)
        , success_threshold_(success_thresh) {}
    
    bool allow_request() {
        if (state_.load() == State::OPEN) {
            std::lock_guard lock(state_mutex_);
            auto now = std::chrono::steady_clock::now();
            if (now - last_failure_time_ > reset_timeout_) {
                state_.store(State::HALF_OPEN);
                return true;
            }
            return false;
        }
        return true;
    }
    
    void record_success() {
        if (state_.load() == State::HALF_OPEN) {
            success_count_.fetch_add(1);
            if (success_count_.load() >= success_threshold_) {
                std::lock_guard lock(state_mutex_);
                state_.store(State::CLOSED);
                failure_count_.store(0);
                success_count_.store(0);
            }
        }
    }
    
    void record_failure() {
        failure_count_.fetch_add(1);
        
        if (failure_count_.load() >= failure_threshold_) {
            std::lock_guard lock(state_mutex_);
            state_.store(State::OPEN);
            last_failure_time_ = std::chrono::steady_clock::now();
        }
    }
    
    State get_state() const {
        return state_.load();
    }
};

template<typename Key, typename Value, size_t MaxSize = 10000>
class LRUCache {
    struct Node {
        Key key;
        Value value;
        Node* prev{nullptr};
        Node* next{nullptr};
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::unordered_map<Key, Node*> map_;
    Node* head_{nullptr};
    Node* tail_{nullptr};
    mutable std::shared_mutex mutex_;
    size_t size_{0};
    OptimizedAllocator<Node> allocator_;
    
    void move_to_front(Node* node) {
        if (node == head_) return;
        
        if (node->prev) node->prev->next = node->next;
        if (node->next) node->next->prev = node->prev;
        
        if (node == tail_) {
            tail_ = node->prev;
        }
        
        node->prev = nullptr;
        node->next = head_;
        
        if (head_) {
            head_->prev = node;
        }
        
        head_ = node;
        
        if (!tail_) {
            tail_ = node;
        }
    }
    
    void evict_lru() {
        if (!tail_) return;
        
        Node* to_remove = tail_;
        map_.erase(to_remove->key);
        
        tail_ = tail_->prev;
        if (tail_) {
            tail_->next = nullptr;
        } else {
            head_ = nullptr;
        }
        
        allocator_.deallocate(to_remove, 1);
        --size_;
    }
    
public:
    std::optional<Value> get(const Key& key) {
        std::shared_lock lock(mutex_);
        
        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        
        Node* node = it->second;
        move_to_front(node);
        node->timestamp = std::chrono::steady_clock::now();
        
        return node->value;
    }
    
    void put(const Key& key, Value value) {
        std::unique_lock lock(mutex_);
        
        auto it = map_.find(key);
        if (it != map_.end()) {
            Node* node = it->second;
            node->value = std::move(value);
            node->timestamp = std::chrono::steady_clock::now();
            move_to_front(node);
            return;
        }
        
        if (size_ >= MaxSize) {
            evict_lru();
        }
        
        Node* new_node = allocator_.allocate(1);
        new (new_node) Node{key, std::move(value), nullptr, head_, std::chrono::steady_clock::now()};
        
        if (head_) {
            head_->prev = new_node;
        }
        head_ = new_node;
        
        if (!tail_) {
            tail_ = new_node;
        }
        
        map_[key] = new_node;
        ++size_;
    }
    
    size_t size() const {
        std::shared_lock lock(mutex_);
        return size_;
    }
    
    void cleanup_expired(std::chrono::seconds max_age) {
        std::unique_lock lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        Node* current = tail_;
        
        while (current) {
            if (now - current->timestamp > max_age) {
                Node* to_remove = current;
                current = current->prev;
                
                map_.erase(to_remove->key);
                
                if (to_remove->prev) {
                    to_remove->prev->next = to_remove->next;
                }
                if (to_remove->next) {
                    to_remove->next->prev = to_remove->prev;
                }
                
                if (to_remove == head_) {
                    head_ = to_remove->next;
                }
                if (to_remove == tail_) {
                    tail_ = to_remove->prev;
                }
                
                allocator_.deallocate(to_remove, 1);
                --size_;
            } else {
                break;
            }
        }
    }
};

}
