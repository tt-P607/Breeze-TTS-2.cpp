#pragma once

#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace breeze {

class BreezeModel;
class GenScheduler;

class Models {
public:
    struct Selection {
        std::string id;
        std::string path;
        bool register_id = false;
    };

    class Lease {
    public:
        ~Lease();
        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;

        const std::string & slot() const { return m_slot; }

    private:
        friend class Models;
        Lease(Models & owner, std::string path, std::string slot, bool base);

        Models * m_owner;
        std::string m_path;
        std::string m_slot;
        bool m_base;
    };

    Models(BreezeModel & model, GenScheduler & sched, const std::string & base_id,
           const std::map<std::string, std::string> & loras, int capacity, int idle_timeout_seconds);
    ~Models();

    // Resolve a request without loading an adapter. A supplied ID and path are registered only
    // after acquire() has loaded the path successfully.
    bool select(bool has_id, const std::string & id, bool has_path, const std::string & path,
                Selection & out, int & status, std::string & error) const;
    std::shared_ptr<Lease> acquire(const Selection & selected, int & status, std::string & error);

    bool set_default(const std::string & id, int & status, std::string & error);
    std::string default_id() const;
    std::string json() const;

    void reap_idle();
    void stop();

private:
    struct Entry {
        std::string slot;
        std::size_t users = 0;
        std::chrono::steady_clock::time_point last_used;
    };

    static bool valid_id(const std::string & id);
    static bool normalize_path(const std::string & path, bool require_absolute, std::string & out);
    static std::string escape_json(const std::string & value);

    void release(const std::string & path, bool base);
    void evict_to_capacity_locked();
    void unload_locked(const std::string & path, const Entry & entry);
    void reaper_loop();

    BreezeModel & m_model;
    GenScheduler & m_sched;
    const int m_capacity;
    const int m_idle_timeout_seconds;

    mutable std::mutex m_mutex;
    std::map<std::string, std::string> m_registered; // model ID -> normalized path; base ID has an empty path
    std::map<std::string, Entry> m_cache;             // normalized path -> resident adapter
    std::map<std::string, std::chrono::steady_clock::time_point> m_last_used;
    std::string m_base_id;
    std::string m_default_id;
    std::size_t m_base_users = 0;
    std::uint64_t m_next_slot = 0;

    std::mutex m_reaper_mutex;
    std::condition_variable m_reaper_cv;
    bool m_stopping = false;
    std::thread m_reaper;
};

} // namespace breeze
