#include "models.h"

#include "breeze/model.h"
#include "breeze/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace breeze {

Models::Lease::Lease(Models & owner, std::string path, std::string slot, bool base)
    : m_owner(&owner), m_path(std::move(path)), m_slot(std::move(slot)), m_base(base) {}

Models::Lease::~Lease() {
    if (m_owner) m_owner->release(m_path, m_base);
}

Models::Models(BreezeModel & model, GenScheduler & sched, const std::string & base_id,
               const std::map<std::string, std::string> & loras, int capacity, int idle_timeout_seconds)
    : m_model(model), m_sched(sched), m_capacity(capacity),
      m_idle_timeout_seconds(idle_timeout_seconds), m_base_id(base_id), m_default_id(base_id) {
    m_registered.emplace(base_id, "");
    for (const auto & item : loras) {
        std::string path;
        if (!normalize_path(item.second, false, path))
            throw std::runtime_error("invalid startup LoRA path for model " + item.first);
        m_registered[item.first] = path;
    }
    m_reaper = std::thread([this] { reaper_loop(); });
}

Models::~Models() { stop(); }

bool Models::valid_id(const std::string & id) {
    if (id.empty()) return false;
    for (unsigned char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.'))
            return false;
    return true;
}

bool Models::normalize_path(const std::string & path, bool require_absolute, std::string & out) {
    if (path.empty()) return false;
    try {
        std::filesystem::path value = std::filesystem::u8path(path);
        if (require_absolute && !value.is_absolute()) return false;
        std::error_code ec;
        if (!value.is_absolute()) value = std::filesystem::absolute(value, ec);
        if (ec) return false;
        out = value.lexically_normal().u8string();
        return !out.empty();
    } catch (const std::exception &) {
        return false;
    }
}

std::string Models::escape_json(const std::string & value) {
    std::string out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof b, "\\u%04x", (unsigned) c);
                    out += b;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

bool Models::select(bool has_id, const std::string & id, bool has_path, const std::string & path,
                    Selection & out, int & status, std::string & error) const {
    status = 400;
    error.clear();
    out = Selection{};

    if (has_id && !valid_id(id)) {
        error = "invalid model_id";
        return false;
    }

    std::string normalized;
    if (has_path && !normalize_path(path, true, normalized)) {
        error = "lora_path must be a nonempty absolute path";
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (has_path) {
        if (has_id && id == m_base_id) {
            error = "the base model ID cannot name a LoRA adapter";
            return false;
        }
        out.id = has_id ? id : "";
        out.path = std::move(normalized);
        out.register_id = has_id;
        status = 200;
        return true;
    }

    out.id = has_id ? id : m_default_id;
    const auto it = m_registered.find(out.id);
    if (it == m_registered.end()) {
        status = 404;
        error = "unknown model_id";
        return false;
    }
    out.path = it->second;
    status = 200;
    return true;
}

std::shared_ptr<Models::Lease> Models::acquire(const Selection & selected, int & status, std::string & error) {
    status = 200;
    error.clear();
    std::lock_guard<std::mutex> lock(m_mutex);

    if (selected.path.empty()) {
        if (selected.id != m_base_id) {
            status = 404;
            error = "unknown model_id";
            return {};
        }
        ++m_base_users;
        return std::shared_ptr<Lease>(new Lease(*this, "", "", true));
    }

    auto found = m_cache.find(selected.path);
    if (found == m_cache.end()) {
        bool can_evict = m_cache.size() < (std::size_t) m_capacity;
        if (!can_evict) {
            for (const auto & entry : m_cache)
                if (entry.second.users == 0) { can_evict = true; break; }
        }
        if (!can_evict) {
            status = 409;
            error = "LoRA cache is full and every adapter is in use";
            return {};
        }

        const std::string slot = "__breeze_lora_" + std::to_string(m_next_slot++);
        bool loaded = false;
        m_sched.exec([&] { loaded = m_model.load_lora(slot, selected.path); });
        if (!loaded) {
            status = 422;
            error = "failed to load LoRA adapter";
            return {};
        }
        const auto now = std::chrono::steady_clock::now();
        found = m_cache.emplace(selected.path, Entry{slot, 1, now}).first;
        m_last_used[selected.path] = now;
        if (selected.register_id && !selected.id.empty()) m_registered[selected.id] = selected.path;
        evict_to_capacity_locked();
        printf("LoRA cache loaded: %s\n", selected.path.c_str());
        fflush(stdout);
    } else {
        ++found->second.users;
        if (selected.register_id && !selected.id.empty()) m_registered[selected.id] = selected.path;
    }

    return std::shared_ptr<Lease>(new Lease(*this, selected.path, found->second.slot, false));
}

bool Models::set_default(const std::string & id, int & status, std::string & error) {
    status = 200;
    error.clear();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_registered.find(id) == m_registered.end()) {
        status = 404;
        error = "unknown model_id";
        return false;
    }
    m_default_id = id;
    return true;
}

std::string Models::default_id() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_default_id;
}

std::string Models::json() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto now = std::chrono::steady_clock::now();
    std::string out = "{\"object\":\"list\",\"active_model\":\"" + escape_json(m_default_id) +
                      "\",\"lora_cache\":{\"capacity\":" + std::to_string(m_capacity) +
                      ",\"idle_timeout_seconds\":" + std::to_string(m_idle_timeout_seconds) +
                      ",\"loaded\":" + std::to_string(m_cache.size()) + "},\"data\":[";
    bool first = true;
    for (const auto & item : m_registered) {
        if (!first) out += ",";
        first = false;
        const bool base = item.first == m_base_id;
        const auto cache = m_cache.find(item.second);
        const bool loaded = base || cache != m_cache.end();
        const bool in_use = base ? m_base_users != 0 : cache != m_cache.end() && cache->second.users != 0;
        auto last = m_last_used.find(item.second);
        const auto last_used = cache != m_cache.end() ? cache->second.last_used :
                               (last != m_last_used.end() ? last->second : now);
        const auto idle = in_use ? 0 : std::max<int64_t>(0,
            std::chrono::duration_cast<std::chrono::seconds>(now - last_used).count());
        out += "{\"id\":\"" + escape_json(item.first) + "\",\"object\":\"model\",\"loaded\":" +
               (loaded ? "true" : "false") + ",\"in_use\":" + (in_use ? "true" : "false") +
               ",\"idle_seconds\":" + std::to_string(idle) + "}";
    }
    return out + "]}";
}

void Models::release(const std::string & path, bool base) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto now = std::chrono::steady_clock::now();
    if (base) {
        if (m_base_users > 0) --m_base_users;
        m_last_used[""] = now;
        return;
    }
    auto it = m_cache.find(path);
    if (it == m_cache.end()) return;
    if (it->second.users > 0) --it->second.users;
    it->second.last_used = now;
    m_last_used[path] = now;
}

void Models::unload_locked(const std::string & path, const Entry & entry) {
    m_sched.exec([&] { m_sched.unload_lora(entry.slot); });
    printf("LoRA cache unloaded: %s\n", path.c_str());
    fflush(stdout);
}

void Models::evict_to_capacity_locked() {
    while (m_cache.size() > (std::size_t) m_capacity) {
        auto victim = m_cache.end();
        for (auto it = m_cache.begin(); it != m_cache.end(); ++it) {
            if (it->second.users != 0) continue;
            if (victim == m_cache.end() || it->second.last_used < victim->second.last_used) victim = it;
        }
        if (victim == m_cache.end()) return;
        const std::string path = victim->first;
        unload_locked(path, victim->second);
        m_cache.erase(victim);
    }
}

void Models::reap_idle() {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(m_idle_timeout_seconds);
    for (auto it = m_cache.begin(); it != m_cache.end();) {
        if (it->second.users == 0 && now - it->second.last_used >= timeout) {
            const std::string path = it->first;
            unload_locked(path, it->second);
            it = m_cache.erase(it);
        } else {
            ++it;
        }
    }
}

void Models::reaper_loop() {
    const auto interval = std::chrono::seconds(std::min(m_idle_timeout_seconds, 30));
    std::unique_lock<std::mutex> lock(m_reaper_mutex);
    while (!m_stopping) {
        if (m_reaper_cv.wait_for(lock, interval, [this] { return m_stopping; })) break;
        lock.unlock();
        try {
            reap_idle();
        } catch (const std::exception & e) {
            fprintf(stderr, "LoRA reaper error: %s\n", e.what());
        }
        lock.lock();
    }
}

void Models::stop() {
    {
        std::lock_guard<std::mutex> lock(m_reaper_mutex);
        m_stopping = true;
    }
    m_reaper_cv.notify_all();
    if (m_reaper.joinable()) m_reaper.join();
}

} // namespace breeze
