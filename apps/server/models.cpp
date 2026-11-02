#include "models.h"
#include "json.h"

#include "breeze/model.h"
#include "breeze/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <cmath>
#include <cstring>
#include <set>

namespace breeze {

Models::Lease::Lease(Models & owner, std::vector<std::string> paths, std::string slot, bool base)
    : m_owner(&owner), m_paths(std::move(paths)), m_slot(std::move(slot)), m_base(base) {}

Models::Lease::~Lease() {
    if (m_owner) m_owner->release(m_paths, m_slot, m_base);
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

bool Models::configure(const JVal & body, Selection & selected, int & status, std::string & error) const {
    status = 400;
    const auto weight = [&](const JVal & object, const char * key, float & out) {
        const auto it = object.obj.find(key);
        if (it == object.obj.end()) return true;
        if (it->second.kind != JVal::Num || !std::isfinite(it->second.num) ||
            it->second.num < 0 || !std::isfinite((float) it->second.num)) {
            error = std::string(key) + " must be a finite nonnegative number";
            return false;
        }
        out = (float) it->second.num;
        return true;
    };
    selected.weight = 1.0f;
    selected.extra_loras.clear();
    if (!weight(body, "lora_weight", selected.weight)) return false;
    std::set<std::string> paths;
    if (!selected.path.empty()) paths.insert(selected.path);
    const auto extras = body.obj.find("extra_loras");
    if (extras != body.obj.end()) {
        if (extras->second.kind != JVal::Arr) {
            error = "extra_loras must be an array";
            return false;
        }
        for (const auto & item : extras->second.arr) {
            const auto path = item.obj.find("lora_path");
            Adapter adapter;
            if (item.kind != JVal::Obj || path == item.obj.end() || path->second.kind != JVal::Str ||
                !normalize_path(path->second.str, true, adapter.path)) {
                error = "each extra_loras item needs an absolute lora_path";
                return false;
            }
            if (!weight(item, "weight", adapter.weight)) return false;
            if (!paths.insert(adapter.path).second) {
                error = "a LoRA path may occur only once per request";
                return false;
            }
            selected.extra_loras.push_back(std::move(adapter));
        }
    }
    std::size_t count = !selected.path.empty() && selected.weight != 0 ? 1 : 0;
    for (const auto & adapter : selected.extra_loras) if (adapter.weight != 0) ++count;
    if (count > (std::size_t) m_capacity) {
        error = "request uses more LoRAs than --lora-cache-size allows";
        return false;
    }
    status = 200;
    error.clear();
    return true;
}

std::shared_ptr<Models::Lease> Models::acquire(const Selection & selected, int & status, std::string & error,
                                             const std::function<bool()> & cancelled) {
    status = 200;
    error.clear();
    std::unique_lock<std::mutex> lock(m_mutex);

    const auto aborted = [&] {
        if (!cancelled || !cancelled()) return false;
        status = 499;
        error = "request cancelled";
        return true;
    };
    if (aborted()) return {};

    std::map<std::string, float> requested;
    if (!selected.path.empty() && selected.weight != 0) requested.emplace(selected.path, selected.weight);
    for (const auto & adapter : selected.extra_loras)
        if (adapter.weight != 0) requested.emplace(adapter.path, adapter.weight);
    if (requested.size() > (std::size_t) m_capacity) {
        status = 400;
        error = "request uses more LoRAs than --lora-cache-size allows";
        return {};
    }
    if (requested.empty()) {
        ++m_base_users;
        return std::shared_ptr<Lease>(new Lease(*this, {}, "", true));
    }
    // Reserve the whole set together: two queued stacks must never each hold half their adapters.
    const auto fits = [&] {
        std::size_t busy_outside = 0;
        for (const auto & item : m_cache)
            if (item.second.users != 0 && !requested.count(item.first)) ++busy_outside;
        return busy_outside + requested.size() <= (std::size_t) m_capacity;
    };
    while (!fits()) {
        m_available.wait_for(lock, std::chrono::milliseconds(100));
        if (aborted()) return {};
    }
    std::size_t missing = 0;
    for (const auto & item : requested) if (!m_cache.count(item.first)) ++missing;
    while (m_cache.size() + missing > (std::size_t) m_capacity) {
        auto victim = m_cache.end();
        for (auto it = m_cache.begin(); it != m_cache.end(); ++it) {
            if (it->second.users || requested.count(it->first)) continue;
            if (victim == m_cache.end() || it->second.last_used < victim->second.last_used) victim = it;
        }
        unload_locked(victim->first, victim->second);
        m_cache.erase(victim);
    }
    const auto now = std::chrono::steady_clock::now();
    std::vector<std::string> paths;
    std::vector<LoraWeight> weights;
    std::string key;
    for (const auto & item : requested) {
        auto found = m_cache.find(item.first);
        if (found == m_cache.end()) {
            const std::string slot = "__breeze_lora_" + std::to_string(m_next_slot++);
            bool loaded = false;
            m_sched.exec([&] { loaded = m_model.load_lora(slot, item.first); });
            if (!loaded) {
                status = 422;
                error = "failed to load LoRA adapter";
                return {};
            }
            found = m_cache.emplace(item.first, Entry{slot, 0, now}).first;
            m_last_used[item.first] = now;
            printf("LoRA cache loaded: %s\n", item.first.c_str());
            fflush(stdout);
        }
        paths.push_back(item.first);
        weights.push_back({found->second.slot, item.second});
        uint32_t bits;
        std::memcpy(&bits, &item.second, sizeof bits);
        key += found->second.slot + ":" + std::to_string(bits) + ";";
    }
    std::string slot = weights.front().id;
    if (weights.size() != 1 || weights.front().weight != 1.0f) {
        auto stack = m_stacks.find(key);
        if (stack == m_stacks.end()) {
            slot = "__breeze_stack_" + std::to_string(m_next_slot++);
            m_sched.exec([&] { m_model.add_lora_stack(slot, weights); });
            stack = m_stacks.emplace(key, Stack{Entry{slot, 0, now}, paths}).first;
            printf("LoRA stack %s:", slot.c_str());
            for (const auto & item : weights) printf(" %s@%.6g", item.id.c_str(), item.weight);
            printf("\n");
            fflush(stdout);
        }
        slot = stack->second.cache.slot;
        ++stack->second.cache.users;
    }
    for (const auto & path : paths) ++m_cache.at(path).users;
    if (selected.register_id && !selected.id.empty() && m_cache.count(selected.path))
        m_registered[selected.id] = selected.path;
    trim_stacks_locked();
    return std::shared_ptr<Lease>(new Lease(*this, std::move(paths), slot, false));
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
                      ",\"loaded\":" + std::to_string(m_cache.size()) +
                      ",\"weighted_compositions\":" + std::to_string(m_stacks.size()) + "},\"data\":[";
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

void Models::release(const std::vector<std::string> & paths, const std::string & slot, bool base) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto now = std::chrono::steady_clock::now();
    if (base) {
        if (m_base_users > 0) --m_base_users;
        m_last_used[""] = now;
        return;
    }
    for (const auto & path : paths) {
        auto & entry = m_cache.at(path);
        --entry.users;
        entry.last_used = now;
        m_last_used[path] = now;
    }
    for (auto & item : m_stacks) if (item.second.cache.slot == slot) {
        --item.second.cache.users;
        item.second.cache.last_used = now;
        break;
    }
    trim_stacks_locked();
    m_available.notify_all();
}

void Models::unload_locked(const std::string & path, const Entry & entry) {
    for (auto it = m_stacks.begin(); it != m_stacks.end();) {
        if (std::find(it->second.paths.begin(), it->second.paths.end(), path) != it->second.paths.end()) {
            m_sched.exec([&] { m_sched.unload_lora(it->second.cache.slot); });
            it = m_stacks.erase(it);
        } else ++it;
    }
    m_sched.exec([&] { m_sched.unload_lora(entry.slot); });
    printf("LoRA cache unloaded: %s\n", path.c_str());
    fflush(stdout);
}

void Models::trim_stacks_locked() {
    while (m_stacks.size() > (std::size_t) m_capacity) {
        auto victim = m_stacks.end();
        for (auto it = m_stacks.begin(); it != m_stacks.end(); ++it) {
            if (it->second.cache.users != 0) continue;
            if (victim == m_stacks.end() || it->second.cache.last_used < victim->second.cache.last_used) victim = it;
        }
        if (victim == m_stacks.end()) return;
        m_sched.exec([&] { m_sched.unload_lora(victim->second.cache.slot); });
        m_stacks.erase(victim);
    }
}

void Models::reap_idle() {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(m_idle_timeout_seconds);
    for (auto it = m_stacks.begin(); it != m_stacks.end();) {
        if (it->second.cache.users == 0 && now - it->second.cache.last_used >= timeout) {
            m_sched.exec([&] { m_sched.unload_lora(it->second.cache.slot); });
            it = m_stacks.erase(it);
        } else ++it;
    }
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
