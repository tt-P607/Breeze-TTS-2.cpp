#pragma once

#include <mutex>
#include <shared_mutex>

namespace breeze {

// Serial requests take exclusive access. Parallel requests share access with other parallel ones.
class RequestExecution {
public:
    RequestExecution(std::shared_mutex & gate, bool parallel, bool wait = false)
        : m_parallel(parallel), m_exclusive(gate, std::defer_lock), m_shared(gate, std::defer_lock) {
        if (m_parallel) {
            if (wait) m_shared.lock();
            else (void) m_shared.try_lock();
        } else {
            if (wait) m_exclusive.lock();
            else (void) m_exclusive.try_lock();
        }
    }

    RequestExecution(const RequestExecution &) = delete;
    RequestExecution & operator=(const RequestExecution &) = delete;

    bool owns_lock() const { return m_parallel ? m_shared.owns_lock() : m_exclusive.owns_lock(); }

private:
    bool m_parallel;
    std::unique_lock<std::shared_mutex> m_exclusive;
    std::shared_lock<std::shared_mutex> m_shared;
};

} // namespace breeze
