#pragma once

#include "breeze/scheduler.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>

namespace breeze {

// All clients share one FIFO. Parallel work shares slots; serial work runs alone.
class RequestQueue {
public:
    explicit RequestQueue(GenScheduler & scheduler) : m_scheduler(scheduler) {}

private:
    friend class RequestExecution;

    bool acquire(const void * ticket, bool parallel, int cost,
                 const std::function<bool()> & cancelled, const std::function<void()> & queued) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_waiting.push_back(ticket);
        bool announced = false;
        for (;;) {
            if (cancelled && cancelled()) {
                m_waiting.erase(std::find(m_waiting.begin(), m_waiting.end(), ticket));
                m_changed.notify_all();
                return false;
            }
            if (m_waiting.front() == ticket && !m_serial &&
                (parallel || m_active == 0) && m_scheduler.try_acquire(cost)) {
                m_waiting.pop_front();
                ++m_active;
                m_serial = !parallel;
                m_changed.notify_all();
                return true;
            }
            if (!announced && queued) {
                announced = true;
                lock.unlock();
                queued();
                lock.lock();
            } else {
                // HTTP disconnects and WebSocket cancel messages do not signal this condition.
                m_changed.wait_for(lock, std::chrono::milliseconds(100));
            }
        }
    }

    void release(bool parallel, int cost) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_scheduler.release(cost);
        --m_active;
        if (!parallel) m_serial = false;
        m_changed.notify_all();
    }

    GenScheduler & m_scheduler;
    std::mutex m_mutex;
    std::condition_variable m_changed;
    std::deque<const void *> m_waiting;
    int m_active = 0;
    bool m_serial = false;
};

class RequestExecution {
public:
    RequestExecution(RequestQueue & queue, bool parallel, int cost,
                     const std::function<bool()> & cancelled = {}, const std::function<void()> & queued = {})
        : m_queue(queue), m_parallel(parallel), m_cost(cost),
          m_acquired(queue.acquire(this, parallel, cost, cancelled, queued)) {}

    ~RequestExecution() { release(); }
    RequestExecution(const RequestExecution &) = delete;
    RequestExecution & operator=(const RequestExecution &) = delete;

    bool acquired() const { return m_acquired; }
    void release() {
        if (!m_acquired) return;
        m_acquired = false;
        m_queue.release(m_parallel, m_cost);
    }

private:
    RequestQueue & m_queue;
    bool m_parallel;
    int m_cost;
    bool m_acquired;
};

} // namespace breeze
