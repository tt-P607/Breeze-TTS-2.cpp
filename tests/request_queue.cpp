#include "execution.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <future>

using namespace breeze;
using namespace std::chrono_literals;

static void require(bool value, const char * message) {
    if (value) return;
    fprintf(stderr, "%s\n", message);
    std::exit(1);
}

struct Event {
    std::mutex mutex;
    std::condition_variable changed;
    bool ready = false;
    void signal() {
        std::lock_guard<std::mutex> lock(mutex);
        ready = true;
        changed.notify_all();
    }
    bool wait(std::chrono::milliseconds timeout = 2000ms) {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, timeout, [&] { return ready; });
    }
};

static void serial_barrier(RequestQueue & queue) {
    RequestExecution initial(queue, true, 1);
    Event serial_queued, serial_started, release_serial, parallel_queued, parallel_started;
    auto serial = std::async(std::launch::async, [&] {
        RequestExecution permit(queue, false, 1, {}, [&] { serial_queued.signal(); });
        require(permit.acquired(), "serial request was rejected");
        serial_started.signal();
        require(release_serial.wait(), "serial release timed out");
    });
    require(serial_queued.wait(), "serial request did not wait for active parallel work");
    auto parallel = std::async(std::launch::async, [&] {
        RequestExecution permit(queue, true, 1, {}, [&] { parallel_queued.signal(); });
        require(permit.acquired(), "parallel request was rejected");
        parallel_started.signal();
    });
    require(parallel_queued.wait(), "later parallel request bypassed the serial barrier");
    initial.release();
    require(serial_started.wait(), "queued serial request never started");
    require(!parallel_started.wait(50ms), "parallel work overlapped a serial request");
    release_serial.signal();
    serial.get();
    parallel.get();
    require(parallel_started.wait(), "parallel work did not resume after serial completion");
}

static void weighted_slots(RequestQueue & queue) {
    RequestExecution initial(queue, true, 1);
    Event big_queued, big_started, release_big, small_queued, small_started;
    auto big = std::async(std::launch::async, [&] {
        RequestExecution permit(queue, true, 2, {}, [&] { big_queued.signal(); });
        require(permit.acquired(), "two-slot request was rejected");
        big_started.signal();
        require(release_big.wait(), "two-slot request release timed out");
    });
    require(big_queued.wait(), "two-slot request ignored the slot limit");
    auto small = std::async(std::launch::async, [&] {
        RequestExecution permit(queue, true, 1, {}, [&] { small_queued.signal(); });
        small_started.signal();
    });
    require(small_queued.wait(), "smaller request bypassed an earlier two-slot request");
    initial.release();
    require(big_started.wait(), "two-slot request never started after slots became free");
    require(!small_started.wait(50ms), "slot capacity was exceeded");
    release_big.signal();
    big.get();
    small.get();
    require(small_started.wait(), "small request never resumed");
}

static void cancellation(RequestQueue & queue) {
    RequestExecution initial(queue, false, 1);
    std::atomic<bool> cancelled{false};
    Event queued;
    auto pending = std::async(std::launch::async, [&] {
        RequestExecution permit(queue, true, 1, [&] { return cancelled.load(); }, [&] { queued.signal(); });
        return permit.acquired();
    });
    require(queued.wait(), "cancel test did not enter the queue");
    cancelled = true;
    require(pending.wait_for(500ms) == std::future_status::ready, "cancelled waiter stayed blocked");
    require(!pending.get(), "cancelled request was admitted");
    initial.release();
    RequestExecution next(queue, false, 2);
    require(next.acquired(), "cancelled ticket or slots leaked into the next request");
}

int main() {
    BreezeModel model;
    GenScheduler scheduler;
    scheduler.start(model, 2);
    {
        RequestQueue queue(scheduler);
        {
            RequestExecution a(queue, true, 1), b(queue, true, 1);
            require(a.acquired() && b.acquired(), "parallel requests could not share slots");
        }
        serial_barrier(queue);
        weighted_slots(queue);
        cancellation(queue);
    }
    require(scheduler.try_acquire(2), "request permits leaked generation slots");
    scheduler.release(2);
    scheduler.stop();
    puts("request queue: FIFO, serial barrier, parallel slots and cancellation passed");
}
