#pragma once

#include "ephemeris/simulation.hpp"
#include "ephemeris/spk_exporter.hpp"
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace sf::ephemeris {

struct SimulationPublication {
    SystemSnapshot snapshot;
    SimulationClock clock;
    CommandResult result;
    double advance_ms = 0.0;
    double copy_ms = 0.0;
    double nbody_epsilon = 0.0;
};

struct TimelineRenderProgress {
    bool active = false;
    float progress = 0.0f;
    size_t steps_done = 0;
    size_t steps_total = 0;
    double sim_rate_yr_s = 0.0;
    bool completed = false;
    bool cancelled = false;
    std::string error;
};

class SimulationRunner {
public:
    ~SimulationRunner();
    void set_threaded(bool enabled);
    bool threaded() const noexcept { return worker_.joinable(); }
    const SystemSnapshot& snapshot() const noexcept { return threaded() ? front_.snapshot : simulation_.snapshot(); }
    const SimulationClock& clock() const noexcept { return threaded() ? front_.clock : simulation_.clock(); }
    double nbody_epsilon() const noexcept { return threaded() ? front_.nbody_epsilon : simulation_.nbody_epsilon(); }
    const SimulationPublication& publication() const noexcept { return front_; }

    CommandResult replace_system(LoadedSystem loaded);
    template<class Command> CommandResult apply(Command command) {
        return execute([command = std::move(command)](Simulation& simulation) { return simulation.apply(command); });
    }
    CommandResult advance(units::Seconds elapsed);
    CommandResult advance_async(units::Seconds elapsed);
    CommandResult poll();
    CommandResult wait();

    bool is_rendering_timeline() const noexcept { return timeline_active_.load(std::memory_order_relaxed); }
    TimelineRenderProgress timeline_progress() const;
    void start_timeline_render(units::Years target_time, size_t requested_steps = 0);
    void cancel_timeline_render();
    std::shared_ptr<TimelineRecording> take_completed_timeline();
    void apply_timeline_frame(const TimelineRecording& rec, size_t frame_index);

private:
    CommandResult execute(std::function<CommandResult(Simulation&)> command);
    CommandResult submit(units::Seconds elapsed);
    void adopt_locked();
    void work();
    void stop() noexcept;

    Simulation simulation_;
    SimulationPublication front_, ready_, back_;
    std::mutex mutex_;
    std::condition_variable work_ready_, completed_;
    std::thread worker_;
    std::function<CommandResult(Simulation&)> command_;
    CommandResult command_result_;
    std::exception_ptr failure_;
    double pending_seconds_ = 0.0;
    uint64_t submitted_id_ = 0, completed_id_ = 0;
    bool pending_ = false, ready_available_ = false, stopping_ = false;

    std::thread timeline_worker_;
    mutable std::mutex timeline_mutex_;
    std::atomic<bool> timeline_active_{false};
    std::atomic<bool> timeline_cancel_{false};
    std::atomic<bool> timeline_completed_{false};
    std::atomic<float> timeline_progress_{0.0f};
    std::atomic<size_t> timeline_steps_done_{0};
    std::atomic<size_t> timeline_steps_total_{0};
    std::atomic<double> timeline_sim_rate_{0.0};
    std::string timeline_error_;
    std::shared_ptr<TimelineRecording> completed_timeline_;
};

}
