// Frame interpolation (RX_VK_FRAMEGEN=1; off by default): frames made between the game's, from its last two, shown in
// between by a display thread, so a game at 60 fps is shown at 120 (RX_VK_FRAMEGEN_MULT=3: 180). The WebGPU
// renderer's (gpu/render/rx_framegen.cpp, whose comments explain it), on Vulkan:
//   1. each frame is copied (it may be drawn over next frame) and reduced to quarter-size luma;
//   2. block matching: a vector per 16x16 pixels within +-32 pixels (RX_VK_FRAMEGEN_RADIUS quarter texels, default
//      8), then a 3x3 median of them;
//   3. each in-between frame at time t blends the last frame moved back along t and the new one forward along 1 - t.
// Recorded into the present's own recording, after the game's work; the display thread shows the made frames and
// then the game's, evenly over the game's recent frame time, about one frame behind the game. Loading hitches
// (frames over 100 ms apart) are not interpolated.
//
// (hookshot) With DLSS 5's layer and a present queue, the model runs on the game's frames before anything is made from them
// (the layer's vkDlssnrProcessImageHOOKSHOT, about 13 ms at 1440p): the replay thread only copies a frame into a ring slot, a
// worker runs the model on it and then the steps above, and presents go through the layer unprocessed. The made frames are
// interpolated between pictures already processed, so they carry the same look. RX_VK_DLSS5_BEFORE_FG=0: the model at the present.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#include "vk_framegen.h"
#include "vk_menu.h"
#include "vk_texture.h"

#include "shaders/fg_copy_comp_spv.h"
#include "shaders/fg_flow_comp_spv.h"
#include "shaders/fg_luma_comp_spv.h"
#include "shaders/fg_median_comp_spv.h"
#include "shaders/fg_scale_comp_spv.h"
#include "shaders/fg_synth_comp_spv.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION   // (older Windows SDKs lack it: Windows 10 1803 and later take it)
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace mkwvk {
extern std::atomic<uint32_t> present_kind;   // (vk_replay.cpp: the frame log's)
namespace {

constexpr int kMaxMult = 8;   // (up to 8x: players asked; the made frames are allocated for the multiplier chosen only)

struct Img {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;
};
struct OutSet {
    Img frame[kMaxMult];
    int made = 0;   // (frame[0..made) are made frames; frame[made] is the game's)
    bool busy = false;
};
struct Job {
    OutSet* set;
    int64_t at;
    uint64_t value;   // (the recording that makes its frames; with wait, that timeline's value)
    VkSemaphore wait = VK_NULL_HANDLE;   // (the worker path's: made on the other queue, waited for on the one that shows them)
};
enum RingState { kRingFree, kRingQueued, kRingHeld };   // (kRingHeld: the worker's last frame, kept to blend with the next)
struct RingSlot {
    Img frame;
    RingState state = kRingFree;
    bool gap = false;   // (a loading hitch came before it: nothing to blend)
    uint64_t wait = 0;   // (the copy into it is done at this value of the ring timeline)
};
constexpr int kRingSlots = 4;   // (one kept as the worker's previous frame, one being processed, two waiting)

struct Pipe {
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
};

struct State {
    bool init = false, failed = false;
    uint32_t w = 0, h = 0;
    Pipe copy, luma, flow, median, synth;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    Img frames[2], lumas[2], flows[2];
    OutSet sets[3];   // (one shown, one waiting its place on the line, one being made: the pacer never holds the game back)
    int cur = 0, next_set = 0;
    bool have_prev = false;
    int64_t last_at = 0;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    bool thread_started = false;
    RingSlot ring[kRingSlots];   // (the worker path's frames, in place of frames[2]; ensure_size)
    int ring_next = 0, ring_prev = -1, ring_recorded = -1;   // (ring_recorded: filled by framegen_record, for framegen_queue)
    std::deque<int> work;   // (the worker's slots, oldest first)
    bool worker = false, worker_started = false, worker_busy = false;
    bool worker_latched = false;   // (it has run: the layer no longer processes presents, so it goes on at 1x too)
    bool prev_raw = false;   // (the kept frame is the game's, the model having failed on it: a raw frame and a processed one are not blended)
    VkSemaphore ring_tl = VK_NULL_HANDLE;   // (a timeline: each copy into a ring slot done)
    uint64_t ring_value = 0;   // (the replay's: only it signals)
    VkSemaphore out_tl = VK_NULL_HANDLE;   // (a timeline: the worker's frames done, for the display thread's submit on the other queue)
    uint64_t out_value = 0;   // (the worker's)
    OutSet* recorded = nullptr;   // (the set the last framegen_record filled, for framegen_queue)
    int64_t recorded_at = 0;
    int mult = 0;   // (the multiplier the sets' frames were made for)
} fg;

// Used only by present_image, whose lock serializes direct and interpolated presents.
struct ScaleState {
    bool init = false, failed = false;
    Pipe pipe;
    VkSampler linear = VK_NULL_HANDLE;
    Img cropped, scaled;
    uint32_t cropped_w = 0, cropped_h = 0, scaled_w = 0, scaled_h = 0;
} scale;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int multiplier() { return std::clamp(opts.framegen, 1, kMaxMult); }   // (the Insert menu; 1: the game's frames alone)

bool framegen_allowed() {   // (RX_VK_FRAMEGEN=0: none made, the menu's multiplier aside)
    static const bool allow = !getenv("RX_VK_FRAMEGEN") || atoi(getenv("RX_VK_FRAMEGEN")) != 0;
    return allow;
}

/// The worker path (the layer's model on the game's frames, on dev.present_queue), not available with the shots of made frames
/// (RX_VK_SHOT_MADE reads them in the present's recording)
bool worker_wanted() {
    if (!dev.dlss5_process || !dev.present_queue || getenv("RX_VK_SHOT_MADE")) return false;
    if (multiplier() >= 2 && framegen_allowed()) fg.worker_latched = true;
    return fg.worker_latched;
}

/// A compute pipeline over bindings of these types (push descriptors), push constants of push bytes
bool make_pipe(Pipe& p, const uint32_t* code, size_t bytes, std::initializer_list<VkDescriptorType> types, uint32_t push,
               const VkSpecializationInfo* spec = nullptr) {
    std::vector<VkDescriptorSetLayoutBinding> b;
    for (VkDescriptorType t : types) b.push_back({(uint32_t)b.size(), t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = (uint32_t)b.size();
    li.pBindings = b.data();
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &p.set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, std::max(push, 4u)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &p.set;
    pi.pushConstantRangeCount = push ? 1 : 0;
    pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &p.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = bytes;
    mi.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev.device, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = spec;
    ci.layout = p.layout;
    const VkResult r = vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe);
    vkDestroyShaderModule(dev.device, m, nullptr);
    return r == VK_SUCCESS;
}

bool init() {
    if (fg.init) return !fg.failed;
    fg.init = true;
    fg.failed = true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const int radius = getenv("RX_VK_FRAMEGEN_RADIUS") ? std::clamp(atoi(getenv("RX_VK_FRAMEGEN_RADIUS")), 1, 16) : 8;
    const VkSpecializationMapEntry me{0, 0, 4};
    const VkSpecializationInfo spec{1, &me, 4, &radius};
    if (!make_pipe(fg.copy, FG_COPY_COMP_SPV, sizeof FG_COPY_COMP_SPV, {S, I}, 16) ||
        !make_pipe(fg.luma, FG_LUMA_COMP_SPV, sizeof FG_LUMA_COMP_SPV, {S, I}, 0) ||
        !make_pipe(fg.flow, FG_FLOW_COMP_SPV, sizeof FG_FLOW_COMP_SPV, {S, S, I}, 0, &spec) ||
        !make_pipe(fg.median, FG_MEDIAN_COMP_SPV, sizeof FG_MEDIAN_COMP_SPV, {S, I}, 0) ||
        !make_pipe(fg.synth, FG_SYNTH_COMP_SPV, sizeof FG_SYNTH_COMP_SPV, {S, S, S, I}, 4))
        return false;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.25f;
    vkCreateSampler(dev.device, &si, nullptr, &fg.nearest);
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(dev.device, &si, nullptr, &fg.linear);
    fg.failed = false;
    logf("vk: frame interpolation ready: %dx the game's frames (made from its last two, shown evenly)", multiplier());
    return true;
}

bool init_scale() {
    if (scale.init) return !scale.failed;
    scale.init = scale.failed = true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return false;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (make_pipe(scale.pipe, FG_SCALE_COMP_SPV, sizeof FG_SCALE_COMP_SPV, {S, I}, 8)) {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.25f;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        if (vkCreateSampler(dev.device, &si, nullptr, &scale.linear) == VK_SUCCESS) {
            scale.failed = false;
            return true;
        }
    }
    if (scale.pipe.pipe) vkDestroyPipeline(dev.device, scale.pipe.pipe, nullptr);
    if (scale.pipe.layout) vkDestroyPipelineLayout(dev.device, scale.pipe.layout, nullptr);
    if (scale.pipe.set) vkDestroyDescriptorSetLayout(dev.device, scale.pipe.set, nullptr);
    scale.pipe = {};
    return false;
}

bool make_img(Img& i, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = f;
    ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = usage;
    if (!create_image(ci, i.image, i.alloc)) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = i.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = f;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(dev.device, &vi, nullptr, &i.view) == VK_SUCCESS;
}
void drop_img(Img& i) {
    if (i.view) vkDestroyImageView(dev.device, i.view, nullptr);
    if (i.image) destroy_image(i.image, i.alloc);
    i = {};
}

void to_general(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

bool ensure_size(VkCommandBuffer cmd, uint32_t w, uint32_t h) {
    const bool worker = worker_wanted();
    if (fg.w == w && fg.h == h && fg.mult == multiplier() && fg.worker == worker) return true;
    {   // (the worker and the display thread are done with the old images first)
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] {
            return fg.work.empty() && !fg.worker_busy && fg.jobs.empty() && !fg.sets[0].busy && !fg.sets[1].busy && !fg.sets[2].busy;
        });
    }
    queue_flush();
    {
        std::lock_guard qk(dev.present_mu);   // (every queue idle: the present queue's too, device_wait_idle_locked)
        device_wait_idle_locked();
    }
    for (int i = 0; i < 2; i++) {
        drop_img(fg.frames[i]), drop_img(fg.lumas[i]), drop_img(fg.flows[i]);
    }
    for (RingSlot& s : fg.ring) drop_img(s.frame), s.state = kRingFree;
    fg.ring_next = 0, fg.ring_prev = -1;
    fg.worker = worker;
    for (OutSet& set : fg.sets)
        for (Img& t : set.frame) drop_img(t);
    fg.w = w, fg.h = h;
    const uint32_t qw = (w + 3) / 4, qh = (h + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bool ok = true;
    for (int i = 0; i < 2; i++) {
        if (!worker) ok = ok && make_img(fg.frames[i], w, h, VK_FORMAT_R8G8B8A8_UNORM, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        ok = ok && make_img(fg.lumas[i], qw, qh, VK_FORMAT_R16_SFLOAT, rw);
        ok = ok && make_img(fg.flows[i], cw, ch, VK_FORMAT_R16G16B16A16_SFLOAT, rw);
    }
    if (worker) {   // (the model writes its answer back into the slot: TRANSFER_DST)
        for (RingSlot& s : fg.ring)
            ok = ok && make_img(s.frame, w, h, VK_FORMAT_R8G8B8A8_UNORM, rw | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (!fg.ring_tl) {
            VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            sci.pNext = &tci;
            ok = ok && vkCreateSemaphore(dev.device, &sci, nullptr, &fg.ring_tl) == VK_SUCCESS;
            ok = ok && vkCreateSemaphore(dev.device, &sci, nullptr, &fg.out_tl) == VK_SUCCESS;
        }
    }
    fg.mult = multiplier();
    for (OutSet& set : fg.sets)
        for (int k = 0; k < fg.mult; k++)
            ok = ok && make_img(set.frame[k], w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                rw | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!ok) {
        fg.failed = true;
        logf("vk: frame interpolation: its images (%ux%u) were not made: off", w, h);
        return false;
    }
    for (int i = 0; i < 2; i++) {
        if (!worker) to_general(cmd, fg.frames[i].image);
        to_general(cmd, fg.lumas[i].image), to_general(cmd, fg.flows[i].image);
    }
    if (worker)
        for (RingSlot& s : fg.ring) to_general(cmd, s.frame.image);
    for (OutSet& set : fg.sets)
        for (Img& t : set.frame)
            if (t.image) to_general(cmd, t.image);   // (only the multiplier's frames are made: a barrier on no image crashed the driver)
    fg.have_prev = false;
    return true;
}

/// One pass: pipe over w x h in 8x8 groups, the bindings in order (a sampler when one is given: a sampled image)
void pass(VkCommandBuffer cmd, const Pipe& p, std::initializer_list<std::pair<VkImageView, VkSampler>> bindings,
          uint32_t w, uint32_t h, const void* push = nullptr, uint32_t push_bytes = 0) {
    VkDescriptorImageInfo ii[8];
    VkWriteDescriptorSet ws[8];
    uint32_t n = 0;
    for (const auto& [view, sampler] : bindings) {
        ii[n] = {sampler, view, VK_IMAGE_LAYOUT_GENERAL};
        ws[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        ws[n].dstBinding = n;
        ws[n].descriptorCount = 1;
        ws[n].descriptorType = sampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ws[n].pImageInfo = &ii[n];
        n++;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, n, ws);
    if (push_bytes) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cmd);
}

// (hookshot) The display thread's wait for a frame's time, on a high-resolution timer. sleep_for rests on the system timer,
// whose 1 ms (timeBeginPeriod) Windows 11 does not keep for a process whose window it takes as unseen: with the presents
// no longer blocking (a present queue of their own) the thread slept before most frames and woke up to 20 ms after their time.
void wait_until_near(int64_t ns) {
    static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    LARGE_INTEGER d;
    d.QuadPart = -std::max<int64_t>(1, ns / 100);   // (relative, 100 ns units)
    if (timer && SetWaitableTimerEx(timer, &d, 0, nullptr, nullptr, nullptr, 0)) {
        WaitForSingleObject(timer, INFINITE);
        return;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
}

void display_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk display");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    int64_t interval = 16666667, last_at = 0;
    struct { int on_time, late, skipped; int64_t since; } stat{};
    int64_t next_show = 0, shown_at = 0;
    int hist[6] = {};
    for (;;) {
        Job j;
        {
            std::unique_lock lk(fg.mu);
            fg.cv.wait(lk, [] { return !fg.jobs.empty(); });
            j = fg.jobs.front();
            fg.jobs.pop_front();
        }
        if (last_at) {
            const int64_t d = j.at - last_at;
            if (d > 0 && d < 100000000) interval = (interval * 7 + std::clamp<int64_t>(d, 4000000, 50000000)) / 8;
        }
        last_at = j.at;
        if (!stat.since) stat.since = j.at;
        if (j.at - stat.since > 5000000000LL) {
            logf("vk: frame interpolation: %d frames shown on time, %d late (over 2 ms); shown apart: <7 ms %d, 7-9 %d, 9-12 %d, "
                 "12-17 %d, 17-25 %d, over 25 %d", stat.on_time, stat.late, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
            memset(hist, 0, sizeof hist);
            stat = {0, 0, 0, j.at};
        }
        // (hookshot) The pacer. The game hands over frames 12 to 28 ms apart around a mean of 19, and shown as they came that
        // was the stutter, whatever the frame rate. Every frame now has its place on an even line, the recent frame time
        // apart, a frame early waiting for it: never more than one game frame behind what the game handed over, and a line
        // that has fallen behind what came starts again from now. RX_VK_PACE=0: as they come.
        const bool pace = opts.pace != 0;
        const int shown = j.set->made + 1;
        const int64_t step = interval / shown;
        for (int k = 0; k <= j.set->made; k++) {
            int64_t due = pace ? next_show : 0;
            const int64_t now = now_ns();
            if (due < now) due = now;
            if (due > j.at + interval + step) due = j.at + interval + step;
            if (due - now > 1000000) wait_until_near(due - now - 1000000);
            while (now_ns() < due) SwitchToThread();
            const int64_t t = now_ns();
            present_kind.store(k < j.set->made ? 1 : 0);
            present_image(j.set->frame[k].image, fg.w, fg.h, nullptr, j.set->frame[k].view, j.wait, j.value);
            menu_note_frame(k == j.set->made);
            if (shown_at) {
                const int64_t d = t - shown_at;
                hist[d < 7000000 ? 0 : d < 9000000 ? 1 : d < 12000000 ? 2 : d < 17000000 ? 3 : d < 25000000 ? 4 : 5]++;
            }
            shown_at = t;
            next_show = t + step;
            (t - due > 2000000 ? stat.late : stat.on_time)++;
        }
        std::lock_guard lk(fg.mu);
        j.set->busy = false;
        fg.cv.notify_all();
    }
}

/// The worker path's thread: the model on the game's frame, then the frames made from it, all on dev.present_queue. It holds the
/// game back only by the ring filling up
void worker_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk dlss5");
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = dev.queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(dev.device, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.commandBufferCount = 1;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev.device, &ai, &cmd);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(dev.device, &fi, nullptr, &fence);
    logf("vk: DLSS 5 before frame generation: running (the model on the game's frames, frames made after it)");
    menu_set_dlss5_live();
    int refused = 0;
    bool said_ok = false, was_wanted = true;
    for (;;) {
        int i;
        OutSet* set;
        {   // (a set to fill too: display_main's, which holds this thread back the way it held the replay's)
            std::unique_lock lk(fg.mu);
            fg.cv.wait(lk, [] { return !fg.work.empty() && !fg.sets[fg.next_set].busy; });
            i = fg.work.front();
            fg.work.pop_front();
            fg.worker_busy = true;
            set = &fg.sets[fg.next_set];
            set->busy = true;
            fg.next_set = (fg.next_set + 1) % 3;
        }
        RingSlot& s = fg.ring[i];
        // (the copy's submit went out before this slot was queued, framegen_queue. Waited here, and again on the queue for its
        // memory: two queues, and no submission order between them to lean on)
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &fg.ring_tl;
        wi.pValues = &s.wait;
        vkWaitSemaphores(dev.device, &wi, UINT64_MAX);
        const DlssnrProcessImageInfoHOOKSHOT info{s.frame.image, VK_FORMAT_R8G8B8A8_UNORM, fg.w, fg.h, VK_IMAGE_LAYOUT_GENERAL, fg.ring_tl, s.wait};
        // (the menu's row, read each frame: off, the frame goes on raw, as when the layer refuses one)
        const bool wanted = opts.dlss5 != 0;
        if (wanted != was_wanted) was_wanted = wanted, logf("vk: DLSS 5 %s from the menu", wanted ? "on" : "off");
        const VkResult res = wanted ? dev.dlss5_process(dev.device, dev.present_queue, &info) : VK_NOT_READY;
        const bool raw = res != VK_SUCCESS;
        if (raw && wanted && refused++ < 8) logf("vk: DLSS 5 before frame generation: the layer gave %d for a frame: it goes on raw", (int)res);
        if (!raw && !said_ok) said_ok = true, logf("vk: DLSS 5 before frame generation: first frame processed (%ux%u)", fg.w, fg.h);

        const int m = fg.mult, cur = fg.cur, prev = cur ^ 1;   // (fg.mult: what the sets were made for, the menu may have moved)
        const bool blend = fg.have_prev && !s.gap && raw == fg.prev_raw && fg.ring_prev >= 0;
        const uint32_t uw = fg.w, uh = fg.h, qw = (uw + 3) / 4, qh = (uh + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
        vkResetCommandPool(dev.device, pool, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        full_barrier(cmd);   // (what the replay's copy and the model wrote, on the other queue or earlier on this one)
        pass(cmd, fg.luma, {{s.frame.view, fg.nearest}, {fg.lumas[cur].view, VK_NULL_HANDLE}}, qw, qh);
        set->made = 0;
        if (blend && m > 1) {
            const Img& pf = fg.ring[fg.ring_prev].frame;
            pass(cmd, fg.flow, {{fg.lumas[prev].view, fg.nearest}, {fg.lumas[cur].view, fg.nearest},
                                {fg.flows[0].view, VK_NULL_HANDLE}}, cw, ch);
            pass(cmd, fg.median, {{fg.flows[0].view, fg.nearest}, {fg.flows[1].view, VK_NULL_HANDLE}}, cw, ch);
            for (int k = 1; k < m; k++) {
                const float t = (float)k / (float)m;
                pass(cmd, fg.synth, {{pf.view, fg.linear}, {s.frame.view, fg.linear},
                                     {fg.flows[1].view, fg.nearest}, {set->frame[k - 1].view, VK_NULL_HANDLE}},
                     uw, uh, &t, 4);
            }
            set->made = m - 1;
        }
        const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {uw, uh, 1}};
        vkCmdCopyImage(cmd, s.frame.image, VK_IMAGE_LAYOUT_GENERAL, set->frame[set->made].image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        full_barrier(cmd);
        vkEndCommandBuffer(cmd);
        VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait.semaphore = fg.ring_tl;   // (the copy, on the other queue: an order of its own, as the frame's model may have been refused before it submitted)
        wait.value = s.wait;
        wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cbi.commandBuffer = cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.waitSemaphoreInfoCount = 1;
        si.pWaitSemaphoreInfos = &wait;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        VkSemaphoreSubmitInfo done{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        done.semaphore = fg.out_tl;
        done.value = ++fg.out_value;
        done.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &done;
        {
            std::lock_guard pq(dev.present_queue_mu);
            vkQueueSubmit2(dev.present_queue, 1, &si, fence);
        }
        vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(dev.device, 1, &fence);
        // (stamped now: the display thread paces from when the frame is ready, and one stamped when the replay copied it, the
        // model's time ago, ran into the clamp at once)
        const int64_t at = now_ns();
        std::lock_guard lk(fg.mu);
        fg.have_prev = true;
        fg.prev_raw = raw;
        fg.cur = prev;
        if (fg.ring_prev >= 0) fg.ring[fg.ring_prev].state = kRingFree;
        fg.ring_prev = i;
        s.state = kRingHeld;
        fg.worker_busy = false;
        fg.jobs.push_back({set, at, done.value, fg.out_tl});
        fg.cv.notify_all();
    }
}

} // namespace

// The Insert menu's Frame generation is off at 0. While it was ignored, every present still copied the frame and
// reduced it to luma (framegen_record), then a second thread showed it. Open areas paid for that on a full GPU and
// gained no made frames. 2x and up is the interpolation; RX_VK_FRAMEGEN=0 keeps it off entirely.
bool framegen_wanted(int menu_value, int allow) {
    return allow && menu_value >= 2;
}

bool framegen_on() {
    // (with a present queue and no frames to make, the same thread shows the game's frames, so what a Vulkan layer does in the
    // present, DLSS 5's, does not hold the game back)
    return (framegen_wanted(opts.framegen, framegen_allowed()) || dev.present_queue) && !dev.offscreen && !fg.failed;
}

VkImage framegen_scale(VkCommandBuffer cmd, VkImageView src, VkImage image, uint32_t sw, uint32_t sh, const int crop[4], uint32_t dw, uint32_t dh) {
    if (!opts.upscale && !opts.sharpness) return VK_NULL_HANDLE;
    if (!src || !image || !sw || !sh || !dw || !dh) return VK_NULL_HANDLE;
    if (crop && crop[2] > 0 && crop[3] > 0 &&
        (crop[0] < 0 || crop[1] < 0 || (uint32_t)crop[0] >= sw || (uint32_t)crop[1] >= sh)) return VK_NULL_HANDLE;
    if (!init_scale()) return VK_NULL_HANDLE;
    // (the game draws into part of its window textures, 2048 x 1152 of 2560 x 1440 here: the filter was given the whole texture, and the
    // picture came out in the top left corner of the window with black around it. The crop first, then the filter)
    if (crop && crop[2] > 0 && crop[3] > 0 && (crop[0] || crop[1] || (uint32_t)crop[2] < sw || (uint32_t)crop[3] < sh)) {
        const uint32_t cw = std::min((uint32_t)crop[2], sw - (uint32_t)crop[0]), ch = std::min((uint32_t)crop[3], sh - (uint32_t)crop[1]);
        if (scale.cropped_w != cw || scale.cropped_h != ch) {
            {
                std::lock_guard qk(dev.present_mu);
                device_wait_idle_locked();
            }
            drop_img(scale.cropped);
            scale.cropped_w = scale.cropped_h = 0;
            if (!make_img(scale.cropped, cw, ch, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
                drop_img(scale.cropped);
                return VK_NULL_HANDLE;
            }
            scale.cropped_w = cw, scale.cropped_h = ch;
            to_general(cmd, scale.cropped.image);
        }
        VkImageBlit b{};
        b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {crop[0], crop[1], 0};
        b.srcOffsets[1] = {crop[0] + (int)cw, crop[1] + (int)ch, 1};
        b.dstOffsets[1] = {(int)cw, (int)ch, 1};
        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, scale.cropped.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
        full_barrier(cmd);
        src = scale.cropped.view;
    }
    if (scale.scaled_w != dw || scale.scaled_h != dh) {
        {
            std::lock_guard qk(dev.present_mu);   // (the queue is idle only while nothing else submits to it)
            device_wait_idle_locked();
        }
        drop_img(scale.scaled);
        scale.scaled_w = scale.scaled_h = 0;
        if (!make_img(scale.scaled, dw, dh, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            drop_img(scale.scaled);
            return VK_NULL_HANDLE;
        }
        scale.scaled_w = dw, scale.scaled_h = dh;
        to_general(cmd, scale.scaled.image);
    }
    const struct { int mode; float sharp; } pc{opts.upscale, opts.sharpness / 100.0f};
    pass(cmd, scale.pipe, {{src, scale.linear}, {scale.scaled.view, VK_NULL_HANDLE}}, dw, dh, &pc, sizeof pc);
    static const bool report = getenv("RX_VK_UPSCALE_LOG") != nullptr;
    static bool reported = false;
    if (report && !reported) {
        reported = true;
        logf("vk: upscale dispatched: mode %d, sharpness %d, source %ux%u, crop request %d,%d %dx%d, destination %ux%u",
             opts.upscale, opts.sharpness, sw, sh, crop ? crop[0] : 0, crop ? crop[1] : 0,
             crop ? crop[2] : (int)sw, crop ? crop[3] : (int)sh, dw, dh);
    }
    return scale.scaled.image;
}

VkImage framegen_made_image(uint32_t* w, uint32_t* h) {
    if (!fg.recorded || !fg.recorded->made) return VK_NULL_HANDLE;
    *w = fg.w, *h = fg.h;
    return fg.recorded->frame[0].image;
}

bool framegen_record(VkCommandBuffer cmd, VkImageView src, uint32_t sw, uint32_t sh, const int crop[4]) {
    int x0 = 0, y0 = 0, w = (int)sw, h = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], w = crop[2], h = crop[3];
    w = std::min(w, (int)sw - x0), h = std::min(h, (int)sh - y0);
    if (!src || w <= 0 || h <= 0 || !init() || !ensure_size(cmd, (uint32_t)w, (uint32_t)h)) return false;
    if (!fg.thread_started) {
        fg.thread_started = true;
        std::thread(display_main).detach();
    }
    if (fg.worker && !fg.worker_started) {
        fg.worker_started = true;
        std::thread(worker_main).detach();
    }
    const int64_t now = now_ns();
    const bool gap = fg.last_at && now - fg.last_at > 100000000;   // (a loading hitch: nothing to blend)
    fg.last_at = now;
    if (fg.worker) {   // (only the copy: it waits for a free slot, and the worker does the rest after the submit)
        int i;
        {
            std::unique_lock lk(fg.mu);
            fg.cv.wait(lk, [] { return fg.ring[fg.ring_next].state == kRingFree; });
            i = fg.ring_next;
            fg.ring[i].state = kRingQueued;
            fg.ring_next = (fg.ring_next + 1) % kRingSlots;
        }
        fg.ring[i].gap = gap;
        full_barrier(cmd);
        const int32_t rpc[4] = {x0, y0, w, h};
        pass(cmd, fg.copy, {{src, fg.nearest}, {fg.ring[i].frame.view, VK_NULL_HANDLE}}, (uint32_t)w, (uint32_t)h, rpc, sizeof rpc);
        fg.ring_recorded = i;
        return true;
    }
    OutSet* set;
    {
        std::unique_lock lk(fg.mu);
        fg.cv.wait(lk, [] { return !fg.sets[fg.next_set].busy; });
        set = &fg.sets[fg.next_set];
        set->busy = true;
        fg.next_set = (fg.next_set + 1) % 3;
    }
    const int cur = fg.cur, prev = cur ^ 1, m = multiplier();
    const bool blend = fg.have_prev && !gap;
    const uint32_t uw = (uint32_t)w, uh = (uint32_t)h;
    const uint32_t qw = (uw + 3) / 4, qh = (uh + 3) / 4, cw = (qw + 3) / 4, ch = (qh + 3) / 4;
    full_barrier(cmd);
    const int32_t pc[4] = {x0, y0, w, h};
    pass(cmd, fg.copy, {{src, fg.nearest}, {fg.frames[cur].view, VK_NULL_HANDLE}}, uw, uh, pc, sizeof pc);
    pass(cmd, fg.luma, {{fg.frames[cur].view, fg.nearest}, {fg.lumas[cur].view, VK_NULL_HANDLE}}, qw, qh);
    set->made = 0;
    if (blend && m > 1) {
        pass(cmd, fg.flow, {{fg.lumas[prev].view, fg.nearest}, {fg.lumas[cur].view, fg.nearest},
                            {fg.flows[0].view, VK_NULL_HANDLE}}, cw, ch);
        pass(cmd, fg.median, {{fg.flows[0].view, fg.nearest}, {fg.flows[1].view, VK_NULL_HANDLE}}, cw, ch);
        for (int k = 1; k < m; k++) {
            const float t = (float)k / (float)m;
            pass(cmd, fg.synth, {{fg.frames[prev].view, fg.linear}, {fg.frames[cur].view, fg.linear},
                                 {fg.flows[1].view, fg.nearest}, {set->frame[k - 1].view, VK_NULL_HANDLE}},
                 uw, uh, &t, 4);
        }
        set->made = m - 1;
    }
    const VkImageCopy c{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0},
                        {uw, uh, 1}};
    vkCmdCopyImage(cmd, fg.frames[cur].image, VK_IMAGE_LAYOUT_GENERAL, set->frame[set->made].image,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    full_barrier(cmd);
    fg.have_prev = true;
    fg.cur = prev;
    fg.recorded = set;
    fg.recorded_at = now;
    return true;
}

void framegen_queue(uint64_t value) {
    if (fg.ring_recorded >= 0) {   // (an empty submit behind the recording that copied the frame: the worker's other queue waits on it)
        const int i = fg.ring_recorded;
        fg.ring_recorded = -1;
        VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        sig.semaphore = fg.ring_tl;
        sig.value = ++fg.ring_value;
        sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos = &sig;
        fg.ring[i].wait = sig.value;
        {
            std::lock_guard qk(dev.present_mu);
            queue_flush_locked();
            vkQueueSubmit2(dev.queue, 1, &si, VK_NULL_HANDLE);
        }
        std::lock_guard lk(fg.mu);
        fg.work.push_back(i);
        fg.cv.notify_all();
        return;
    }
    if (!fg.recorded) return;
    std::lock_guard lk(fg.mu);
    fg.jobs.push_back({fg.recorded, fg.recorded_at, value});
    fg.recorded = nullptr;
    fg.cv.notify_all();
}

} // namespace mkwvk
