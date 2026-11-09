"""The menu's DLSS 5 row and the present queue it asks for (gpu/vk/vk_main.cpp, vk_replay.cpp, vk_framegen.cpp), no game or GPU required.

python -I -B tests/test_dlss5_present.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"


def need(text, what, why):
    if what not in text:
        raise SystemExit("%s is missing (%s)" % (what, why))


def main():
    vk_main = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    instance = vk_main[vk_main.index("bool create_instance() {"):]
    instance = instance[:instance.index("\n}\n")]
    need(instance, 'setenv("VKLayer_DLSS5", "1", 1)', "the layer turns on by this variable")
    if instance.index('setenv("VKLayer_DLSS5"') > instance.index("vkCreateInstance(&ci"):
        raise SystemExit("the DLSS 5 layer's variable is set after the instance that reads it is made")
    if instance.index("menu_init()") > instance.index('setenv("VKLayer_DLSS5"'):
        raise SystemExit("the DLSS 5 row is used before the saved options are read")
    need(instance, 'setenv("NVPRESENT_ENABLE_SMOOTH_MOTION", "1", 1)', "Smooth Motion turns on by this variable")
    need(instance, '"VK_LAYER_NV_dlssnr:VK_LAYER_NV_present"', "Smooth Motion after DLSS 5, so the model sees only the game's frames")
    need(vk_main, "q.queueCount = separate_present ? 2 : 1", "one queue unless the presents are apart")
    need(vk_main, '"RX_VK_ASYNC_PRESENT"', "the switch to compare")
    need(vk_main, "dev.dlss5_layer", "only with the layer really there")

    replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")
    present = replay[replay.index("void present_image(VkImage src"):]
    present = present[:present.index("\n}\n")]
    need(present, "qk.unlock();", "the queue's lock is not held through a present on the other queue")
    if present.index("qk.unlock();") > present.index("vkQueuePresentKHR(dev.present_queue"):
        raise SystemExit("present_image presents on the present queue before it lets the submits' lock go")
    if present.index("qk.lock();") < present.index("vkQueuePresentKHR(dev.present_queue"):
        raise SystemExit("present_image takes the submits' lock back before the present is done")
    held = replay.replace("std::lock_guard pq(dev.present_queue_mu); vkDeviceWaitIdle(dev.device);", "")
    if "vkDeviceWaitIdle(dev.device);" in held.replace("std::lock_guard pq(dev.present_queue_mu);\n    vkDeviceWaitIdle(dev.device);", ""):
        raise SystemExit("vk_replay.cpp: a vkDeviceWaitIdle that does not hold the present queue")

    framegen = (VK / "vk_framegen.cpp").read_text(encoding="utf-8")
    need(framegen, "|| dev.present_queue)", "with a present queue the display thread shows the frames")

    need(vk_main, '"vkDlssnrProcessImageHOOKSHOT"', "the layer's entry point is found by name")
    need(vk_main, '"RX_VK_DLSS5_BEFORE_FG"', "the switch to compare")
    need(vk_main, "not found", "the case with no entry point is said in the log")
    internal = (VK / "vk_internal.h").read_text(encoding="utf-8")
    need(internal, "PFN_vkDlssnrProcessImageHOOKSHOT dlss5_process", "null keeps the model at the present")

    need(framegen, "RingSlot ring[kRingSlots]", "the worker path's frames")
    need(framegen, "VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);\n        if (!fg.ring_tl)", "the model writes its answer back into the slot")
    worker = framegen[framegen.index("void worker_main()"):]
    worker = worker[:worker.index("\n} // namespace")]
    need(worker, "dev.dlss5_process(dev.device, dev.present_queue", "the model on the present queue, not the replay's")
    need(worker, "fg.jobs.push_back({set, at, done.value, fg.out_tl})", "the job stamped once the worker has finished, and carrying the order across queues")
    need(framegen, "j.set->frame[k].view, j.wait, j.value)", "the display thread's submit on the other queue waits for the worker's")
    need(present, "wait ? 2 : 1", "present_image waits for what another queue wrote")
    if worker.index("const int64_t at = now_ns();") < worker.index("vkWaitForFences(dev.device, 1, &fence"):
        raise SystemExit("the display thread's stamp is taken before the worker's GPU work is done")
    need(worker, "vkWaitSemaphores", "the worker waits for the copy before the model reads it")
    if worker.index("vkWaitSemaphores") > worker.index("dev.dlss5_process("):
        raise SystemExit("the model runs before the copy into the slot is known done")
    need(worker, "raw == fg.prev_raw", "a raw frame is not blended with a processed one")
    if "dev.dlss5_process(" in framegen[:framegen.index("void worker_main()")]:
        raise SystemExit("the model's call off the worker thread: the replay thread must not wait for it")
    record = framegen[framegen.index("bool framegen_record("):]
    record = record[:record.index("\n}\n")]
    need(record, "if (fg.worker) {", "the replay thread only copies on the worker path")
    ring_use = record[record.index("if (fg.worker) {"):]
    ring_use = ring_use[:ring_use.index("return true;")]
    if "fg.luma" in ring_use or "fg.flow" in ring_use or "fg.synth" in ring_use:
        raise SystemExit("the replay thread records the worker's passes")
    need(worker, "wanted ? dev.dlss5_process(", "the menu's row decides each frame, while the game runs")
    need(worker, "menu_set_dlss5_live();", "the row stops asking for a restart once the worker calls the model")
    size = framegen[framegen.index("bool ensure_size("):]
    size = size[:size.index("\n}\n")]
    need(size, "fg.work.empty() && !fg.worker_busy", "the worker is drained before the images go")
    if size.index("!fg.worker_busy") > size.index("device_wait_idle_locked"):
        raise SystemExit("ensure_size waits for the device before the worker has stopped using it")
    queue = framegen[framegen.index("void framegen_queue("):]
    queue = queue[:queue.index("\n}\n")]
    need(queue, "queue_flush_locked();", "the copy's recording is out before the worker waits for it")
    if queue.index("queue_flush_locked();") > queue.index("vkQueueSubmit2(dev.queue"):
        raise SystemExit("framegen_queue signals the ring timeline ahead of the held copy it is to follow")

    menu = (VK / "vk_menu.cpp").read_text(encoding="utf-8")
    row = menu.index('"DLSS 5", &opts.dlss5')
    if menu.rfind("#ifndef _WIN32", 0, row) < menu.rfind("#endif", 0, row):
        raise SystemExit("the DLSS 5 row is not Linux only: the layer has no Windows build")
    need(menu, "r.value != &opts.dlss5", "the row does not reset the quality preset to Custom")
    need(menu, "r.value != &opts.smooth_motion", "nor does the Smooth Motion row")
    need(menu, '"On (restart)"', "without the worker the row still needs a restart")

    # (nothing of NVIDIA's or of the layer's in the repository: the player installs them)
    found = [p.name for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and
             (p.name.lower().startswith(("nvngx_dlssnr", "libvklayer_nv_dlssnr", "dlssnr_helper")))]
    if found:
        raise SystemExit("DLSS 5 files must not be in the repository: %s" % ", ".join(found))
    print("DLSS 5 row and present queue: ok")


if __name__ == "__main__":
    main()
