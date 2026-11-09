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
    if "vkDeviceWaitIdle(dev.device);" in replay.replace("std::lock_guard pq(dev.present_queue_mu); vkDeviceWaitIdle(dev.device);", ""):
        raise SystemExit("vk_replay.cpp: a vkDeviceWaitIdle that does not hold the present queue")

    framegen = (VK / "vk_framegen.cpp").read_text(encoding="utf-8")
    need(framegen, "|| dev.present_queue)", "with a present queue the display thread shows the frames")

    menu = (VK / "vk_menu.cpp").read_text(encoding="utf-8")
    row = menu.index('"DLSS 5", &opts.dlss5')
    if menu.rfind("#ifndef _WIN32", 0, row) < menu.rfind("#endif", 0, row):
        raise SystemExit("the DLSS 5 row is not Linux only: the layer has no Windows build")
    need(menu, "r.value != &opts.dlss5", "the row does not reset the quality preset to Custom")
    need(menu, "r.value != &opts.smooth_motion", "nor does the Smooth Motion row")

    # (nothing of NVIDIA's or of the layer's in the repository: the player installs them)
    found = [p.name for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and
             (p.name.lower().startswith(("nvngx_dlssnr", "libvklayer_nv_dlssnr", "dlssnr_helper")))]
    if found:
        raise SystemExit("DLSS 5 files must not be in the repository: %s" % ", ".join(found))
    print("DLSS 5 row and present queue: ok")


if __name__ == "__main__":
    main()
