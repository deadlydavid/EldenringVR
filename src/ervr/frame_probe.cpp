#include "game_version.h"
// M2 spike: map ELDEN RING's frame task system.
//
// True stereo needs the scene drawn twice from two cameras at ONE simulation time. Every
// mechanism for that goes through Dantelion2's task scheduler: the frame is a strictly
// ordered list of NAMED task groups, all simulation lives upstream of `Draw_Pre`, and the
// draw itself is `DrawStep, DrawBegin, GameSceneDraw, AdhocDraw, DrawEnd`. If those can be
// re-driven with a second camera, we get stereo for ~1x CPU and 2x GPU. If they cannot, we
// fall back to double-stepping the whole engine with a banked delta time.
//
// Before any of that we have to know the table is really there on eldenring.exe 2.6.2.0 and
// that its order matches the map libER documents (Dasaav, Apache-2.0 w/ LLVM exceptions;
// include/coresystem/taskgroups.inl). This file only READS - it locates the table, walks it,
// and reports what it found. Nothing here changes game state.
#include "frame_probe.h"
#include "debug_keys.h"
#include "dx12_blit.h"
#include "game_er.h"
#include "combat.h"
#include "pass2_interact.h"
#include "virtual_pad.h"
#include "patches_er.h"
#include "game_memory.h"
#include "stereo.h"
#include "rtti.h"
#include "reshade_addon.h"
#include "config.h"
#include "log.h"
#include <MinHook.h>
#include <intrin.h>
#include <windows.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ervr {
namespace {

// libER: CS::CSTaskBase::cstg_control { cstgi id; wchar_t name[64]; bool active; }
// The stride is discovered at runtime rather than assumed - padding is a compiler choice.
constexpr size_t kNameOff = 4;
constexpr size_t kActiveOff = kNameOff + 64 * sizeof(wchar_t);   // 132, confirmed on 2.6.2.0
constexpr int    kMaxGroups = 256;

TaskGroupTable g_table;

// The frame order libER documents. Used only to VERIFY what we find; the live table wins.
const wchar_t* kExpected[] = {
    L"FrameBegin", L"SteamThread0", L"SteamThread1", L"SteamThread2", L"SteamThread3",
    L"SteamThread4", L"SteamThread5", L"SystemStep", L"ResStep", L"PadStep", L"GameFlowStep",
    L"EndShiftWorldPosition", L"GameMan", L"TaskLineIdx_Sys", L"TaskLineIdx_Test",
    L"TaskLineIdx_NetworkFlowStep", L"TaskLineIdx_InGame_InGameStep",
    L"TaskLineIdx_InGame_InGameStayStep", L"MovieStep", L"RemoStep",
    L"TaskLineIdx_InGame_MoveMapStep", L"FieldArea_EndWorldAiManager", L"EmkSystem_Pre",
    L"EmkSystem_ConditionStatus", L"EmkSystem_Post", L"EventMan",
    L"FlverResDelayDelectiionBegin", L"TaskLineIdx_InGame_FieldAreaStep",
    L"TaskLineIdx_InGame_TestNetStep", L"TaskLineIdx_InGame_InGameMenuStep",
    L"TaskLineIdx_InGame_TitleMenuStep", L"TaskLineIdx_InGame_CommonMenuStep",
    L"TaskLineIdx_FrpgNet_Sys", L"TaskLineIdx_FrpgNet_Lobby", L"TaskLineIdx_FrpgNet_ConnectMan",
    L"TaskLineIdx_FrpgNet_Connect", L"TaskLineIdx_FrpgNet_Other", L"SfxMan", L"FaceGenMan",
    L"FrpgNetMan", L"NetworkUserManager", L"SessionManager", L"BlockList", L"LuaConsoleServer",
    L"RmiMan", L"ResMan", L"SfxDebugger", L"REMOTEMAN", L"Geom_WaitActivateFade",
    L"Geom_UpdateDraw", L"Grass_BatchUpdate", L"Grass_ResourceLoadKick", L"Grass_ResourceLoad",
    L"Grass_ResourceCleanup", L"WorldChrMan_Respawn", L"WorldChrMan_Prepare",
    L"ChrIns_CalcUpdateInfo_PerfBegin", L"ChrIns_CalcUpdateInfo", L"ChrIns_CalcUpdateInfo_PerfEnd",
    L"WorldChrMan_PrePhysics", L"WorldChrMan_CalcOmissionLevel_Begin",
    L"WorldChrMan_CalcOmissionLevel", L"WorldChrMan_CalcOmissionLevel_End",
    L"WorldChrMan_ConstructUpdateList", L"WorldChrMan_ChrNetwork", L"ChrIns_Prepare",
    L"ChrIns_NaviCache", L"ChrIns_AILogic_PerfBegin", L"ChrIns_AILogic", L"ChrIns_AILogic_PerfEnd",
    L"AI_SimulationStep", L"ChrIns_PreBehavior", L"ChrIns_PreBehaviorSafe",
    L"GeomModelInsCreatePartway_Begin", L"HavokBehavior", L"GeomModelInsCreatePartway_End",
    L"ChrIns_BehaviorSafe", L"ChrIns_PrePhysics_Begin", L"ChrIns_PrePhysics",
    L"ChrIns_PrePhysics_End", L"NetFlushSendData", L"ChrIns_PrePhysicsSafe", L"ChrIns_RagdollSafe",
    L"ChrIns_GarbageCollection", L"GeomModelInsCreate", L"AiBeginCollectGabage",
    L"WorldChrMan_Update_RideCheck", L"InGameDebugViewer", L"LocationStep",
    L"LocationUpdate_PrePhysics", L"LocationUpdate_PrePhysics_Parallel",
    L"LocationUpdate_PrePhysics_Post", L"LocationUpdate_PostCloth",
    L"LocationUpdate_PostCloth_Parallel", L"LocationUpdate_PostCloth_Post",
    L"LocationUpdate_DebugDraw", L"EventCondition_BonfireNearEnemyCheck", L"HavokWorldUpdate_Pre",
    L"RenderingSystemUpdate", L"HavokWorldUpdate_Post", L"ChrIns_PreCloth", L"ChrIns_PreClothSafe",
    L"HavokClothUpdate_Pre_AddRemoveRigidBody", L"HavokClothUpdate_Pre_ClothModelInsSafe",
    L"HavokClothUpdate_Pre_ClothModelIns", L"HavokClothUpdate_Pre_ClothManager", L"CameraStep",
    L"DrawParamUpdate", L"GetNPAuthCode", L"SoundStep", L"HavokClothUpdate_Post_ClothManager",
    L"HavokClothUpdate_Post_ClothModelIns", L"HavokClothVertexUpdateFinishWait",
    L"ChrIns_PostPhysics", L"ChrIns_PostPhysicsSafe", L"CSDistViewManager_Update",
    L"HavokAi_SilhouetteGeneratorHelper_Begin", L"WorldChrMan_PostPhysics",
    L"GameFlowInGame_MoveMap_PostPhysics_0", L"HavokAi_SilhouetteGeneratorHelper_End",
    L"DmgMan_Pre", L"DmgMan_ShapeCast", L"DmgMan_Post",
    L"GameFlowInGame_MoveMap_PostPhysics_1_Core0", L"GameFlowInGame_MoveMap_PostPhysics_1_Core1",
    L"GameFlowInGame_MoveMap_PostPhysics_1_Core2", L"MenuMan",
    L"WorldChrMan_Update_BackreadRequestPre", L"ChrIns_Update_BackreadRequest",
    L"WorldChrMan_Update_BackreadRequestPost", L"HavokAi_World",
    L"WorldAiManager_BeginUpdateFormation", L"WorldAiManager_EndUpdateFormation",
    L"GameFlowInGame_TestNet", L"GameFlowInGame_InGameMenu", L"GameFlowInGame_TitleMenu",
    L"GameFlowInGame_CommonMenu", L"GameFlowFrpgNet_Sys", L"GameFlowFrpgNet_Lobby",
    L"GameFlowFrpgNet_ConnectMan", L"GameFlowFrpgNet_Connect", L"GameFlowStep_Post",
    L"ScaleformStep", L"FlverResDelayDelectiionEnd", L"Draw_Pre", L"GraphicsStep",
    L"DebugDrawMemoryBar", L"DbgMenuStep", L"DbgRemoteStep", L"PlaylogSystemStep", L"ReviewMan",
    L"ReportSystemStep", L"DbgDispStep", L"DrawStep", L"DrawBegin", L"GameSceneDraw",
    L"AdhocDraw", L"DrawEnd", L"Draw_Post", L"SoundPlayLimitterUpdate",
    L"BeginShiftWorldPosition", L"FileStep", L"FileStepUpdate_Begin", L"FileStepUpdate_End",
    L"Flip", L"DelayDeleteStep", L"AiEndCollectGabage", L"RecordHeapStats", L"FrameEnd",
};
constexpr int kExpectedCount = (int)(sizeof(kExpected) / sizeof(kExpected[0]));

// A snapshot of every readable committed region, sorted, so validity checks are a binary
// search instead of a syscall. The discovery pass makes millions of them; at one VirtualQuery
// each it would take minutes.
struct Region { uintptr_t lo, hi; };
std::vector<Region> g_regions;

void BuildRegionMap() {
    g_regions.clear();
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    uint8_t* p = (uint8_t*)si.lpMinimumApplicationAddress;
    uint8_t* const hi = (uint8_t*)si.lpMaximumApplicationAddress;
    MEMORY_BASIC_INFORMATION mbi{};
    while (p < hi && VirtualQuery(p, &mbi, sizeof(mbi))) {
        uint8_t* base = (uint8_t*)mbi.BaseAddress;
        if (mbi.State == MEM_COMMIT && !(mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
            g_regions.push_back({(uintptr_t)base, (uintptr_t)base + mbi.RegionSize});
        uint8_t* next = base + mbi.RegionSize;
        if (next <= p) break;
        p = next;
    }
    std::sort(g_regions.begin(), g_regions.end(),
              [](const Region& a, const Region& b) { return a.lo < b.lo; });
}

bool Readable(const void* p, size_t n) {
    const uintptr_t a = (uintptr_t)p;
    if (g_regions.empty()) {   // before the map is built, fall back to the syscall
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(p, &mbi, sizeof(mbi))) return false;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
        return n <= (uintptr_t)mbi.BaseAddress + mbi.RegionSize - a;
    }
    auto it = std::upper_bound(g_regions.begin(), g_regions.end(), a,
                               [](uintptr_t v, const Region& r) { return v < r.lo; });
    if (it == g_regions.begin()) return false;
    --it;
    // Never a + n: statics hold sentinels like -0x10, and the sum wraps past every region's end.
    return a >= it->lo && a < it->hi && n <= it->hi - a;
}

// Wide-string compare that tolerates a non-terminated fixed field.
bool NameIs(const uint8_t* entry, const wchar_t* want) {
    const wchar_t* got = (const wchar_t*)(entry + kNameOff);
    for (int i = 0; i < 64; ++i) {
        if (got[i] != want[i]) return false;
        if (want[i] == 0) return true;
    }
    return false;
}

uint64_t g_bytesScanned = 0;

// One region's worth of searching, isolated so the SEH frame contains no C++ objects.
const uint8_t* ScanRegion(const uint8_t* base, size_t size, const uint8_t* needle, size_t n) {
    __try {
        for (size_t i = 0; i + n <= size; ++i)
            if (base[i] == needle[0] && memcmp(base + i, needle, n) == 0) return base + i;
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return nullptr;
}

// Walks committed private memory for the first byte sequence `needle`.
//
// This is measured in gigabytes on a running ELDEN RING, so it MUST NOT run on the worker
// thread - that thread also drives BackendsTick(), and blocking it means the OpenXR instance
// is never created and the headset never comes up at all.
uint8_t* ScanHeap(const uint8_t* needle, size_t n, size_t maxRegion, uint8_t* startAt,
                  uint64_t deadlineTick) {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    uint8_t* p = startAt ? startAt : (uint8_t*)si.lpMinimumApplicationAddress;
    uint8_t* const hi = (uint8_t*)si.lpMaximumApplicationAddress;
    MEMORY_BASIC_INFORMATION mbi{};
    while (p < hi && VirtualQuery(p, &mbi, sizeof(mbi))) {
        uint8_t* base = (uint8_t*)mbi.BaseAddress;
        size_t   size = mbi.RegionSize;
        const bool usable = mbi.State == MEM_COMMIT && (mbi.Type & MEM_PRIVATE) &&
                            !(mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) &&
                            (mbi.Protect & (PAGE_READWRITE | PAGE_READONLY | PAGE_WRITECOPY |
                                            PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE));
        if (usable && size >= n && size <= maxRegion) {
            if (GetTickCount64() > deadlineTick) {
                LOGW("[m2] scan deadline hit after %llu MB", (unsigned long long)(g_bytesScanned >> 20));
                return nullptr;
            }
            g_bytesScanned += size;
            if (const uint8_t* hit = ScanRegion(base, size, needle, n)) return (uint8_t*)hit;
        }
        uint8_t* next = base + size;
        if (next <= p) break;
        p = next;
    }
    return nullptr;
}

// Finds a qword anywhere in committed private memory whose value is `target`. Used to walk
// back from the task-group ARRAY to the vector that owns it, and from there to the object.
uint8_t* FindPointerTo(const void* target, size_t maxRegion, uint64_t deadlineTick) {
    uint8_t needle[8];
    memcpy(needle, &target, 8);
    return ScanHeap(needle, 8, maxRegion, nullptr, deadlineTick);
}

// A real vtable's slots all point into the module's EXECUTABLE sections. The first attempt at
// this just read *obj and printed it, which produced convincing-looking nonsense - vft[0] came
// back equal to `this`. Verify before believing.
bool ExecRange(const void* p, uint8_t*& lo, uint8_t*& hi) {
    static uint8_t* s_lo = nullptr; static uint8_t* s_hi = nullptr;
    if (!s_lo) {
        auto* base = (uint8_t*)GetModuleHandleW(nullptr);
        auto* dos = (IMAGE_DOS_HEADER*)base;
        auto* nt = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
        auto* sec = IMAGE_FIRST_SECTION(nt);
        for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
            if (!(sec->Characteristics & IMAGE_SCN_MEM_EXECUTE)) continue;
            uint8_t* a = base + sec->VirtualAddress;
            uint8_t* b = a + sec->Misc.VirtualSize;
            if (!s_lo || a < s_lo) s_lo = a;
            if (!s_hi || b > s_hi) s_hi = b;
        }
    }
    lo = s_lo; hi = s_hi;
    return p >= s_lo && p < s_hi;
}

bool LooksLikeVTable(void** vt, int need) {
    if (!Readable(vt, (size_t)need * 8)) return false;
    uint8_t *lo, *hi;
    for (int i = 0; i < need; ++i)
        if (!ExecRange(vt[i], lo, hi)) return false;
    return true;
}

// Given a candidate object, look for the task_groups vector inside it: a { begin, end, cap }
// triple whose begin points at an entry that reads { id, L"FrameBegin" }. Returns the array
// and measures the element stride from where entry 1 lands.
//
// SEH-guarded, and that is not belt-and-braces: the region map is a SNAPSHOT, ELDEN RING's
// allocator churns constantly, and across ~100k candidates something will inevitably be freed
// between the check and the read. A fault here just means "not this one".
// Locals are deliberately all POD so the __try is legal.
int VectorInObject(uint8_t* obj, size_t searchBytes, uint8_t** entry0, size_t* stride,
                   size_t* vecOff) {
    __try {
        // Deliberately NO "first slot must be a vtable" pre-filter. The static that leads here
        // points 0x18 BEFORE the object's start, so slot 0 is a plain field - that filter threw
        // the real hit away and turned a working search into a silent miss.
        for (size_t off = 0; off + 8 <= searchBytes; off += 8) {
            if (!Readable(obj + off, 8)) continue;
            uint8_t* begin = *(uint8_t**)(obj + off);
            if (!begin || ((uintptr_t)begin & 7)) continue;
            if (!Readable(begin, kNameOff + 64 * sizeof(wchar_t))) continue;
            if (!NameIs(begin, kExpected[0])) continue;
            for (size_t s = 132; s <= 200; s += 4) {
                if (!Readable(begin + s, kNameOff + 64 * sizeof(wchar_t))) continue;
                if (!NameIs(begin + s, kExpected[1])) continue;
                *entry0 = begin; *stride = s; *vecOff = off;
                return 1;
            }
        }
        return 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return -1;
    }
}

// Returns true when the probe is finished (found it, or gave up).
bool RunOnce(int attempt) {
    LOGI("[m2] ---- frame task-system probe (attempt %d) ----", attempt);
    g_bytesScanned = 0;
    const uint64_t t0 = GetTickCount64();

    uint8_t* entry0 = nullptr;
    uint8_t* taskObj = nullptr;
    size_t   stride = 0, vecOff = 0;

    // 1) Discovery through the module's own statics. Dantelion2 keeps its managers in globals,
    //    so the object is reachable from a few MB of .data - as opposed to hunting the string
    //    in the heap, which on a live ELDEN RING means touching gigabytes of committed-but-cold
    //    pages at ~9 MB/s while evicting the game's own working set.
    //
    //    Deliberately NOT driven by RTTI class names. 640 classes match "Task" on this build,
    //    FindVTable() is a full section scan EACH (~2 s, so ~20 minutes for the set), and MSVC's
    //    mangling of nested template classes does not survive a naive demangle anyway. The
    //    object's CONTENT identifies it far better than its name: exactly one object in the
    //    process holds a vector whose first element is named L"FrameBegin".
    BuildRegionMap();

    // 0) Named RTTI first. The concrete class is CS::CSTaskImp - NOT "CS::CSTask", which has
    //    no type descriptor of its own. But the Imp is NOT the callee: per libER's task.hpp,
    //    CSTaskImp is just { vptr, CSTask* inst } - a singleton handle. The object that owns
    //    the task_groups vector and the add/set/queue virtuals is the CSTask at Imp+0x8.
    //    (The first attempt called the IMP's vtable with the CSTask's `this` and faulted.)
    uint8_t* cstaskFromImp = nullptr;
    if (void** vt = FindVTable("CS::CSTaskImp")) {
        std::vector<void*> insts = FindInstancesOfVTable(vt, 4);
        LOGI("[m2] rtti CS::CSTaskImp: vtable %p, %zu instance(s)", (void*)vt, insts.size());
        if (!insts.empty() && Readable((uint8_t*)insts[0] + 8, 8)) {
            g_table.imp = (uint8_t*)insts[0];
            cstaskFromImp = *(uint8_t**)((uint8_t*)insts[0] + 8);
            LOGI("[m2] CSTaskImp::inst (the CSTask) = %p", (void*)cstaskFromImp);
        }
    }

    // The CSTask from the Imp is authoritative when it checks out: it must hold the vector
    // (libER: vptr@0, allocator@8, task_groups@0x10 with std::vector's allocator first, so
    // begin lands at +0x18 - exactly where the statics scan found it last time).
    if (cstaskFromImp) {
        const int r = VectorInObject(cstaskFromImp, 0x120, &entry0, &stride, &vecOff);
        if (r == 1) {
            taskObj = cstaskFromImp;
            LOGI("[m2] CSTaskImp::inst holds the table (vector at +0x%zx) - it IS the CSTask",
                 vecOff);
        } else {
            LOGW("[m2] CSTaskImp::inst does NOT hold the table (r=%d) - falling back to statics", r);
        }
    }

    std::vector<void*> statics = ModuleStaticPointers();
    long faults = 0;
    if (!taskObj) for (void* cand : statics) {
        if (!Readable(cand, 0x10)) continue;   // VectorInObject bounds-checks each slot itself
        const int r = VectorInObject((uint8_t*)cand, 0x120, &entry0, &stride, &vecOff);
        if (r < 0) { ++faults; continue; }
        if (r == 0) continue;
        taskObj = (uint8_t*)cand;
        LOGI("[m2] FOUND via module statics: object at %p, task_groups vector at +0x%zx "
             "(%zu candidates, %zu regions, %ld faults, %llu ms)",
             cand, vecOff, statics.size(), g_regions.size(), faults,
             (unsigned long long)(GetTickCount64() - t0));
        break;
    }
    if (taskObj && cstaskFromImp && taskObj != cstaskFromImp)
        LOGW("[m2] statics scan (%p) and CSTaskImp::inst (%p) DISAGREE about the CSTask - "
             "using the one that holds the vector", (void*)taskObj, (void*)cstaskFromImp);
    if (!taskObj)
        LOGW("[m2] no module static points at an object holding the table "
             "(%zu candidates, %ld faults, %llu ms); trying the heap",
             statics.size(), faults, (unsigned long long)(GetTickCount64() - t0));

    // 2) Fallback: hunt the array itself in the heap. Tightly budgeted - this is the path
    //    that thrashed the working set, so it only gets a small slice and small regions.
    if (!entry0) {
        const uint64_t deadline = GetTickCount64() + 12000;
        const wchar_t* first = kExpected[0];
        const size_t needleBytes = (wcslen(first) + 1) * sizeof(wchar_t);
        uint8_t* resume = nullptr;
        for (int probe = 0; probe < 8 && !entry0; ++probe) {
            uint8_t* hit = ScanHeap((const uint8_t*)first, needleBytes, 32u << 20, resume, deadline);
            if (!hit) break;
            resume = hit + 2;
            uint8_t* cand = hit - kNameOff;
            for (size_t s = 132; s <= 200 && !entry0; s += 4) {
                if (!Readable(cand + s, kNameOff + 64 * sizeof(wchar_t))) continue;
                if (NameIs(cand + s, kExpected[1])) { entry0 = cand; stride = s; }
            }
        }
        if (entry0) LOGI("[m2] found the table by heap scan (%llu MB)",
                         (unsigned long long)(g_bytesScanned >> 20));
    }

    if (!entry0) {
        LOGW("[m2] table not found this pass (%llu MB heap in %llu ms) - engine may still be booting",
             (unsigned long long)(g_bytesScanned >> 20),
             (unsigned long long)(GetTickCount64() - t0));
        return false;
    }
    LOGI("[m2] task-group table at %p, stride %zu bytes (id@0, name@%zu, active@%zu) in %llu ms",
         (void*)entry0, stride, kNameOff, kNameOff + 64 * sizeof(wchar_t),
         (unsigned long long)(GetTickCount64() - t0));

    // 2) The vector's `end` gives the real count. Walking until the name looks wrong overran
    //    into unrelated heap and reported 256 groups where there are 169.
    int listed = 0;
    if (taskObj && Readable(taskObj + vecOff, 24)) {
        uint8_t* vbegin = *(uint8_t**)(taskObj + vecOff);
        uint8_t* vend   = *(uint8_t**)(taskObj + vecOff + 8);
        if (vbegin == entry0 && vend > vbegin && (size_t)(vend - vbegin) % stride == 0) {
            listed = (int)((vend - vbegin) / stride);
            LOGI("[m2] vector says %d groups (begin %p end %p)", listed, (void*)vbegin, (void*)vend);
        }
    }

    // 3) Walk it, comparing against the documented order as we go.
    int count = 0, mismatches = 0;
    std::string line;
    char buf[160];
    const int walkMax = listed > 0 ? listed : kMaxGroups;
    for (int i = 0; i < walkMax; ++i) {
        uint8_t* e = entry0 + (size_t)i * stride;
        if (!Readable(e, stride)) break;
        const wchar_t* nm = (const wchar_t*)(e + kNameOff);
        if (nm[0] == 0) break;
        const int32_t id = *(const int32_t*)e;
        const uint8_t active = e[kNameOff + 64 * sizeof(wchar_t)];
        // A name field that is not NUL-terminated inside 64 chars means we walked off the end.
        bool terminated = false;
        for (int k = 0; k < 64; ++k) if (nm[k] == 0) { terminated = true; break; }
        if (!terminated) break;

        char nameA[80]{};
        WideCharToMultiByte(CP_UTF8, 0, nm, -1, nameA, sizeof(nameA) - 1, nullptr, nullptr);
        const bool known = i < kExpectedCount;
        const bool same = known && NameIs(e, kExpected[i]);
        if (known && !same) ++mismatches;
        snprintf(buf, sizeof(buf), "%d:%s%s%s ", id, nameA, active ? "" : "(off)", same || !known ? "" : "!");
        line += buf;
        if (line.size() > 150) { LOGI("[m2]   %s", line.c_str()); line.clear(); }
        ++count;
    }
    if (!line.empty()) LOGI("[m2]   %s", line.c_str());

    LOGI("[m2] %d groups live (libER documents %d); %d name mismatches%s",
         count, kExpectedCount, mismatches,
         mismatches == 0 && count == kExpectedCount
             ? " - the frame map matches this build EXACTLY, indices are usable as-is"
             : " - indices must be taken from the live table, not from libER's enum");

    // 3) The draw range is the whole point of the spike; call it out explicitly.
    for (int i = 0; i < count; ++i) {
        uint8_t* e = entry0 + (size_t)i * stride;
        for (const wchar_t* want : {L"CameraStep", L"Draw_Pre", L"GraphicsStep", L"DrawStep",
                                    L"DrawBegin", L"GameSceneDraw", L"AdhocDraw", L"DrawEnd",
                                    L"Draw_Post", L"Flip"}) {
            if (!NameIs(e, want)) continue;
            char nameA[80]{};
            WideCharToMultiByte(CP_UTF8, 0, want, -1, nameA, sizeof(nameA) - 1, nullptr, nullptr);
            // The id is NOT the index: it is 0x90000000 | index. set_task_group_state wants
            // the id, so print both.
            LOGI("[m2]   key group %-16s index %3d  id 0x%08X  entry %p  active %d",
                 nameA, i, (unsigned)*(const int32_t*)e, (void*)e,
                 e[kNameOff + 64 * sizeof(wchar_t)]);
        }
    }

    // 4) The object is what we actually need: `this` for set_task_group_state / queue_tasks.
    //    RTTI already gave it to us; the heap-scan fallback has to work backwards to it.
    if (!taskObj) {
        uint8_t* vecBegin = FindPointerTo(entry0, 32u << 20, GetTickCount64() + 12000);
        if (!vecBegin) {
            LOGW("[m2] table found but the owning object was not; no `this` for the vtable calls");
        } else {
            auto* base = (uint8_t*)GetModuleHandleW(nullptr);
            auto* modEnd = base + 0x10000000ull;
            for (size_t back = 8; back <= 64; back += 8) {
                uint8_t* obj = vecBegin - back;
                if (!Readable(obj, 8)) continue;
                uint8_t* maybeVt = *(uint8_t**)obj;
                if (maybeVt < base || maybeVt >= modEnd || !Readable(maybeVt, 8)) continue;
                uint8_t* fn = *(uint8_t**)maybeVt;
                if (fn < base || fn >= modEnd) continue;
                taskObj = obj;
                LOGI("[m2] owning object likely at %p (vector at +0x%zx)", (void*)obj, back);
                break;
            }
        }
    }
    // Module statics can point 0x18 bytes before the CSTask object. In that
    // case the task-group vector appears at +0x30, rather than CSTask's +0x18.
    // Only correct the pointer when BOTH the vtable and vector validate.
    if (taskObj && vecOff == 0x30) {
        uint8_t* adjusted = taskObj + 0x18;
        if (Readable(adjusted, 8)) {
            void** adjustedVt = *(void***)adjusted;
            if (adjustedVt && LooksLikeVTable(adjustedVt, 6)) {
                uint8_t* checkEntry = nullptr;
                size_t checkStride = 0, checkOff = 0;
                if (VectorInObject(adjusted, 0x40, &checkEntry, &checkStride, &checkOff) == 1 &&
                    checkEntry == entry0 && checkStride == stride && checkOff == 0x18) {
                    LOGI("[m2] corrected CSTask owner: %p -> %p (static was -0x18)",
                         (void*)taskObj, (void*)adjusted);
                    taskObj = adjusted;
                    vecOff = checkOff;
                }
            }
        }
    }
    if (taskObj) {
        LOGI("[m2] owner object `this` = %p", (void*)taskObj);
        // The callee vtable is the CSTask's OWN vptr at +0. CSTask : CSTaskBase is single
        // inheritance, and both declare a virtual destructor, so slot 0 is the dtor and the
        // libER-declared virtuals start at 1. The first attempt indexed them from 0 (and used
        // the Imp's vtable) - that is why "slot 2" faulted: it was unk_set_task_group on the
        // wrong object.
        void** vt = Readable(taskObj, 8) ? *(void***)taskObj : nullptr;
        if (vt && LooksLikeVTable(vt, 6)) {
            g_table.vtable = vt;
            auto* base = (uint8_t*)GetModuleHandleW(nullptr);
            const std::string cls = RttiNameOfVTable(vt);
            LOGI("[m2] CSTask vtable %p (rva 0x%llx) rtti \"%s\"", (void*)vt,
                 (unsigned long long)((uint8_t*)vt - base),
                 cls.empty() ? "<no rtti>" : cls.c_str());
            const char* names[6] = {"~dtor", "add_task_group", "unk_set_task_group",
                                    "set_task_group_state", "debug_print", "queue_tasks"};
            for (int i = 0; i < 6; ++i) {
                uint8_t *lo, *hi;
                if (!ExecRange(vt[i], lo, hi)) break;
                LOGI("[m2]   vft[%d] %-22s rva 0x%llx", i, names[i],
                     (unsigned long long)((uint8_t*)vt[i] - base));
            }
        } else {
            g_table.vtable = nullptr;
            LOGW("[m2] object at %p has no verified vtable at +0 - set_task_group_state stays "
                 "unavailable this run", (void*)taskObj);
        }
    }

    g_table.valid = true;
    g_table.owner = taskObj;
    g_table.entry0 = entry0;
    g_table.stride = stride;
    g_table.count = count;
    LOGI("[m2] ---- probe complete ----");
    return true;
}

std::atomic<bool> g_started{false};

// ---- experiment: register our own tasks into the frame ----------------------------------
// fromsoftware-rs (Dasaav, MIT/Apache-2.0; crates/eldenring/src/cs/task.rs + shared/task.rs,
// the same rva_ww.rs our validated camera RVAs come from) proves the game's internal
// registration function accepts ANY object shaped like { vft, pad } whose vtable is
// { get_runtime_class, destructor, execute(this, FD4TaskData*) } - no CSEzTask, no proxy,
// no RTTI. Registration: register_task(CSTaskImp* imp, uint32_t groupIndex, task*), plain
// 0-based group INDEX (not the 0x90000000|idx id). The engine then calls execute every time
// that group runs, which is the in-frame execution point everything in M3 hangs off.
constexpr uint32_t RVA_RegisterTask = 0xeb1fe0;   // exe 2.6.2.0 (rva_ww.rs `register_task`)

// FD4TaskData (libER fd4_task.hpp): FD4Time{vptr,float} then group id then runner seed.
struct FD4TaskDataView {
    void*    timeVft;    // +0x00
    float    dt;         // +0x08
    float    _pad;       // +0x0c
    uint32_t groupId;    // +0x10  (0x90000000 | index)
    int      seed;       // +0x14  task runner 0..5
};

struct ErvrFrameTask {
    void**      vft;           // +0 - the game calls through this
    uint64_t    unk8 = 0;      // +8 - present in the proven layout; never seen written
    // ours, invisible to the engine:
    const char* name = "";
    volatile long calls = 0;
    float       lastDt = 0.f;
    uint32_t    lastGroupId = 0;
    int         lastSeed = 0;
    DWORD       lastThread = 0;
};

// The registered probes the toggle experiment reads. Deliberately leaked: the engine keeps
// the pointer forever and there is no unregister on our side yet.
ErvrFrameTask* g_probeTask[3] = {};   // FrameBegin, GameSceneDraw, DrawEnd
volatile __int64 g_drawEndQpc = 0;    // stamped by the DrawEnd probe; read by the queue_tasks probe

void* TaskGetRuntimeClass(void*) { return nullptr; }
void* TaskDtor(void*, unsigned) { return nullptr; }   // engine never destroys our tasks
void  TaskExecute(ErvrFrameTask* self, FD4TaskDataView* d) {
    // Runs on the game's task-runner threads. Keep it exception-free and tiny.
    const long n = InterlockedIncrement(&self->calls);
    if (d) {
        self->lastDt = d->dt;
        self->lastGroupId = d->groupId;
        self->lastSeed = d->seed;
    }
    self->lastThread = GetCurrentThreadId();
    if (self == g_probeTask[2]) {
        LARGE_INTEGER q; QueryPerformanceCounter(&q);
        g_drawEndQpc = q.QuadPart;
    }
    if (n == 1) {
        LOGI("[m2] task \"%s\": FIRST call (dt %.5f, groupId 0x%08X, seed %d, thread %lu)",
             self->name, self->lastDt, self->lastGroupId, self->lastSeed, self->lastThread);
        // The return chain is the engine's task-driver: it leads from the runner up to
        // whoever calls CSTask::queue_tasks each frame - the function a second draw-only
        // pass will have to hook. Logged module-relative so it survives ASLR.
        void* frames[16] = {};
        const USHORT got = RtlCaptureStackBackTrace(1, 16, frames, nullptr);
        auto* base = (uint8_t*)GetModuleHandleW(nullptr);
        char line[512]; int p = 0;
        for (USHORT i = 0; i < got && p < (int)sizeof(line) - 24; ++i) {
            HMODULE m = nullptr;
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)frames[i], &m);
            if (m == (HMODULE)base)
                p += snprintf(line + p, sizeof(line) - p, " exe+0x%llx",
                              (unsigned long long)((uint8_t*)frames[i] - base));
            else
                p += snprintf(line + p, sizeof(line) - p, " [%p]", frames[i]);
        }
        LOGI("[m2] task \"%s\" call stack:%s", self->name, line);
    }
}
void* g_taskVft[3] = {(void*)&TaskGetRuntimeClass, (void*)&TaskDtor, (void*)&TaskExecute};

// SEH-wrapped engine call; POD locals only.
int CallRegisterTask(void* fn, void* imp, uint32_t groupIndex, void* task) {
    __try {
        ((void(__fastcall*)(void*, uint32_t, void*))fn)(imp, groupIndex, task);
        return 1;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return -1; }
}

ErvrFrameTask* RegisterFrameTask(const char* name, int groupIndex) {
    if (!g_table.imp || groupIndex < 0 || !GameRva(RVA_RegisterTask)) return nullptr;
    // Require both the verified executable map and the matching 169-group task table.
    auto* base = (uint8_t*)GetModuleHandleW(nullptr);
    auto* t = new ErvrFrameTask();
    t->vft = g_taskVft;
    t->name = name;
    const int r = CallRegisterTask(base + GameRva(RVA_RegisterTask), g_table.imp, (uint32_t)groupIndex, t);
    if (r < 0) {
        LOGE("[m2] register_task(\"%s\", %d) FAULTED - verified task registration failed", name, groupIndex);
        return nullptr;   // t leaks; the engine may already hold it
    }
    LOGI("[m2] register_task(\"%s\", index %d) returned; waiting for first call", name, groupIndex);
    return t;
}

// ---- experiment: probe CSTask::queue_tasks --------------------------------------------
// Disassembly (2.6.2.0, rva 0x268c5d0): queue_tasks walks the cstg_control table, and for
// every entry whose `active` byte is set calls the group-queue fn (0x269ef80) keyed by the
// group id via a map at arg3+0x18, then writes [this+0xe0] to runnerMgr+0xb8 and kicks the
// 6 task runners. It IS the once-per-frame frame driver, and the active flags are consumed
// here - which the toggle experiment confirmed (our gated task stopped running). The two
// unknowns that decide mechanism B' (a second, draw-only pass per frame):
//   1. WHO calls it, on which thread - the hook point for a double pump.
//   2. Is it SYNCHRONOUS - has the whole frame finished when it returns? Measured by
//      comparing its exit timestamp against our DrawEnd task's last-call timestamp.
using PfnQueueTasks = uint64_t(__fastcall*)(void*, void*, void*);
PfnQueueTasks g_origQueueTasks = nullptr;
volatile long g_qtCalls = 0;
volatile __int64 g_qtLastEnterQpc = 0, g_qtLastLeaveQpc = 0;
void* volatile g_qtArgA = nullptr;   // captured args: the frame-driver context queue_tasks
void* volatile g_qtArgB = nullptr;   // receives every frame (b+0x18 = id->task_entry_group map)

// ---- experiment: double-pump the frame driver -------------------------------------------
// Run 3 proved the whole frame executes SYNCHRONOUSLY inside queue_tasks (~5 ms at 60 Hz on
// the main-loop thread) and that the active bytes gate exactly the groups' REGISTERED task
// lists, while the core world draw is carried by machinery the bytes do not gate. Both facts
// together make this the true-stereo mechanism: after the real frame returns, gate the sim
// range, call the frame driver AGAIN, restore. Pass 2 re-runs the frame chain (draw included)
// with every simulation task list empty - a draw-only engine frame at the same sim time.
volatile long g_pumpMode = 0;    // 0 off | 1 experiment: unconditional | 2 stereo: when a 2nd eye is stashed
volatile long g_pumpPasses = 0;
std::atomic<int> g_curPass{1};   // 1 = real frame, 2 = draw-only second pass
// Last present that arrived while g_curPass == 2 (stereo.cpp reports it). "Pass 2 called"
// and "engine actually ran pass 2 to Flip" are different facts - the 2026-08-22 wedged-
// runner incident is the proof - and this is the evidence for the second one.
std::atomic<ULONGLONG> g_p2PresentTick{0};
std::atomic<float> g_pumpLastMs{0.f};
DWORD g_qtThread = 0;            // the main-loop thread queue_tasks runs on
// Wall time of each engine pass's queue_tasks call ([0] = pass 1, [1] = pass 2), summed per
// heartbeat. Against the loop period it says how much of a stereo pair is spent in neither.
std::atomic<long long> g_passUs[2]{}, g_passMaxUs[2]{};
std::atomic<long>      g_passN[2]{};
void NotePassTime(int idx, long long qpcTicks) {
    static const long long f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x.QuadPart; }();
    const long long us = qpcTicks * 1000000 / f;
    g_passUs[idx].fetch_add(us, std::memory_order_relaxed);
    if (us > g_passMaxUs[idx].load(std::memory_order_relaxed)) g_passMaxUs[idx].store(us, std::memory_order_relaxed);
    g_passN[idx].fetch_add(1, std::memory_order_relaxed);
}

// Pass windows for the present-pairing probe: while g_winTrace is on, every pass-1 and
// pass-2 execution records its [enter, exit] QPC window.
struct PassWin { long long t0, t1; };
std::atomic<bool> g_winTrace{false};
PassWin g_p1Win[256], g_p2Win[256];
std::atomic<int> g_p1N{0}, g_p2N{0};

uint8_t ReadActiveRaw(int i) { return g_table.entry0[(size_t)i * g_table.stride + kActiveOff]; }
void    WriteActiveRaw(int i, uint8_t v) { g_table.entry0[(size_t)i * g_table.stride + kActiveOff] = v; }

// Which groups stay active in the draw-only second pass. Config-driven ("144-158,164" style
// ranges) so renderer-state experiments - exposure, Scaleform/UI, TAA behaviour - are an ini
// edit away instead of a rebuild; the ini hot-reloads.
std::mutex  g_pass2OverrideMtx;
std::string g_pass2Override;   // content-probe sweep: non-empty replaces [VR] Pass2Keep

void ParseKeepSpec(const char* spec, bool out[256]) {
    memset(out, 0, 256);
    const char* p = spec;
    while (*p) {
        while (*p == ',' || *p == ' ') ++p;
        if (!isdigit((unsigned char)*p)) break;
        int lo = (int)strtol(p, (char**)&p, 10), hi = lo;
        if (*p == '-') hi = (int)strtol(p + 1, (char**)&p, 10);
        for (int i = lo; i <= hi && i >= 0; ++i)
            if (i < 256) out[i] = true;
    }
}

// [VR] Pass2Characters=false: the second eye's pass skips the character pipeline, so it
// renders the poses pass 1 just computed - the same sim instant either way. Measured
// 2026-11-03 (9800X3D, 90 Hz Quest 3, standing in-world): ~1 ms of pass 2's ~4.6 ms of work,
// stereo depth intact (the camera chain is in MoveMap 118/123-125, which stays). The whole
// pipeline goes at once: pre-physics without post-physics crashed in August. Latched at the
// first pass - adding chr groups back mid-session poisoned queue_tasks (2026-08-22).
constexpr int kChrPipeline[] = {54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71,
                                72, 73, 74, 75, 76, 77, 78, 79, 81, 83, 84, 113, 114, 117, 127, 128, 129};

bool Pass2SkipsCharacters() {
    static const bool s_skip = [] {
        const bool skip = !Cfg().vrPass2Characters;
        if (skip)
            LOGI("[stereo] Pass2Characters=false: the second eye's pass skips the character "
                 "pipeline (groups 54-63, 65-79, 81, 83-84, 113-114, 117, 127-129) for this launch");
        return skip;
    }();
    return s_skip;
}

// The active keep set: a Numpad-2 override wins in either mode; otherwise the mode's own
// ini default (`defaultSpec`). Cached parse, re-logged whenever the effective spec changes.
const bool* Pass2KeepSet(const std::string& defaultSpec) {
    static std::string s_spec;
    static bool s_keep[256];
    std::string spec;
    {
        std::lock_guard<std::mutex> g(g_pass2OverrideMtx);
        spec = g_pass2Override.empty() ? defaultSpec : g_pass2Override;
    }
    // [VR] Pass2EventsOnce: the event manager (25) out of pass 2 whatever the set says, with the action-button
    // update (pass2_interact.cpp) - the two step together. Part of the cache key so a hot reload re-parses.
    const bool eventsOnce = Cfg().vrPass2EventsOnce;
    if (eventsOnce) spec += " |events once";
    if (spec != s_spec) {
        s_spec = spec;
        ParseKeepSpec(spec.c_str(), s_keep);
        if (Pass2SkipsCharacters())
            for (int g : kChrPipeline) s_keep[g] = false;
        if (eventsOnce) s_keep[25] = false;
        std::string names;
        for (int i = 0; i < g_table.count && i < 256; ++i)
            if (s_keep[i]) { names += std::to_string(i); names += ' '; }
        LOGI("[stereo] pass-2 keeps groups: %s", names.c_str());
    }
    return s_keep;
}

std::atomic<bool> g_probePitchOn{false};   // content probe: tilt ONLY pass 2's camera

// ---- GPU serialisation between the two engine passes -----------------------------------
// The pump starts pass 2 on the CPU the moment pass 1's queue_tasks returns - while the GPU
// (GPU-bound at ~45 Hz in stereo) is still executing pass 1's draws. The engine sizes its
// per-frame upload memory for frames-in-flight keyed by a frame index; if that advance
// lives in a group pass 2 gates, both passes write the SAME ring slot and pass 2's constant
// uploads land while the GPU is mid-read on pass 1's - per-object corruption, worst in
// pass 1's image (its data is overwritten ~1 ms after submit), a different, lighter set in
// pass 2's (overwritten by the NEXT frame's pass 1, a whole frame later). [VR] GpuSync
// drains the present queue with a fence so the passes cannot overlap on the GPU. It
// serialises CPU against GPU - real cost - but it is hot-togglable (Numpad 4), so the
// hypothesis is a ten-second in-headset test: 1 should clean up the left eye, 2 both.
std::atomic<ID3D12CommandQueue*> g_gpuQueue{nullptr};
ID3D12Fence*  g_syncFence = nullptr;
ID3D12Device* g_syncFenceDev = nullptr;   // identity only, never dereferenced
HANDLE        g_syncEvent = nullptr;
uint64_t      g_syncVal = 0;
std::atomic<float> g_syncLastMs{0.f};

int GpuSyncLevel() {
    const int k = DebugKeysGpuSync();
    return k >= 0 ? k : Cfg().vrGpuSync;
}

// Signal-and-wait on the game's present queue. Runs on the main-loop thread between engine
// passes - the same thread that presents, so this is like the game's own vsync wait, NOT a
// block inside the ReShade present event (which crashed D3D12Core; see dx12_blit notes).
void PumpGpuDrain(const char* where) {
    ID3D12CommandQueue* q = g_gpuQueue.load(std::memory_order_acquire);
    if (!q) return;
    ID3D12Device* dev = nullptr;
    if (FAILED(q->GetDevice(IID_PPV_ARGS(&dev))) || !dev) return;
    if (g_syncFence && g_syncFenceDev != dev) {   // device changed under us
        g_syncFence->Release();
        g_syncFence = nullptr;
    }
    if (!g_syncFence) {
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_syncFence)))) {
            g_syncFence = nullptr;
            dev->Release();
            return;
        }
        g_syncFenceDev = dev;
        g_syncVal = 0;
        if (!g_syncEvent) g_syncEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        LOGI("[stereo] GpuSync: fence created on the present queue");
    }
    dev->Release();
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    const uint64_t v = ++g_syncVal;
    if (FAILED(q->Signal(g_syncFence, v))) return;
    if (g_syncFence->GetCompletedValue() < v && g_syncEvent) {
        g_syncFence->SetEventOnCompletion(v, g_syncEvent);
        if (WaitForSingleObject(g_syncEvent, 2000) != WAIT_OBJECT_0) {
            LOGW("[stereo] GpuSync %s: fence wait timed out (device hung or removed?)", where);
            return;
        }
    }
    QueryPerformanceCounter(&t1);
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    g_syncLastMs.store((float)((double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart),
                       std::memory_order_relaxed);
}

// ---- pass-2 runtime dt swap ------------------------------------------------------------
// The framelock imm is an INSTRUCTION operand: `mov [reg+disp], 1/60f` in the frame loop,
// executed once per REAL frame BEFORE queue_tasks. Swapping the imm around pass 2 therefore
// changed nothing - the instruction never re-executes inside the pass, and pass 2 ran a
// FULL-dt engine frame (the 2026-08-21 "everything runs fast" report; gating experiments
// chased per-invocation steppers that were never the cause). The fix: find where that
// instruction wrote this frame's dt - the runtime slot(s) inside the queue_tasks argument
// structs - by scanning them for the exact framelock bit pattern, then overwrite those
// slots with Pass2Dt around the second call. Offsets are remembered (never raw pointers)
// and every write is preceded by a bit-exact match check, so a reallocated struct or a
// changed layout degrades to "no swap this pass", not corruption.
struct DtSlot { int base; int deep; uint32_t ptrOff; uint32_t off; };
DtSlot g_dtSlots[16];
int    g_dtSlotCount = -1;       // -1 = not scanned yet

// SEH-wrapped primitives: POD locals only, no logging inside.
int TryReadU32(uint8_t* p, uint32_t* out) {
    __try { *out = *(volatile uint32_t*)p; return 1; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}
int TryWriteU32(uint8_t* p, uint32_t v) {
    __try { *(volatile uint32_t*)p = v; return 1; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}
int TryWriteU8(uint8_t* p, uint8_t v) {
    __try { *(volatile uint8_t*)p = v; return 1; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}
int ScanFlat(uint8_t* p, size_t len, uint32_t bits, uint32_t* outOff, int maxOut) {
    int n = 0;
    __try {
        for (size_t o = 0; o + 4 <= len && n < maxOut; o += 4)
            if (*(uint32_t*)(p + o) == bits) outOff[n++] = (uint32_t)o;
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return n;
}
uint8_t* DerefPtr(uint8_t* p, uint32_t off) {
    __try {
        uint8_t* q = *(uint8_t**)(p + off);
        return ((uintptr_t)q > 0x10000 && (uintptr_t)q < 0x7FFFFFFF0000ull) ? q : nullptr;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
}

// Hardware-breakpoint capture of the engine's frame-timing struct. The dt write is
// `mov [rbx+disp8], imm32` (C7 43 dd ii ii ii ii) at PatchesFrameDtImm()-3, executed on the
// frame thread each frame. A one-shot execute breakpoint there hands us RBX = the timing
// struct; pass 2 then swaps [rbx+disp8] directly. Version-proof: address and disp8 are read
// from the live instruction, nothing hard-coded.
std::atomic<uint8_t*> g_dtStruct{nullptr};
std::atomic<int>      g_dtBpState{0};   // 0 idle | 1 armed | 2 captured | -1 failed
uint8_t               g_dtDisp = 0;

// The staged-steps copy `mov [rbx+0xdc], eax` (exe+0xb19633 on 2.6.2.0, AOB-resolved):
// runs twice per OUTER frame before the passes, staging the frame's step count into the
// task-runner object(s) whose dispatch the passes then consume. Dr1 captures the rbx
// destinations (dedup, up to 4) and disarms after a dozen hits.
uint8_t*          g_stepsCopyIns = nullptr;
uint8_t*          g_stepsDst[4] = {};
std::atomic<int>  g_stepsDstN{0};
std::atomic<int>  g_stepsHits{0};

// FD4TaskData capture (Dr2): the IGT caller reads dt via `movss xmm0,[rdi+8]` - rdi IS the
// FD4TaskData handed to the task body. Capture a handful of hits to learn the pointer AND
// whether it is stable across frames (a reused member => patchable between passes).
uint8_t*              g_fd4Ins = nullptr;
std::atomic<uint8_t*> g_fd4Data{nullptr};
std::atomic<int>      g_fd4Hits{0};
uint8_t*              g_fd4Seen[4] = {};
std::atomic<int>      g_fd4SeenN{0};

LONG CALLBACK DtVeh(EXCEPTION_POINTERS* xp) {
    if (xp->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP)
        return EXCEPTION_CONTINUE_SEARCH;
    uint8_t* addr = (uint8_t*)xp->ExceptionRecord->ExceptionAddress;
    float* imm = PatchesFrameDtImm();
    if (imm && addr == (uint8_t*)imm - 3) {
        g_dtStruct.store((uint8_t*)xp->ContextRecord->Rbx, std::memory_order_release);
        g_dtBpState.store(2, std::memory_order_release);
        xp->ContextRecord->Dr0 = 0;                   // disarm: one shot is all we need
        xp->ContextRecord->Dr7 &= ~1ull;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (g_fd4Ins && addr == g_fd4Ins) {
        uint8_t* rdi = (uint8_t*)xp->ContextRecord->Rdi;
        g_fd4Data.store(rdi, std::memory_order_release);
        const int n = g_fd4SeenN.load(std::memory_order_relaxed);
        bool known = false;
        for (int i = 0; i < n; ++i) if (g_fd4Seen[i] == rdi) known = true;
        if (!known && n < 4) { g_fd4Seen[n] = rdi; g_fd4SeenN.store(n + 1, std::memory_order_release); }
        // One-time: the task-data is stack-built, so name the BUILDER - scan the caller
        // frames for .text return addresses (same trick as PanicVeh).
        static std::atomic<bool> s_chainLogged{false};
        if (!s_chainLogged.exchange(true)) {
            auto* base = (uint8_t*)GetModuleHandleW(nullptr);
            uint64_t* sp = (uint64_t*)xp->ContextRecord->Rsp;
            char line[500]; int lp = 0; line[0] = 0;
            for (int k = 0; k < 96 && lp < 440; ++k) {
                const uint64_t v = sp[k];
                if (v > (uint64_t)(uintptr_t)base + 0x1000 &&
                    v < (uint64_t)(uintptr_t)base + 0x29a2c00)
                    lp += snprintf(line + lp, sizeof(line) - lp, " +0x%llx",
                                   (unsigned long long)(v - (uint64_t)(uintptr_t)base));
            }
            LOGI("[stereo] fd4 dispatch chain (stack scan, fd4=%p):%s", (void*)rdi, line);
        }
        xp->ContextRecord->EFlags |= 1u << 16;
        if (g_fd4Hits.fetch_add(1, std::memory_order_relaxed) + 1 >= 16) {
            xp->ContextRecord->Dr2 = 0;
            xp->ContextRecord->Dr7 &= ~0x10ull;
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (g_stepsCopyIns && addr == g_stepsCopyIns) {
        uint8_t* rbx = (uint8_t*)xp->ContextRecord->Rbx;
        const int n = g_stepsDstN.load(std::memory_order_relaxed);
        bool known = false;
        for (int i = 0; i < n; ++i) if (g_stepsDst[i] == rbx) known = true;
        if (!known && n < 3) { g_stepsDst[n] = rbx; g_stepsDstN.store(n + 1, std::memory_order_release); }
        xp->ContextRecord->EFlags |= 1u << 16;        // RF: run this instruction this once
        if (g_stepsHits.fetch_add(1, std::memory_order_relaxed) + 1 >= 12) {
            xp->ContextRecord->Dr1 = 0;
            xp->ContextRecord->Dr7 &= ~4ull;
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void ArmDtBreakpoint(DWORD tid) {
    uint8_t* ins = (uint8_t*)PatchesFrameDtImm() - 3;
    if (ins[0] != 0xC7 || ins[1] != 0x43) {           // expect mov [rbx+disp8], imm32
        LOGW("[stereo] pass-2 dt: unexpected dt-write encoding %02X %02X - hw-bp capture skipped",
             ins[0], ins[1]);
        g_dtBpState.store(-1);
        return;
    }
    g_dtDisp = ins[2];
    static bool s_veh = false;
    if (!s_veh) { s_veh = true; AddVectoredExceptionHandler(1, DtVeh); }
    // Second breakpoint: the staged-steps copy (see g_stepsCopyIns). AOB keeps it
    // version-proof; +10 skips the rip-relative load and the [rax+0x18] read.
    if (uint8_t* hit = AobScan("48 8B 05 ?? ?? ?? ?? 8B 40 18 89 83 DC 00 00 00 C6 83 D8 00 00 00 01"))
        g_stepsCopyIns = hit + 10;
    else
        LOGW("[stereo] pass-2 steps: staged-copy pattern not found; destination capture off");
    // The IGT caller's dt load: movss xmm0,[rdi+8]; call ..; movss xmm1,[rdi+8].
    if (uint8_t* hit = AobScan("F3 0F 10 47 08 E8 ?? ?? ?? ?? F3 0F 10 4F 08"))
        g_fd4Ins = hit;
    else
        LOGW("[stereo] pass-2 dt: FD4TaskData load pattern not found; fd4 capture off");
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT,
                          FALSE, tid);
    if (!h) { g_dtBpState.store(-1); LOGW("[stereo] pass-2 dt: OpenThread(%lu) failed", tid); return; }
    bool ok = false;
    if (SuspendThread(h) != (DWORD)-1) {
        CONTEXT c{};
        c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        if (GetThreadContext(h, &c)) {
            c.Dr0 = (DWORD64)(uintptr_t)ins;
            c.Dr7 = (c.Dr7 & ~(0xFull << 16)) | 1;    // L0 enabled, exec, len 1
            if (g_stepsCopyIns) {
                c.Dr1 = (DWORD64)(uintptr_t)g_stepsCopyIns;
                c.Dr7 = (c.Dr7 & ~(0xFull << 20)) | 4;   // L1 enabled, exec, len 1
            }
            // Dr2 (FD4TaskData capture) stays DISARMED: it served its diagnostic purpose
            // (proved the task data is stack-built and named the builder), and while armed
            // it kept overwriting g_fd4Seen with stack pointers. Re-enable only for RE.
            ok = SetThreadContext(h, &c) != 0;
        }
        ResumeThread(h);
    }
    CloseHandle(h);
    if (ok) LOGI("[stereo] pass-2 dt: hw breakpoints armed at %p + %p (frame thread %lu, disp 0x%02X)",
                 ins, g_stepsCopyIns, tid, g_dtDisp);
    else { g_dtBpState.store(-1); LOGW("[stereo] pass-2 dt: setting debug registers failed"); }
}

// The engine's OWN frame-rate override, found in the timing update's tail (2.6.2.0 rva
// 0xe82c1c..0xe82c53, reached every frame AFTER the adaptive dt switch): when the byte at
// [struct+0x2c8] is set and [struct+0x2c9] is clear (and the mode at [struct+0xc] != 6),
// the engine itself computes dt = 1.0f / [struct+0x2c4] - overriding whatever the switch
// picked. This is FromSoft's dormant debug override and the ONLY reliable dt control:
// patching the framelock imm covers one switch case, and the engine falls back to the
// UNPATCHED 1/60 case whenever the loop misses the target (measured 2026-08-21: pumped
// windows at 58 pairs/s ran IGT at 1.75x with the imm at 1/120). With the override armed
// at 120 Hz, every pass advances exactly 1/120 s regardless of load, so a stereo pair is
// always 1/60 - realtime, degrading to smooth slow-motion (never fast-forward) under load.
// DO NOT arm the [struct+0x2c8] "debug fps override" byte: tried 2026-08-21 - instead of
// pinning dt to 1/[struct+0x2c4] it latched the whole session into a persistent 4x-speed
// state (dt 1/30 at 120 presents/s) that clearing the byte did NOT undo. Whatever that
// path latches is not understood; treat it as radioactive.

void ScanDtSlots(uint8_t* a, uint8_t* b, uint32_t bits) {
    g_dtSlotCount = 0;
    uint8_t* bases[2] = {a, b};
    char msg[256]; int mp = 0; msg[0] = 0;
    for (int bi = 0; bi < 2 && g_dtSlotCount < 16; ++bi) {
        if (!bases[bi]) continue;
        uint32_t offs[8];
        const int n = ScanFlat(bases[bi], 0x800, bits, offs, 8);
        for (int k = 0; k < n && g_dtSlotCount < 16; ++k) {
            g_dtSlots[g_dtSlotCount++] = {bi, 0, 0, offs[k]};
            mp += snprintf(msg + mp, sizeof(msg) - mp, " %c+0x%x", bi ? 'b' : 'a', offs[k]);
        }
    }
    if (!g_dtSlotCount) {
        // One pointer hop: the dt may live in a struct the args point AT rather than in them.
        for (int bi = 0; bi < 2 && g_dtSlotCount < 16; ++bi) {
            if (!bases[bi]) continue;
            for (uint32_t po = 0; po < 0x200 && g_dtSlotCount < 16; po += 8) {
                uint8_t* q = DerefPtr(bases[bi], po);
                if (!q) continue;
                uint32_t offs[4];
                const int n = ScanFlat(q, 0x200, bits, offs, 4);
                for (int k = 0; k < n && g_dtSlotCount < 16; ++k) {
                    g_dtSlots[g_dtSlotCount++] = {bi, 1, po, offs[k]};
                    mp += snprintf(msg + mp, sizeof(msg) - mp, " [%c+0x%x]+0x%x",
                                   bi ? 'b' : 'a', po, offs[k]);
                }
            }
        }
    }
    if (g_dtSlotCount)
        LOGI("[stereo] pass-2 runtime dt: %d slot(s) matching frame dt bits 0x%08X:%s "
             "- pass 2 will run at Pass2Dt", g_dtSlotCount, bits, msg);
    else
        LOGW("[stereo] pass-2 runtime dt: NO slot matching 0x%08X found in queue_tasks args "
             "(direct 0x800 + one hop 0x200) - pass 2 keeps the REAL dt and runs fast", bits);
}

// ---- baked-entry dt candidates -----------------------------------------------------------
// Every upstream patch (timing-struct float, steps field, staged copies, dispatch [obj+
// 0x1a8]) left the per-pass advance untouched: the frame's dt is BAKED into per-entry
// structures before queue_tasks runs, and pass 2 replays the baked list. So find the baked
// values themselves: scan the dispatch object and its 0x170-stride entry array for numbers
// that look like a frame dt (float or double), remember the addresses, and zero them for
// pass 2 with plausibility re-checks (entries may be rebuilt; a stale address then just
// fails the check and is skipped).
struct BakedDt { uint8_t* p; int dbl; };
BakedDt g_baked[32];
int     g_bakedN = -1;

int PlausibleDtF(uint32_t bits) {
    float v; memcpy(&v, &bits, 4);
    return v > 0.0005f && v < 0.12f;
}
int TryReadU64(uint8_t* p, uint64_t* out) {
    __try { *out = *(volatile uint64_t*)p; return 1; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}
int TryWriteU64(uint8_t* p, uint64_t v) {
    __try { *(volatile uint64_t*)p = v; return 1; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}
int ScanBaked(uint8_t* base, size_t len, float ref, BakedDt* out, int maxOut) {
    int n = 0;
    __try {
        for (size_t o = 0; o + 8 <= len && n < maxOut; o += 4) {
            float f = *(float*)(base + o);
            if (f > 0.f && f<0.12f && f> 0.0005f && f > ref * 0.5f && f < ref * 2.f)
                out[n++] = {base + o, 0};
            if ((o & 7) == 0) {
                double d = *(double*)(base + o);
                if (d > 0.0005 && d < 0.12 && d > ref * 0.5 && d < ref * 2.0 && n < maxOut)
                    out[n++] = {base + o, 1};
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return n;
}

// ---- IGT-advance trace -------------------------------------------------------------------
// [Experimental] IgtTrace: hook the IGT-advance function (takes xmm0 = this call's dt in
// seconds) and log its CALLER + dt + pass for the first distinct callers. The caller's
// disassembly then shows where the per-pass dt is actually loaded from - the ground truth
// the upstream patches kept missing.
using PfnIgtAdvance = void(__fastcall*)(float);
PfnIgtAdvance g_origIgtAdvance = nullptr;

void __fastcall HkIgtAdvance(float dt) {
    static std::atomic<long> s_n{0};
    const long n = s_n.fetch_add(1, std::memory_order_relaxed);
    if (n < 24) {
        auto* base = (uint8_t*)GetModuleHandleW(nullptr);
        uint8_t* ret = (uint8_t*)_ReturnAddress();
        LOGI("[igt] advance #%ld: dt=%.6f pass=%d caller=exe+0x%llx", n, dt,
             FramePumpCurrentPass(),
             (unsigned long long)(ret - base));
    }
    g_origIgtAdvance(dt);
}

void InstallIgtTrace() {
    if (!Cfg().igtTrace) return;
    uint8_t* fn = AobScan("48 83 EC 58 48 C7 44 24 20 FE FF FF FF 0F 29 74 24 40 0F 28 F0 48 8B 0D");
    if (!fn) { LOGW("[igt] advance-function pattern not found"); return; }
    if (MH_CreateHook(fn, (void*)&HkIgtAdvance, (void**)&g_origIgtAdvance) == MH_OK &&
        MH_EnableHook(fn) == MH_OK)
        LOGI("[igt] advance hook installed at %p", fn);
    else
        LOGW("[igt] advance hook failed");
}

// Writes newDt into every remembered slot whose current value is bit-exactly expectBits.
// Returns a mask of slots written; RestoreDtSlots puts expectBits back into exactly those.
uint32_t ApplyDtSlots(uint8_t* a, uint8_t* b, uint32_t expectBits, float newDt) {
    uint8_t* bases[2] = {a, b};
    uint32_t newBits; memcpy(&newBits, &newDt, 4);
    uint32_t mask = 0;
    for (int k = 0; k < g_dtSlotCount; ++k) {
        uint8_t* p = bases[g_dtSlots[k].base];
        if (!p) continue;
        if (g_dtSlots[k].deep && !(p = DerefPtr(p, g_dtSlots[k].ptrOff))) continue;
        p += g_dtSlots[k].off;
        uint32_t cur;
        if (!TryReadU32(p, &cur) || cur != expectBits) continue;
        if (TryWriteU32(p, newBits)) mask |= 1u << k;
    }
    if (uint8_t* s = g_dtStruct.load(std::memory_order_acquire)) {
        uint8_t* p = s + g_dtDisp;
        uint32_t cur;
        if (TryReadU32(p, &cur) && cur == expectBits && TryWriteU32(p, newBits)) {
            mask |= 0x80000000u;
            static bool s_logged = false;
            if (!s_logged) {
                s_logged = true;
                LOGI("[stereo] pass-2 dt: swapping timing struct [%p+0x%02X] per pass "
                     "- pass 2 now really runs at Pass2Dt", (void*)s, g_dtDisp);
            }
        }
    }
    return mask;
}
void RestoreDtSlots(uint8_t* a, uint8_t* b, uint32_t mask, uint32_t expectBits) {
    uint8_t* bases[2] = {a, b};
    for (int k = 0; k < g_dtSlotCount; ++k) {
        if (!(mask & (1u << k))) continue;
        uint8_t* p = bases[g_dtSlots[k].base];
        if (!p) continue;
        if (g_dtSlots[k].deep && !(p = DerefPtr(p, g_dtSlots[k].ptrOff))) continue;
        TryWriteU32(p + g_dtSlots[k].off, expectBits);
    }
    if (mask & 0x80000000u) {
        if (uint8_t* s = g_dtStruct.load(std::memory_order_acquire))
            TryWriteU32(s + g_dtDisp, expectBits);
    }
}

// Set by FlowTransitionCheck (defined below, run at every pass-1 entry) whenever the
// ENGINE changes a task-group active byte - a gameflow transition marker.
ULONGLONG g_flowChangeTick = 0;
// Last frame a WORLD-DISTURBING pass-2 guard held (menu-only holds do not stamp this - see
// menuHold in DoSecondPass). For vrPass2SafeSeconds after this, pass 2 runs the SAFE keep
// set (no MoveMap_PostPhysics family): the world is still settling after whatever the
// guard was for, and that family walking a settling world is every transition crash of
// 2026-08-22. Cost: depth returns ~20 s after loads instead of instantly.
ULONGLONG g_lastHoldTick = 0;
std::atomic<bool> g_safeSetActive{false};

// Menu-resume grace. Full-screen menus stop the game stashing second-eye cameras (those
// frames are single-pass by design) - and the map, the only road to fast travel, is such a
// menu. Both 2026-08-22 headset crashes fired within ~200 ms of the travel CONFIRM: the map
// closes, stashing resumes that same instant, and pass 2 ran straight into the transfer
// request (pass-1 null deref at exe+0xc7f88b). FOV-based menu detection failed twice
// because its probe - our per-frame FOV write - is disabled on exactly those single-pass
// frames; the stash gap is the signal that cannot lie. A gap longer than ~0.7 s is a real
// menu (single dropped frames happen; menus last seconds), and when the stash returns,
// pass 2 stays out for 6 more seconds: long enough to bridge a confirm into the fade and
// the player-null teardown that follow, short enough that closing the map without
// traveling costs only a moment of mono.
bool MenuResumeGraceActive() {
    static ULONGLONG s_gapStart = 0;
    static ULONGLONG s_graceUntil = 0;
    const ULONGLONG now = GetTickCount64();
    if (!GameERSecondEyePending()) {
        if (!s_gapStart) s_gapStart = now;
    } else {
        if (s_gapStart && now - s_gapStart > 700) {
            s_graceUntil = now + (ULONGLONG)(Cfg().vrPass2MenuGraceSeconds * 1000.f);
            LOGI("[stereo] second-eye stash resumed after a %llu ms gap (menu closed) - "
                 "%.0f s grace before pass 2", (unsigned long long)(now - s_gapStart),
                 Cfg().vrPass2MenuGraceSeconds);
        }
        s_gapStart = 0;
    }
    return now < s_graceUntil;
}

void DoSecondPass(void* self, void* a, void* b) {
    // Stereo mode: only pump frames for which the game stashed the second eye's camera -
    // and write that camera FIRST, while every writer is quiescent. Menus, loading and
    // cinema mode simply never stash one, so those frames stay single-pass.
    // The guard block runs for production stereo (mode 2) AND for the ForcePass2 test
    // harness (mode 1 with the flag set). That is what makes the harness meaningful: the
    // 2026-08-22 headless repro crashed at the exact production address precisely BECAUSE
    // mode 1 used to skip these guards - so a guard change validated under ForcePass2 is
    // validated for the real thing.
    int coopOthers = 0;   // other players in the world (Seamless Co-op); read by the SAFE choice below
    if (g_pumpMode == 2 || (g_pumpMode == 1 && Cfg().m2ForcePass2 && !Cfg().m2Pass2NoGuards)) {
        // Which guard (if any) is holding pass 2 out, logged only when the answer CHANGES.
        // Without this a warp crash is undiagnosable after the fact: the log cannot say
        // whether a guard fired too late or never fired at all, which is exactly the hole
        // that made the 2026-08-22 crash hunt take three attempts. One line per transition
        // is free - normal play never changes state.
        const char* hold = nullptr;
        // A hold that is ONLY about a menu being (or having just been) open. The world was
        // never disturbed, so it does not stamp the SAFE settle timer below: stamping it
        // bought 20 s of SAFE after every map peek and inventory close, which - together
        // with the ungated A-window - kept the 19:44 session bouncing SAFE<->full all
        // session (the visible flat/stereo popping). A menu that DOES lead to travel runs
        // into the teleport/flow/fade/player-null guards, and those still stamp.
        bool menuHold = false;
        // Ticked EVERY frame, not inside the else-if chain: the gap tracker has to see the
        // stash reappear even while another guard is holding, or it re-arms with stale
        // state after every load and taxes the respawn with 6 extra mono seconds.
        const bool menuGrace = g_pumpMode == 2 && MenuResumeGraceActive();
        // Sampled EVERY frame (not inside the else-if chain) so the warp latch arms and the
        // signal edges get logged even while another guard already holds - a transition
        // postmortem needs the full signal timeline, and the latch must not miss a flag
        // that is set and consumed while e.g. the death guard is the one holding.
        const bool warpGuard = GameERWarpGuardActive();
        // Same every-frame sampling: a join that lands while another guard holds must still
        // start its own grace.
        const bool coopHold = GameERCoopRoster(Cfg().vrPass2CoopGraceSeconds, &coopOthers);
        static ULONGLONG s_lastInactive = GetTickCount64();
        // Spawn/teleport guard: pumping within seconds of the player appearing crashed the
        // game at exe+0x1e8b0bc (a draw fn reading a per-frame input that the spawn-fade
        // machinery - gated in pass 2 - had not produced). Give every player-inactive ->
        // active transition a grace period before the second pass resumes.
        if (!GameERPlayerActive()) { s_lastInactive = GetTickCount64(); hold = "player-null (world torn down)"; }
        // Death guard: HP hit 0, so a death warp starts THIS frame. The only guard that can
        // cover the request frame of a death - everything below reacts to consequences.
        else if (!Cfg().m2Pass2NoDeadGuard && GameERPlayerDead()) hold = "player dead (death warp incoming)";
        // Configurable since 2026-08-22 evening (was a hardcoded 15 s): both crashes that
        // sized it - the spawn-fade draw crash exe+0x1e8b0bc (re-attributed to the readback
        // experiment) and the 11 s arrival crash exe+0xc7f88b (root cause: the blanket
        // keep-set restore, fixed) - have root causes that no longer exist, and the user
        // asked for depth sooner. See config.h at vrPass2SpawnGraceSeconds.
        else if (GetTickCount64() - s_lastInactive <
                 (ULONGLONG)(Cfg().vrPass2SpawnGraceSeconds * 1000.f)) hold = "player-respawn grace";
        // Warp-request guard (2026-08-23, the fundamental transition signal): the game's
        // own GameMan.warp_requested byte, set on the request frame of every warp - the
        // exact frame the A-press heuristic (Pass2SafeOnA) was built to approximate, with
        // zero false positives from ordinary interact/jump presses. Latched for
        // Pass2WarpGraceSeconds so the hold bridges into player-null/loading below.
        else if (warpGuard) hold = "warp requested (GameMan flag)";
        // Loading-screen guard: the community loading byte - TRUE for the whole teardown/
        // stream/rebuild window. State, not a timer: releases the moment the world is back.
        else if (GameERLoadingScreen()) hold = "loading screen (event byte)";
        // Co-op connect guard: the game's own join state machine is in flight - it starts
        // before the character re-initialization the connect crashes hit.
        else if (GameERCoopConnecting(Cfg().vrPass2CoopGraceSeconds)) hold = "co-op session connecting (multiplayer join protocol)";
        // Co-op roster guard: another player's character appeared, left or changed load
        // state - a character-grid rebuild with none of the signals above.
        else if (coopHold) hold = "co-op roster change (a player joining, leaving or loading)";
        // Foreign-camera guard: a menu/cutscene camera owns the FOV. The map screen is the
        // only road to fast travel, so this is what covers the fast-travel REQUEST frame -
        // the window where the 13:58 crash fired with no guard holding.
        else if (GameERForeignCamActive()) { hold = "foreign camera (map/cutscene)"; menuHold = true; }
        // Stash-gap guard: a menu just closed (see MenuResumeGraceActive above). This is
        // what covers the fast-travel confirm - the frame the map closes on.
        else if (menuGrace) { hold = "menu-resume grace"; menuHold = true; }
        // Teleport guard: fast travel tears the world down around a still-active player;
        // running a second pass in that window fires (previously fatal) engine asserts.
        else if (GameERTeleportGraceActive()) hold = "teleport grace";
        // Flow-transition guard: the engine flipped task lines recently (map transfer,
        // cutscene, flow change) - the world is mid-transition, sit the pass out.
        else if (g_flowChangeTick && GetTickCount64() - g_flowChangeTick < 2500) hold = "flow transition";
        // Fade guard: the screen just faded to black - fast travel / death warp teardown
        // is underway (both fast-travel crashes fired in exactly this window).
        else if (StereoFadeGuardActive()) hold = "fade guard";
        {
            static const char* s_prevHold = nullptr;
            if (hold != s_prevHold) {
                if (hold) LOGI("[stereo] pass 2 SUSPENDED: %s", hold);
                else      LOGI("[stereo] pass 2 resumed (no guard holding; SAFE keep set for "
                               "%.0f s, then full stereo depth)", Cfg().vrPass2SafeSeconds);
                s_prevHold = hold;
            }
        }
        if (hold) { if (!menuHold) g_lastHoldTick = GetTickCount64(); return; }
    }
    if (g_pumpMode == 2) {
        if (!GameERApplySecondEyeCamera()) return;
    }
    // Content probe (experiment mode): make pass 2's render unmistakable - the harness then
    // reads pixels back at each Present to learn which pass's image that Present carries.
    if (g_pumpMode == 1 && g_probePitchOn.load(std::memory_order_relaxed))
        GameERDebugPitchCamera(80.f);
    // Let the GPU finish pass 1 (its draws AND our left-eye blit, both on the present queue)
    // before pass 2's queue_tasks re-uploads the per-frame GPU data those draws read.
    const int gpuSync = GpuSyncLevel();
    if (gpuSync >= 1) PumpGpuDrain("pre-pass2");
    // Two shapes for the second pass:
    //  * draw-only (default): keep the draw phase + camera propagation, gate everything
    //    else. Cheap, but the environment lighting never found its producer this way.
    //  * full-frame ([VR] Pass2FullFrame / Numpad 5): run a COMPLETE engine frame with the
    //    game's per-frame dt swapped to ~0 - every producer runs, nothing advances. Only
    //    PadStep (input edge replay), the world-origin shift pair and AI GC stay gated.
    const int keyFF = DebugKeysPass2FullFrame();
    const bool fullFrame = keyFF >= 0 ? keyFF != 0 : Cfg().vrPass2FullFrame;
    float dtSaved = 0.f;
    bool dtSwapped = false;
    const ULONGLONG nowTick = GetTickCount64();
    const ULONGLONG aTick = VirtualPadLastATick();
    const bool inSettle =
        (g_lastHoldTick && nowTick - g_lastHoldTick < (ULONGLONG)(Cfg().vrPass2SafeSeconds * 1000.f)) ||
        // A-press window: fast-travel (and every other dangerous confirm) is an A press.
        (Cfg().vrPass2SafeOnA > 0.f && aTick &&
         nowTick - aTick < (ULONGLONG)(Cfg().vrPass2SafeOnA * 1000.f)) ||
        // Other players in the world: their characters stream in and out of the grids the
        // MoveMap family walks at any moment, so co-op stays on the SAFE set (finalize replay
        // keeps the depth).
        (Cfg().vrPass2CoopSafe && (coopOthers > 0 || GameERCoopInSession()));
    // SAFE is also simply "the configured full set IS the safe set" - the intended end
    // state once the finalize replay proves depth without the MoveMap family.
    const bool useSafe = inSettle || Cfg().vrPass2FullFrameKeep == Cfg().vrPass2SafeKeep;
    g_safeSetActive.store(fullFrame && useSafe, std::memory_order_relaxed);
    const bool* keepSet = Pass2KeepSet(!fullFrame     ? Cfg().vrPass2Keep
                                       : useSafe      ? Cfg().vrPass2SafeKeep
                                                      : Cfg().vrPass2FullFrameKeep);
    // Pass2Dt <= 0 = HALF-DT MODE: no freeze attempt at all. The production combination is
    // [Patches] FpsTarget=120 + full-frame pass 2: each engine frame advances 1/120 s, so a
    // stereo PAIR advances exactly 1/60 - realtime by construction, every system at correct
    // speed, eyes 8.3 ms apart in sim time. Chosen after the freeze approach failed twice
    // (2026-08-20/21): the framelock imm never re-executes inside the pass, and even the
    // hw-bp-captured timing struct [rbx+0x1C] is COPIED downstream before queue_tasks (IGT
    // ratio stayed ~1.8x with the struct swapped) - the copy chain is not worth chasing.
    if (fullFrame && Cfg().vrPass2Dt > 0.f) {
        float* imm = PatchesFrameDtImm();
        if (imm) {
            memcpy(&dtSaved, imm, 4);
            dtSwapped = PatchesWriteFrameDt(Cfg().vrPass2Dt);
        }
        static bool s_ffLogged = false;
        if (!s_ffLogged) {
            s_ffLogged = true;
            LOGI("[stereo] pass 2 FULL-FRAME mode: dt %s (saved %.5f -> %.7f)",
                 dtSwapped ? "swapped" : "NOT SWAPPED (framelock imm unresolved!)",
                 dtSaved, Cfg().vrPass2Dt);
        }
    }
    // The imm swap above only matters if the dt-writing instruction re-executes inside the
    // pass; measured 2026-08-21 it does NOT - the RUNTIME slots are what pass 2 reads.
    // Capture the timing struct once (one-shot hw breakpoint; we ARE the frame thread here,
    // so it is planted from a helper thread). Needed by the dt CamWatch and any future
    // per-pass dt control.
    if (fullFrame) {
        int expected = 0;
        if (g_dtBpState.compare_exchange_strong(expected, 1)) {
            const DWORD tid = GetCurrentThreadId();
            std::thread(ArmDtBreakpoint, tid).detach();
        }
    }
    uint32_t dtBits = 0, dtSlotMask = 0;
    if (fullFrame && dtSwapped) {
        memcpy(&dtBits, &dtSaved, 4);
        if (g_dtSlotCount < 0) ScanDtSlots((uint8_t*)a, (uint8_t*)b, dtBits);
        dtSlotMask = ApplyDtSlots((uint8_t*)a, (uint8_t*)b, dtBits, Cfg().vrPass2Dt);
    }
    // Pass-2 steps override. The dt census (2026-08-21) showed the sim consumes the timing
    // struct's STEPS field [s+0x18] fresh on EVERY pass (exe+0xb19630 copies it into the
    // task dispatch, x2 per pumped frame), while the float dt [s+0x1c] only becomes limiter
    // ticks. ELDEN RING advances in fixed 1/60 quanta and steps is how many quanta this
    // frame represents - so steps=0 for pass 2 = a full engine frame that advances NOTHING:
    // true same-sim-time stereo. Restored right after the pass.
    uint32_t stepsSaved = 0; bool stepsSwapped = false;
    uint32_t dstSaved[4] = {}; uint32_t dstMask = 0;
    uint64_t bakedSaved[32] = {}; uint32_t bakedMask = 0;
    // What is deliberately NOT here any more: the FD4TaskData arg-block offset scan. It
    // range-matched "looks like a dt" floats in the queue_tasks arg blocks at whatever
    // instant the first pass 2 ran, then blind-wrote 1e-6 into every match around each
    // pass. Every healthy session ever logged found 0 matches (the freeze below works
    // without it); the one session where it found matches (11 of them, 2026-08-22 17:58)
    // corrupted live task state mid-pass: the engine never completed a second pass again
    // (permanent cinema mode), PadStep died with it (XInputGetState calls -> 0, controls
    // dead), and the wedged frame state crashed the game 90 s later in a foot-effect task.
    // A heuristic whose only observed contributions are nothing or catastrophe earns
    // deletion, not a tweak.
    static float* s_dtGlobal = nullptr;
    static bool s_dtGlobalTried = false;
    uint32_t dtGSaved = 0; bool dtGSwapped = false;
    if (fullFrame && Cfg().vrPass2Steps >= 0) {
        const float tiny = 1e-6f; uint32_t tb; memcpy(&tb, &tiny, 4);
        // THE frame-dt global (fd4-builder disasm, 2026-08-21, exe+0x26d5500): every task
        // invocation builds its FD4TaskData ON THE STACK and copies dt from ONE static
        // float (2.6.2.0: exe+0x3C62508) - `lea rax,[vftable]; mov [rsp+28],rax; movss
        // xmm0,[rip+dtGlobal]; movss [rsp+30],xmm0; call task->execute`. It holds the
        // MEASURED wall dt (why every bit-exact 1/60 scan missed) and is read fresh for
        // every task of BOTH passes. Swapping IT around pass 2 is the true freeze.
        if (!s_dtGlobalTried) {
            s_dtGlobalTried = true;
            // Anchored on the builder's own prologue tail (movsxd rdi,r9d + the vftable
            // lea) so it cannot hit the other fd4-construction that uses an .rdata 1/30
            // CONSTANT - the first attempt matched that one, poisoning a shared constant.
            // Layout: movsxd(3) + mov rbx,[rip](7) + lea rax,[rip](7) + mov [rsp+28],rax(5)
            // = 22 bytes to the movss; its disp32 sits at +26 within the 8-byte instruction.
            if (uint8_t* hit = AobScan("49 63 F9 48 8B 1D ?? ?? ?? ?? 48 8D 05 ?? ?? ?? ?? "
                                       "48 89 44 24 28 F3 0F 10 05 ?? ?? ?? ?? F3 0F 11 44 24 30"))
                s_dtGlobal = (float*)ResolveRipRelative(hit + 22, 4, 8);
            if (s_dtGlobal) {
                // Must be a mutable .data cell holding a plausible measured dt - never a
                // read-only constant.
                MEMORY_BASIC_INFORMATION mbi{};
                const bool writable =
                    VirtualQuery(s_dtGlobal, &mbi, sizeof(mbi)) &&
                    (mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE));
                uint32_t cur = 0;
                TryReadU32((uint8_t*)s_dtGlobal, &cur);
                if (!writable || !PlausibleDtF(cur)) {
                    LOGW("[stereo] frame-dt global %p rejected (writable=%d, value bits 0x%08X)",
                         (void*)s_dtGlobal, (int)writable, cur);
                    s_dtGlobal = nullptr;
                } else {
                    float v; memcpy(&v, &cur, 4);
                    LOGI("[stereo] frame-dt global resolved: %p (current %.6f, mutable .data)",
                         (void*)s_dtGlobal, v);
                }
            } else {
                LOGW("[stereo] frame-dt global NOT found (fd4-builder AOB miss)");
            }
        }
        if (s_dtGlobal) {
            uint32_t cur;
            if (TryReadU32((uint8_t*)s_dtGlobal, &cur) && PlausibleDtF(cur) &&
                TryWriteU32((uint8_t*)s_dtGlobal, tb)) {
                dtGSaved = cur;
                dtGSwapped = true;
            }
        }
    }
    if (fullFrame && Cfg().vrPass2Steps >= 0) {
        if (uint8_t* s = g_dtStruct.load(std::memory_order_acquire)) {
            if (TryReadU32(s + 0x18, &stepsSaved) &&
                TryWriteU32(s + 0x18, (uint32_t)Cfg().vrPass2Steps)) stepsSwapped = true;
        }
        // The staged copies were made BEFORE pass 1 (source-field swaps miss them - measured:
        // igt unchanged), so patch the DESTINATIONS the dispatch consumes.
        const int nd = g_stepsDstN.load(std::memory_order_acquire);
        for (int i = 0; i < nd; ++i) {
            uint8_t* p = g_stepsDst[i] + 0xdc;
            if (TryReadU32(p, &dstSaved[i]) && TryWriteU32(p, (uint32_t)Cfg().vrPass2Steps))
                dstMask |= 1u << i;
        }
        // The TERMINAL store (census chain 2026-08-21): the once-per-frame prep reads the
        // staged steps and calls a setter that is just `mov [obj+0x1a8], edx; ret` on the
        // global step-dispatch object *[exe+0x47EF360] (AOB-resolved). This is the field
        // the frame's task iteration consumes - patch IT between the passes.
        static uint8_t** s_stepObj = nullptr;
        static bool s_stepObjTried = false;
        if (!s_stepObjTried) {
            s_stepObjTried = true;
            if (uint8_t* hit = AobScan("48 8B 1D ?? ?? ?? ?? 8B 97 DC 00 00 00 48 8B CB"))
                s_stepObj = (uint8_t**)ResolveRipRelative(hit, 3, 7);
            if (!s_stepObj) LOGW("[stereo] pass-2 steps: step-dispatch global not found (AOB miss)");
        }
        if (s_stepObj) {
            uint8_t* obj = nullptr;
            if (SafeRead(s_stepObj, &obj, sizeof(obj)) && obj) {
                uint8_t* p = obj + 0x1a8;
                if (TryReadU32(p, &dstSaved[3]) && TryWriteU32(p, (uint32_t)Cfg().vrPass2Steps)) {
                    dstMask |= 1u << 3;
                    g_stepsDst[3] = obj;   // reuse the restore path
                }
                if (g_bakedN < 0) {
                    float ref = 1.f / 60.f;
                    if (float* imm = PatchesFrameDtImm()) { if (*imm > 1e-5f && *imm < 0.12f) ref = *imm; }
                    g_bakedN = ScanBaked(obj, 0x400, ref, g_baked, 16);
                    uint8_t *arr = nullptr, *beg = nullptr, *end = nullptr;
                    if (SafeRead(obj + 8, &arr, 8) && arr &&
                        SafeRead(arr + 8, &beg, 8) && SafeRead(arr + 0x10, &end, 8) &&
                        beg && end > beg && (size_t)(end - beg) <= 0x170u * 64u)
                        g_bakedN += ScanBaked(beg, (size_t)(end - beg), ref, g_baked + g_bakedN,
                                              32 - g_bakedN);
                    // Density guard (2026-08-25 crash): the frame's dt is baked ONE per task
                    // entry (stride 0x170), so real candidates are far apart. When the obj+8
                    // pointer chain lands inside a live object instead of the entry array, the
                    // scan matches a tight run of dt-looking floats packed 4-16 bytes apart -
                    // and zeroing those corrupts whatever they belong to. One session read 12
                    // candidates inside GXBS::GXRenderThread and the zero write access-violated
                    // the render thread. Drop any candidate with another within 0x40 bytes: a
                    // lone legit dt survives, a per-entry list (>=0x170 apart) survives, a dense
                    // cluster is discarded whole.
                    { int keep = 0, dropped = 0;
                      for (int i = 0; i < g_bakedN; ++i) {
                          bool dense = false;
                          for (int j = 0; j < g_bakedN; ++j) {
                              if (j == i) continue;
                              intptr_t d = g_baked[i].p - g_baked[j].p; if (d < 0) d = -d;
                              if (d < 0x40) { dense = true; break; }
                          }
                          if (dense) { ++dropped; continue; }
                          g_baked[keep++] = g_baked[i];
                      }
                      if (dropped) LOGW("[stereo] pass-2 baked-dt scan: dropped %d densely-clustered candidate(s) (a live object, not the baked dt list) - not zeroing them", dropped);
                      g_bakedN = keep;
                    }
                    char offs[192]; int op = 0; offs[0] = 0;
                    for (int i = 0; i < g_bakedN && op < 170; ++i)
                        op += snprintf(offs + op, sizeof(offs) - op, " %p%s",
                                       (void*)g_baked[i].p, g_baked[i].dbl ? "d" : "f");
                    LOGI("[stereo] pass-2 baked-dt scan: %d candidate(s):%s", g_bakedN, offs);
                }
            }
        }
        // Baked-entry dt zeroing: the values the pass actually consumes.
        for (int i = 0; i < g_bakedN && i < 32; ++i) {
            uint8_t* p = g_baked[i].p;
            if (g_baked[i].dbl) {
                uint64_t cur;
                if (!TryReadU64(p, &cur)) continue;
                double d; memcpy(&d, &cur, 8);
                if (d < 0.0005 || d > 0.12) continue;   // stale/rebuilt entry: skip
                double tiny = 1e-6;
                uint64_t nb; memcpy(&nb, &tiny, 8);
                if (TryWriteU64(p, nb)) { bakedSaved[i] = cur; bakedMask |= 1u << i; }
            } else {
                uint32_t cur;
                if (!TryReadU32(p, &cur) || !PlausibleDtF(cur)) continue;
                float tiny = 1e-6f;
                uint32_t nb; memcpy(&nb, &tiny, 4);
                if (TryWriteU32(p, nb)) { bakedSaved[i] = cur; bakedMask |= 1u << i; }
            }
        }
        static bool s_logged = false;
        if (!s_logged && (stepsSwapped || dstMask || bakedMask)) {
            s_logged = true;
            LOGI("[stereo] pass-2 steps override ACTIVE: source %s, %d staged dst, dispatch "
                 "[obj+0x1A8] %s, baked dt slots 0x%X -> %d per pass",
                 stepsSwapped ? "swapped" : "miss", nd,
                 (dstMask & 8) ? "swapped" : "miss", bakedMask, Cfg().vrPass2Steps);
        }
    }
    // Gate, run, restore. The restore is the delicate half: g_origQueueTasks below runs a
    // COMPLETE engine frame, and the engine's own flow machine changes task-group active
    // states inside it - which is exactly what a warp does the instant it is requested.
    // The original code snapshotted all 169 bytes and wrote ALL of them back afterwards,
    // so any state change the engine made during pass 2 was silently reverted to its
    // pre-pass value. That is the fast-travel/death crash (2026-08-22): the transfer
    // flipped its task lines, we put them back, and the next PASS 1 frame ran the warp
    // with a task configuration the engine had already moved on from -> null deref at
    // exe+0xc7f88b, reported in pass 1, which is why it never reproduced with only the
    // second pass forced on.
    // It is also why FlowTransitionCheck never once fired: it samples at pass-1 entry,
    // by which point this restore had already erased the very change it was looking for.
    //
    // So: only ever write back groups WE gated, and only where the byte is still the 0 we
    // wrote. Anything the engine touched during the pass keeps the engine's value.
    uint8_t saved[256];
    bool    gated[256] = {};
    const int n = g_table.count < 256 ? g_table.count : 256;
    for (int i = 0; i < n; ++i) {
        saved[i] = ReadActiveRaw(i);
        if (!keepSet[i]) { gated[i] = true; WriteActiveRaw(i, 0); }
    }
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    // The player's input buffer (CSChrActionRequestModule) is sim state that pass 2 re-evaluates
    // at steps 0 and can drop; restored after the pass unless pass 2 started an animation.
    uint8_t* const arPlayer = fullFrame ? GameERMainPlayer() : nullptr;
    if (arPlayer) CombatPass2Begin(arPlayer);
    Pass2InteractBeforePass2();
    g_curPass.store(2, std::memory_order_release);
    g_origQueueTasks(self, a, b);
    Pass2InteractAfterPass2();
    // Pass 2 is complete: finish and push the second eye's report with the camera its
    // render really used (stashed by GameERApplySecondEyeCamera before the pass). Done
    // while the pass tag still reads 2 and before the keep-set restore below.
    GameERFinalizePassReport();
    g_curPass.store(1, std::memory_order_release);
    if (arPlayer) CombatPass2End(arPlayer);
    QueryPerformanceCounter(&t1);
    int engineChanged = 0;
    for (int i = 0; i < n; ++i) {
        const uint8_t now = ReadActiveRaw(i);
        if (gated[i]) {
            // Still the 0 we wrote -> ours to undo. Anything else means the engine set it
            // DURING the pass; its value is the current truth and must survive.
            if (now == 0) WriteActiveRaw(i, saved[i]);
            else ++engineChanged;
        } else if (now != saved[i]) {
            ++engineChanged;   // never touched by us; nothing to do but count it
        }
    }
    // An engine-made change during pass 2 IS the world-transition signal this file has been
    // missing - the earliest one available, and it fires on the frame the warp is requested.
    if (engineChanged) {
        g_flowChangeTick = GetTickCount64();
        static long s_logged = 0;
        if (s_logged < 24 && ++s_logged)
            LOGI("[stereo] engine changed %d task-group state(s) during pass 2 - world "
                 "transition; their values kept, pass 2 suspended briefly",
                 engineChanged);
    }
    if (stepsSwapped) {
        if (uint8_t* s = g_dtStruct.load(std::memory_order_acquire))
            TryWriteU32(s + 0x18, stepsSaved);
    }
    for (int i = 0; i < 3; ++i)
        if (dstMask & (1u << i)) TryWriteU32(g_stepsDst[i] + 0xdc, dstSaved[i]);
    if (dstMask & 8) TryWriteU32(g_stepsDst[3] + 0x1a8, dstSaved[3]);   // dispatch object
    for (int i = 0; i < g_bakedN && i < 32; ++i) {
        if (!(bakedMask & (1u << i))) continue;
        if (g_baked[i].dbl) TryWriteU64(g_baked[i].p, bakedSaved[i]);
        else                TryWriteU32(g_baked[i].p, (uint32_t)bakedSaved[i]);
    }
    if (dtGSwapped) TryWriteU32((uint8_t*)s_dtGlobal, dtGSaved);
    if (dtSlotMask) RestoreDtSlots((uint8_t*)a, (uint8_t*)b, dtSlotMask, dtBits);
    if (dtSwapped) PatchesWriteFrameDt(dtSaved);
    // Symmetric protection: pass 2's image is read by the GPU while the NEXT frame's pass 1
    // uploads - drain again so its data survives until its draws complete.
    if (gpuSync >= 2) PumpGpuDrain("post-pass2");
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    g_pumpLastMs.store((float)((double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart),
                       std::memory_order_relaxed);
    NotePassTime(1, t1.QuadPart - t0.QuadPart);
    InterlockedIncrement(&g_pumpPasses);
    // Watchdog: pass 2 was CALLED, but did the engine actually run it? In the 2026-08-22
    // incident the first (corrupted) pass wedged the frame runner mid-frame and every later
    // queue_tasks call returned as a no-op - pass 2 "ran" 72x a second, cost 0.0 ms, and
    // never reached Flip, so no pass-2 present ever arrived and the session sat in cinema
    // mode with nothing in the log saying why. This says why.
    {
        static ULONGLONG s_firstAttempt = 0, s_lastAttempt = 0, s_lastWarn = 0;
        const ULONGLONG now = GetTickCount64();
        // A >2 s gap in attempts (menu, guard window) starts a fresh episode, so the first
        // attempt after a long menu is not measured against a present from before it.
        if (!s_firstAttempt || now - s_lastAttempt > 2000) s_firstAttempt = now;
        s_lastAttempt = now;
        const ULONGLONG seen = g_p2PresentTick.load(std::memory_order_relaxed);
        const ULONGLONG ref = seen > s_firstAttempt ? seen : s_firstAttempt;
        if (now - ref > 5000 && now - s_lastWarn > 60000) {
            s_lastWarn = now;
            LOGW("[stereo] pass 2 is being called but the engine never runs it to Flip (no "
                 "pass-2 present for %llu s) - frame runner wedged mid-frame? The headset "
                 "stays on the flat quad; a game restart clears it.",
                 (unsigned long long)((now - ref) / 1000));
        }
    }
    if (g_winTrace.load(std::memory_order_relaxed)) {
        const int i = g_p2N.fetch_add(1, std::memory_order_relaxed);
        if (i < 256) g_p2Win[i] = {t0.QuadPart, t1.QuadPart};
    }
}

// Gameflow-transition guard. The engine's own flow machine flips task-group active bytes
// when entering/leaving map transfers, cutscenes, title flow, ... - the exact windows where
// a second pass replays the frame over half-torn-down world state. Fast travel crashed
// twice through DIFFERENT mechanisms (a DL_PANIC 2026-08-21 10:14, then a raw INT_OVERFLOW
// in a protected code region 10:34 that no assert-eater can catch), both in the fade-out
// BEFORE any camera jump - so the guard keys on the engine's own earliest signal instead:
// ANY active-byte change (that the engine made, observed at pass-1 entry) suspends pass 2
// for a short tail. Normal play never flips lines, so the cost is a moment of mono exactly
// when the screen is fading anyway. (g_flowChangeTick is declared above DoSecondPass.)
void FlowTransitionCheck() {
    if (!g_table.valid) return;
    static uint8_t s_prev[256];
    static bool s_have = false;
    const int n = g_table.count < 256 ? g_table.count : 256;
    bool changed = false;
    for (int i = 0; i < n; ++i) {
        const uint8_t v = ReadActiveRaw(i);
        if (s_have && v != s_prev[i]) {
            changed = true;
            static long s_logged = 0;
            if (s_logged < 24 && ++s_logged) {
                const wchar_t* nm = (const wchar_t*)(g_table.entry0 + (size_t)i * g_table.stride + kNameOff);
                LOGI("[stereo] flow transition: group %d (%S) active %d -> %d; pass 2 suspended",
                     i, nm, (int)s_prev[i], (int)v);
            }
        }
        s_prev[i] = v;
    }
    s_have = true;
    if (changed) g_flowChangeTick = GetTickCount64();
}

uint64_t __fastcall Hook_QueueTasks(void* self, void* a, void* b) {
    const long i = InterlockedIncrement(&g_qtCalls);
    LARGE_INTEGER t0; QueryPerformanceCounter(&t0);
    g_qtLastEnterQpc = t0.QuadPart;
    g_qtArgA = a; g_qtArgB = b;
    FlowTransitionCheck();
    Pass2InteractFrameBegin(i);
    if (i <= 3) {
        void* frames[12] = {};
        const USHORT got = RtlCaptureStackBackTrace(1, 12, frames, nullptr);
        auto* base = (uint8_t*)GetModuleHandleW(nullptr);
        char line[400]; int p = 0;
        for (USHORT k = 0; k < got && p < (int)sizeof(line) - 24; ++k) {
            HMODULE m = nullptr;
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)frames[k], &m);
            p += (m == (HMODULE)base)
                     ? snprintf(line + p, sizeof(line) - p, " exe+0x%llx",
                                (unsigned long long)((uint8_t*)frames[k] - base))
                     : snprintf(line + p, sizeof(line) - p, " [%p]", frames[k]);
        }
        LOGI("[m2] queue_tasks call %ld: this=%p a=%p b=%p thread %lu, callers:%s",
             i, self, a, b, GetCurrentThreadId(), line);
    }
    if (!g_qtThread) g_qtThread = GetCurrentThreadId();
    const uint64_t r = g_origQueueTasks(self, a, b);
    { LARGE_INTEGER tp; QueryPerformanceCounter(&tp); NotePassTime(0, tp.QuadPart - t0.QuadPart); }
    // Pass 1 is complete (every camera writer ran, Flip presented): finish and push its eye
    // report with the camera the renderer really used. Must precede DoSecondPass, which
    // stashes pass 2's own pending report.
    GameERFinalizePassReport();
    if (g_winTrace.load(std::memory_order_relaxed)) {
        LARGE_INTEGER tm; QueryPerformanceCounter(&tm);
        const int k = g_p1N.fetch_add(1, std::memory_order_relaxed);
        if (k < 256) g_p1Win[k] = {t0.QuadPart, tm.QuadPart};
    }
    if (g_pumpMode != 0) DoSecondPass(self, a, b);
    LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
    g_qtLastLeaveQpc = t1.QuadPart;
    // Periodic timing/synchronicity log: experiment-only. In production (full stereo) the
    // DrawEnd comparison is meaningless (no probe tasks) and the line is just noise.
    if (Cfg().taskQueueProbe && (i <= 3 || (i % 600) == 0)) {
        static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
        const double inMs = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)f.QuadPart;
        // If DrawEnd's task fired INSIDE this call, the frame executes synchronously under
        // queue_tasks and a second call from the same spot would be a full second pass.
        const __int64 de = g_drawEndQpc;
        LOGI("[m2] queue_tasks call %ld took %.2f ms; DrawEnd task fired %s it returned",
             i, inMs, (de > t0.QuadPart && de < t1.QuadPart) ? "INSIDE - the frame is synchronous under queue_tasks"
                                                             : "OUTSIDE");
    }
    return r;
}

// ---- experiment: dump every group's registered task list with RTTI names ---------------
// The map at queue_tasks arg3+0x18 is an MSVC std::map<cstgi, task_entry_group*> (node walk
// visible in queue_tasks' own code: key at node+0x20, value at +0x28, nil byte at +0x19).
// task_entry_group (fromsoftware-rs, asserts match the 0x269ef80 disassembly): entries vector
// at +0x08 {alloc,begin,end,cap} of task_entry { vptr, FD4TaskBase* task, void*, cstgi }.
// Dumping the class name of every registered task per group is the map of the entire engine
// frame - it names the group that hosts the real render tasks and the sim tasks a draw-only
// second pass must gate.
struct MapNodeView {
    void* left; void* parent; void* right;
    uint8_t color, isNil; uint8_t _pad[6];
    uint32_t key; uint32_t _pad2;
    void* value;
};

// SEH-isolated: walks live engine structures that can churn under us. POD in/out only.
int DumpOneGroup(void* mapHolder, uint32_t id, void** outTasks, int maxTasks) {
    __try {
        auto* head = *(MapNodeView**)((uint8_t*)mapHolder + 0x18);
        if (!head) return -2;
        MapNodeView* node = (MapNodeView*)head->parent;   // root
        MapNodeView* best = nullptr;
        while (node && !node->isNil) {
            if (node->key >= id) { best = node; node = (MapNodeView*)node->left; }
            else node = (MapNodeView*)node->right;
        }
        if (!best || best->key != id) return -3;
        uint8_t* grp = (uint8_t*)best->value;
        if (!grp) return -4;
        uint8_t* begin = *(uint8_t**)(grp + 0x10);
        uint8_t* end   = *(uint8_t**)(grp + 0x18);
        if (!begin || end < begin || (size_t)(end - begin) > 0x4000) return -5;
        int n = 0;
        for (uint8_t* e = begin; e + 0x20 <= end && n < maxTasks; e += 0x20)
            outTasks[n++] = *(void**)(e + 8);   // task_entry::task
        return n;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return -1; }
}

std::string NameOfObject(void* obj) {
    if (!obj || !Readable(obj, 8)) return {};
    void** vt = *(void***)obj;
    return LooksLikeVTable(vt, 1) ? RttiNameOfVTable(vt) : std::string();
}

// Most groups hold CSEzTaskProxy objects, whose class says nothing about the work. The proxy
// points at its CSEzTask at +0x10 (fromsoftware-rs task.rs), and the update/void task templates
// keep the subsystem they step right after the task base: +0x18 behind a plain CSEzTask, +0x20
// behind the rabbit bases. Naming that subject maps each group to the systems pass 2 re-runs.
std::string DescribeTask(void* task) {
    std::string cls = NameOfObject(task);
    if (cls.empty()) return "?";
    if (cls.find("Proxy") == std::string::npos || !Readable((uint8_t*)task + 0x10, 8)) return cls;
    void* inner = *(void**)((uint8_t*)task + 0x10);
    const std::string innerCls = NameOfObject(inner);
    if (innerCls.empty()) return cls + "->?";
    for (size_t off : {0x18, 0x20}) {
        if (!Readable((uint8_t*)inner + off, 8)) continue;
        const std::string subject = NameOfObject(*(void**)((uint8_t*)inner + off));
        if (!subject.empty()) return "{" + subject + "}";
    }
    return "<" + innerCls + ">";
}

void DumpTaskListsNow(const char* when, bool showEmpty) {
    void* holder = g_qtArgB;
    if (!holder) { LOGW("[m2] TaskDumpLists: queue_tasks never captured its args"); return; }
    BuildRegionMap();   // the snapshot from probe time is stale by now
    LOGI("[m2] ---- task lists per group, %s (map holder %p) ----", when, holder);
    int totalTasks = 0, groupsWithTasks = 0;
    for (int i = 0; i < g_table.count; ++i) {
        uint8_t* e = g_table.entry0 + (size_t)i * g_table.stride;
        const uint32_t id = *(const uint32_t*)e;
        char nameA[80]{};
        WideCharToMultiByte(CP_UTF8, 0, (const wchar_t*)(e + kNameOff), -1, nameA, sizeof(nameA) - 1, nullptr, nullptr);
        void* tasks[64] = {};
        const int n = DumpOneGroup(holder, id, tasks, 64);
        if (n <= 0) {
            if (n < -1 || showEmpty) LOGI("[m2] group %3d %-40s %s", i, nameA, n == 0 ? "[0] empty" : n == -1 ? "<faulted>" : "<lookup failed>");
            continue;
        }
        ++groupsWithTasks; totalTasks += n;
        std::string line;
        for (int k = 0; k < n; ++k) {
            line += DescribeTask(tasks[k]) + " ";
            if (line.size() > 140) { LOGI("[m2] group %3d %-28s [%d] %s", i, nameA, n, line.c_str()); line.clear(); }
        }
        LOGI("[m2] group %3d %-28s [%d] %s", i, nameA, n, line.c_str());
    }
    LOGI("[m2] ---- %d groups hold %d registered tasks ----", groupsWithTasks, totalTasks);
}

void DumpTaskLists() {
    if (!Cfg().taskDumpLists) return;
    // Wait for the hook to capture the frame-driver args.
    for (int w = 0; w < 100 && !g_qtArgB; ++w) Sleep(100);
    // And for the world: at the main menu the gameplay systems have not registered their
    // tasks yet, and a menu-time dump under-reports exactly the groups we care about.
    for (int w = 0; w < 120 && !GameERPlayerActive(); ++w) Sleep(1000);
    LOGI("[m2] TaskDumpLists: player %s", GameERPlayerActive() ? "active - dumping the in-game task lists"
                                                               : "NEVER became active - dumping menu-time lists");
    DumpTaskListsNow("at player-active", false);
    // Some managers register only once the world has settled, and an empty group looks the same
    // as a missing one above - so look again later and say which groups are empty.
    Sleep(30000);
    DumpTaskListsNow("30 s in the world", true);
}

float PresentsPerSecond(float secs);   // defined with the toggle experiment below

void DoublePumpTest() {
    if (!Cfg().taskDoublePumpTest) return;
    if (!g_origQueueTasks) { LOGW("[m2] PumpTest needs TaskQueueProbe=true (no hook)"); return; }
    Sleep((DWORD)(Cfg().taskToggleDelay * 1000.f));   // same "wait until in-game" budget
    ErvrFrameTask* fb = g_probeTask[0];
    ErvrFrameTask* sd = g_probeTask[1];
    LOGI("[m2] PUMP TEST: 3 cycles of 4 s baseline / 4 s double-pumped. Canaries: FrameBegin "
         "must HOLD ~240/4s (sim gated in pass 2), GameSceneDraw must DOUBLE (~480/4s).");
    for (int cyc = 1; cyc <= 3; ++cyc) {
        long fb0 = fb ? fb->calls : 0, sd0 = sd ? sd->calls : 0;
        const float base = PresentsPerSecond(4.f);
        long fb1 = fb ? fb->calls : 0, sd1 = sd ? sd->calls : 0;
        g_pumpMode = 1;
        LOGI("[m2] PUMP cycle %d ON", cyc);
        long fb2 = fb ? fb->calls : 0, sd2 = sd ? sd->calls : 0;
        const float pump = PresentsPerSecond(4.f);
        long fb3 = fb ? fb->calls : 0, sd3 = sd ? sd->calls : 0;
        g_pumpMode = 0;
        LOGI("[m2] PUMP cycle %d: baseline %.1f pps (FrameBegin %ld, GameSceneDraw %ld / 4s) -> "
             "pumped %.1f pps (FrameBegin %ld, GameSceneDraw %ld), pass2 last %.2f ms, total %ld passes",
             cyc, base, fb1 - fb0, sd1 - sd0, pump, fb3 - fb2, sd3 - sd2,
             g_pumpLastMs.load(), (long)g_pumpPasses);
        Sleep(1000);
    }
    // ---- present-pairing probe ----------------------------------------------------------
    // With the pump on, trace every Present for ~1.2 s and classify each against the pass-1
    // and pass-2 execution windows. If presents land INSIDE the pass windows (and on the
    // main-loop thread), L/R pairing is deterministic: whatever presents during pass 2 IS
    // the second (right-eye) image. If they land outside / on another thread, the renderer
    // is decoupled and pairing needs the FIFO with slack instead.
    g_p1N.store(0); g_p2N.store(0);
    g_pumpMode = 1;
    g_winTrace.store(true);
    ReShadeAddonPresentTraceStart();
    Sleep(1200);
    PresentSample smp[512];
    const int ns = ReShadeAddonPresentTraceStop(smp, 512);
    g_winTrace.store(false);
    g_pumpMode = 0;
    int n1 = g_p1N.load(); if (n1 > 256) n1 = 256;
    int n2 = g_p2N.load(); if (n2 > 256) n2 = 256;
    int inP1 = 0, inP2 = 0, outside = 0, onMainThread = 0;
    DWORD otherTid = 0;
    for (int s = 0; s < ns; ++s) {
        if (smp[s].tid == g_qtThread) ++onMainThread; else otherTid = smp[s].tid;
        bool hit = false;
        for (int k = 0; k < n1 && !hit; ++k)
            if (smp[s].qpc >= g_p1Win[k].t0 && smp[s].qpc <= g_p1Win[k].t1) { ++inP1; hit = true; }
        for (int k = 0; k < n2 && !hit; ++k)
            if (smp[s].qpc >= g_p2Win[k].t0 && smp[s].qpc <= g_p2Win[k].t1) { ++inP2; hit = true; }
        if (!hit) ++outside;
    }
    LOGI("[m2] PAIRING: %d presents over 1.2 s pumped (%d pass-1 windows, %d pass-2): "
         "%d inside pass-1, %d inside pass-2, %d outside; %d/%d on the main-loop thread "
         "(%lu), other thread %lu -> %s",
         ns, n1, n2, inP1, inP2, outside, onMainThread, ns, g_qtThread, otherTid,
         (outside <= ns / 10)
             ? "Present is SYNCHRONOUS with the pass - tag the eye by pass window"
             : "Present is DECOUPLED from the frame - pair via FIFO with drop tolerance");

    LOGI("[m2] PUMP TEST done (game still alive = pass 2 is survivable)");
}

// ---- experiment: content probe ----------------------------------------------------------
// Answers, WITHOUT a headset, the two questions the in-headset bugs raised:
//   A. Does a Present carry ITS OWN pass's render, or the PREVIOUS pass's (pipelined)?
//      Method: toggle an 80-degree pitch on pass 2's camera once a second and see which
//      pass-tag's sampled luminance follows the toggle.
//   B. Does the draw-only pass render DARKER than the real pass (the user's dark eye)?
//      Method: same camera both passes, sweep Pass2Keep variants, compare per-tag luminance.
struct CSample { uint8_t pass, pitchOn; float luma; };
constexpr int kMaxCSamples = 4096;
CSample g_cs[kMaxCSamples];
std::atomic<int>  g_csN{0};
std::atomic<bool> g_csOn{false};

struct CellStats { double sum = 0; int n = 0; double Mean() const { return n ? sum / n : 0; } };

// Aggregates the sample buffer into (pass, pitch) cells.
void CsCells(CellStats out[2][2]) {
    const int n = g_csN.load() < kMaxCSamples ? g_csN.load() : kMaxCSamples;
    for (int i = 0; i < n; ++i) {
        const CSample& s = g_cs[i];
        CellStats& c = out[s.pass == 2 ? 1 : 0][s.pitchOn ? 1 : 0];
        c.sum += s.luma; ++c.n;
    }
}

void SetPass2Override(const char* spec) {
    std::lock_guard<std::mutex> g(g_pass2OverrideMtx);
    g_pass2Override = spec;
}

// Screenshot-driven variant: NO GPU work of our own. Three attempts at reading the back
// buffer from inside the present event all crashed D3D12Core at +0x11f85 (blocking wait,
// non-blocking ring, and both barrier states) - submitting our own command list on the
// game's queue during ReShade's present event is simply not viable here.
//
// The monitor shows the LAST image presented, which answers both questions for free:
//   A. Routing: pitch pass 2's camera at the ground. If presents carry their OWN pass's
//      render, the screen shows the tilted view; if they lag one pass, it stays normal.
//   B. Brightness: with the pump on, the screen shows one pass's render every frame - if
//      the draw-only pass is the dark one, the SCREENSHOT is dark. Sweeping Pass2Keep then
//      shows which task group restores it.
// Each phase is timestamped in the log; tools/run_game.ps1 -Shots correlates by filename.
void ContentProbeShots() {
    if (!Cfg().taskContentProbe) return;
    if (!g_origQueueTasks) { LOGW("[cp] needs TaskQueueProbe=true (no frame-driver hook)"); return; }
    for (int w = 0; w < 180 && !GameERPlayerActive(); ++w) Sleep(1000);
    if (!GameERPlayerActive()) { LOGW("[cp] player never became active; aborting"); return; }
    LOGI("[cp] player active; settling 20 s");
    Sleep(20000);

    auto phase = [](const char* what, int secs) {
        LOGI("[cp] PHASE %s (%d s)", what, secs);
        Sleep(secs * 1000);
    };

    phase("BASELINE: pump off, no pitch", 10);

    g_pumpMode = 1;
    phase("A1: pump ON, pass-2 pitch OFF", 10);
    g_probePitchOn.store(true);
    phase("A2: pump ON, pass-2 pitch 80deg AT THE GROUND", 12);
    g_probePitchOn.store(false);

    const char* variants[] = {
        "144-158,164",                  // baseline (what the user saw)
        "107,144-158,164",              // + DrawParamUpdate (lighting/tonemap param banks)
        "98,107,144-158,164",           // + RenderingSystemUpdate
        "49,98,107,144-158,164",        // + Geom_UpdateDraw
        "49,98,107,142,144-158,164",    // + ScaleformStep (UI list)
    };
    for (const char* v : variants) {
        SetPass2Override(v);
        char buf[128];
        snprintf(buf, sizeof(buf), "B keep={%s}", v);
        phase(buf, 10);
    }
    SetPass2Override("");
    g_pumpMode = 0;
    phase("FINAL: pump off (should look like BASELINE)", 8);
    LOGI("[cp] ---- content probe done ----");
}

void ContentProbe() {
    if (!Cfg().taskContentProbe) return;
    if (!g_origQueueTasks) { LOGW("[cp] needs TaskQueueProbe=true (no frame-driver hook)"); return; }
    for (int w = 0; w < 180 && !GameERPlayerActive(); ++w) Sleep(1000);
    if (!GameERPlayerActive()) { LOGW("[cp] player never became active; aborting"); return; }
    // Settle WELL clear of the spawn fade: arming the pump 3 s after spawn crashed the game
    // at exe+0x1e8b0bc (a draw-path fn reading a null per-frame input - the fade machinery
    // lives in groups pass 2 gates). 55+ s post-spawn pumping was always stable.
    LOGI("[cp] player active; settling 20 s before pumping");
    Sleep(20000);

    // Bisection stages: two identical crashes at exe+0x1e8b0bc landed ~1.0 s after pump-on
    // = ~0.5 s after sampling began, while pump-alone ran 600+ passes cleanly in earlier
    // sessions. Separate the suspects so the crash timestamp names the guilty one.
    // Stage 0 A/B: which resource state is the back buffer really in during the present
    // event? PRESENT(==COMMON) crashed D3D12Core at +0x11f85 twice, so try RENDER_TARGET
    // first - surviving it names the state AND flags the same latent assumption in the blit.
    LOGI("[cp] stage 0a: readback with StateBefore=RENDER_TARGET (3 s)");
    Dx12SetProbeSrcState(D3D12_RESOURCE_STATE_RENDER_TARGET);
    g_csN.store(0);
    g_csOn.store(true);
    Sleep(3000);
    g_csOn.store(false);
    LOGI("[cp] stage 0a SURVIVED (%d samples) -> the back buffer is in RENDER_TARGET at "
         "present time; dx12_blit's PRESENT assumption is WRONG too", g_csN.load());

    LOGI("[cp] stage 0b: readback with StateBefore=PRESENT (3 s) - expected to crash");
    Dx12SetProbeSrcState(D3D12_RESOURCE_STATE_PRESENT);
    g_csN.store(0);
    g_csOn.store(true);
    Sleep(3000);
    g_csOn.store(false);
    LOGI("[cp] stage 0b SURVIVED (%d samples) - state is NOT the crash cause", g_csN.load());
    Dx12SetProbeSrcState(D3D12_RESOURCE_STATE_RENDER_TARGET);

    LOGI("[cp] stage 1: pump only, no readback (3 s)");
    g_pumpMode = 1;
    Sleep(3000);
    LOGI("[cp] stage 1 SURVIVED (%ld passes)", (long)g_pumpPasses);
    g_pumpMode = 0;
    Sleep(1000);

    LOGI("[cp] ---- content probe: PHASE A (present routing: pump + readback) ----");
    g_pumpMode = 1;
    Sleep(500);
    g_csN.store(0);
    g_csOn.store(true);
    for (int w = 0; w < 8; ++w) {
        g_probePitchOn.store((w & 1) != 0);
        Sleep(1000);
    }
    g_csOn.store(false);
    g_probePitchOn.store(false);
    {
        CellStats c[2][2]{};
        CsCells(c);
        const double d1 = std::fabs(c[0][1].Mean() - c[0][0].Mean());
        const double d2 = std::fabs(c[1][1].Mean() - c[1][0].Mean());
        LOGI("[cp] A: luma by (pass, pitch): p1/off %.4f (n=%d)  p1/on %.4f (n=%d)  "
             "p2/off %.4f (n=%d)  p2/on %.4f (n=%d)",
             c[0][0].Mean(), c[0][0].n, c[0][1].Mean(), c[0][1].n,
             c[1][0].Mean(), c[1][0].n, c[1][1].Mean(), c[1][1].n);
        LOGI("[cp] A VERDICT: pitch moved p1 presents by %.4f, p2 presents by %.4f -> %s",
             d1, d2,
             (d2 > 2 * d1 && d2 > 0.01) ? "presents carry their OWN pass's render: use PresentLag=0"
             : (d1 > 2 * d2 && d1 > 0.01) ? "presents carry the PREVIOUS pass's render: use PresentLag=1"
             : "INCONCLUSIVE (deltas too similar or too small)");
    }

    LOGI("[cp] ---- content probe: PHASE B (draw-only pass brightness) ----");
    const char* variants[] = {
        "144-158,164",                  // baseline (what the user saw)
        "107,144-158,164",              // + DrawParamUpdate (lighting/tonemap param banks)
        "98,107,144-158,164",           // + RenderingSystemUpdate
        "49,98,107,144-158,164",        // + Geom_UpdateDraw
        "49,98,107,142,144-158,164",    // + ScaleformStep (UI list)
    };
    for (const char* v : variants) {
        SetPass2Override(v);
        Sleep(700);   // let a few pairs render under the new set
        g_csN.store(0);
        g_csOn.store(true);
        Sleep(2500);
        g_csOn.store(false);
        CellStats c[2][2]{};
        CsCells(c);
        const double p1 = c[0][0].Mean() + c[0][1].Mean();   // pitch always off in phase B
        const double p2 = c[1][0].Mean() + c[1][1].Mean();
        LOGI("[cp] B keep={%s}: p1-present luma %.4f (n=%d), p2-present luma %.4f (n=%d) -> "
             "p2/p1 = %.0f%%%s",
             v, p1, c[0][0].n + c[0][1].n, p2, c[1][0].n + c[1][1].n,
             p1 > 1e-6 ? p2 / p1 * 100.0 : 0.0,
             (p1 > 0.02 && p2 < p1 * 0.75) ? "  <-- pass-2-tagged presents are DARK here" : "");
    }
    SetPass2Override("");
    g_pumpMode = 0;
    LOGI("[cp] ---- content probe done ----");
}

void QueueTasksProbeInstall(bool force = false) {
    if ((!Cfg().taskQueueProbe && !force) || !g_table.vtable || g_origQueueTasks) return;
    void* target = g_table.vtable[5];
    if (MH_CreateHook(target, (void*)&Hook_QueueTasks, (void**)&g_origQueueTasks) == MH_OK &&
        MH_EnableHook(target) == MH_OK)
        LOGI("[m2] queue_tasks hooked at %p", target);
    else
        LOGW("[m2] queue_tasks hook FAILED");
}

void RegisterExperiment() {
    if (!Cfg().taskRegisterTest) return;
    if (g_table.count != 169) { LOGW("[m2] TaskRegisterTest: table mismatch, not registering"); return; }
    g_probeTask[0] = RegisterFrameTask("FrameBegin", FrameProbeIndexOf("FrameBegin"));
    g_probeTask[1] = RegisterFrameTask("GameSceneDraw", FrameProbeIndexOf("GameSceneDraw"));
    g_probeTask[2] = RegisterFrameTask("DrawEnd", FrameProbeIndexOf("DrawEnd"));
    // Cadence report: are all three called once per frame, at the same rate, with what dt?
    for (int i = 0; i < 3; ++i) {
        Sleep(5000);
        std::string line = "[m2] task cadence @+" + std::to_string((i + 1) * 5) + "s:";
        char buf[160];
        for (ErvrFrameTask* t : g_probeTask) {
            if (!t) continue;
            snprintf(buf, sizeof(buf), " %s=%ld (dt %.5f, seed %d)",
                     t->name, t->calls, t->lastDt, t->lastSeed);
            line += buf;
        }
        LOGI("%s", line.c_str());
    }
}

// ---- experiment: gate a draw task group -------------------------------------------------
// The decisive question for mechanism B' is whether the draw phase is separable from the rest
// of the frame at all. Clearing `active` on one draw group and watching what stops being drawn
// answers it directly, costs one byte, and restores itself.
// Present rate over the next `secs`, which is the quantitative signal: if gating a group
// really removes GPU work, the frame rate moves. A single before/after screenshot cannot tell
// that apart from the sky changing - ELDEN RING's time of day moved from night to overcast
// inside 20 s during the first attempt at this, which is exactly the kind of thing that makes
// an eyeballed A/B lie.
float PresentsPerSecond(float secs) {
    ReShadeAddonGetStats();                       // reset the counter
    Sleep((DWORD)(secs * 1000.f));
    return (float)ReShadeAddonGetStats().presents / secs;
}

void ToggleExperiment() {
    const Config& c = Cfg();
    if (c.taskToggle.empty()) return;
    const int idx = FrameProbeIndexOf(c.taskToggle.c_str());
    if (idx < 0) { LOGW("[m2] TaskToggle: no group named \"%s\"", c.taskToggle.c_str()); return; }

    bool was = true;
    FrameProbeGetActive(idx, was);
    Sleep((DWORD)(c.taskToggleDelay * 1000.f));
    const float phase = c.taskToggleSeconds > 0.5f ? c.taskToggleSeconds : 3.f;
    const int cycles = c.taskToggleCycles > 0 ? c.taskToggleCycles : 4;
    // "byte" writes cstg_control.active directly; "api" calls CS::CSTask::set_task_group_state.
    // The byte alone provably does nothing - the engine reads it when it BUILDS the queue,
    // not when it runs it - so the API is the one that matters.
    const bool useApi = c.taskToggleMode != "byte";
    auto setState = [&](int i, bool on) {
        return useApi ? FrameProbeSetActiveApi(i, on) : FrameProbeSetActive(i, on);
    };
    LOGI("[m2] TOGGLE: A/B on \"%s\" (index %d, was active=%d) via %s: %d cycles of %.1f s off / %.1f s on",
         c.taskToggle.c_str(), idx, (int)was, useApi ? "set_task_group_state()" : "raw byte",
         cycles, phase, phase);

    // A task of OURS registered in the toggled group is the ground truth for whether the
    // gate gates anything: the screen cannot tell "scene not drawn" from "same image again",
    // but our call counter can tell "the group's tasks stopped running".
    ErvrFrameTask* watch = Cfg().taskRegisterTest ? RegisterFrameTask(_strdup((c.taskToggle + ".watch").c_str()), idx) : nullptr;
    ErvrFrameTask* ref = g_probeTask[0];   // FrameBegin, never gated - the "frames happened" baseline
    if (watch) Sleep(1000);                // let the registration take effect before measuring

    float offSum = 0, onSum = 0;
    long  watchOffSum = 0, watchOnSum = 0, refOffSum = 0, refOnSum = 0;
    for (int i = 0; i < cycles; ++i) {
        if (!setState(idx, false)) { LOGW("[m2] TOGGLE: disable failed"); break; }
        // Phase transitions are logged at the instant they happen so an external screenshot
        // series can be correlated against them. The fps number alone is not enough: ELDEN
        // RING is frame-capped, so removing GPU work often shows up as no change at all.
        bool rb = true; FrameProbeGetActive(idx, rb);
        LOGI("[m2] TOGGLE phase %d OFF (byte reads %d)", i + 1, (int)rb);
        long w0 = watch ? watch->calls : 0, r0 = ref ? ref->calls : 0;
        const float off = PresentsPerSecond(phase);
        const long wOff = watch ? watch->calls - w0 : 0, rOff = ref ? ref->calls - r0 : 0;
        setState(idx, true);
        LOGI("[m2] TOGGLE phase %d ON", i + 1);
        w0 = watch ? watch->calls : 0; r0 = ref ? ref->calls : 0;
        const float on = PresentsPerSecond(phase);
        const long wOn = watch ? watch->calls - w0 : 0, rOn = ref ? ref->calls - r0 : 0;
        offSum += off; onSum += on;
        watchOffSum += wOff; watchOnSum += wOn; refOffSum += rOff; refOnSum += rOn;
        LOGI("[m2] TOGGLE cycle %d: OFF %.1f fps (tasks: FrameBegin %ld, %s %ld), "
             "ON %.1f fps (FrameBegin %ld, %s %ld)",
             i + 1, off, rOff, c.taskToggle.c_str(), wOff, on, rOn, c.taskToggle.c_str(), wOn);
    }
    if (watch)
        LOGI("[m2] TOGGLE TASK VERDICT: during OFF our %s task ran %ld times over %ld frames; "
             "during ON %ld over %ld -> %s",
             c.taskToggle.c_str(), watchOffSum, refOffSum, watchOnSum, refOnSum,
             (watchOnSum > 0 && watchOffSum <= cycles)   // boundary races leak ~1 frame per cycle
                 ? "set_task_group_state STOPS the group's tasks (the gate works; whatever "
                   "kept rendering is not driven by this group's task list)"
                 : (watchOnSum > 0 && watchOffSum >= watchOnSum / 2)
                     ? "the group's tasks KEPT RUNNING while gated - set_task_group_state is "
                       "not the runtime gate either"
                     : "inconclusive (low counts)");
    FrameProbeSetActive(idx, was);
    if (cycles > 0)
        LOGI("[m2] TOGGLE RESULT \"%s\": mean OFF %.1f fps vs ON %.1f fps (%+.1f%%) - %s",
             c.taskToggle.c_str(), offSum / cycles, onSum / cycles,
             onSum > 0 ? (offSum / onSum - 1.f) * 100.f : 0.f,
             (offSum / (onSum > 0 ? onSum : 1.f)) > 1.10f
                 ? "GATING WORKS: disabling the group measurably removed work"
                 : "no measurable effect - either the group is cheap or the byte is not the gate");
}

} // namespace

const TaskGroupTable& FrameProbeGetTable() { return g_table; }

int FrameProbeIndexOf(const char* name) {
    if (!g_table.valid) return -1;
    wchar_t w[80]{};
    MultiByteToWideChar(CP_UTF8, 0, name, -1, w, 79);
    for (int i = 0; i < g_table.count; ++i)
        if (NameIs(g_table.entry0 + (size_t)i * g_table.stride, w)) return i;
    return -1;
}

bool FrameProbeGetActive(int index, bool& out) {
    if (!g_table.valid || index < 0 || index >= g_table.count) return false;
    out = g_table.entry0[(size_t)index * g_table.stride + kActiveOff] != 0;
    return true;
}

bool FrameProbeSetActive(int index, bool active) {
    if (!g_table.valid || index < 0 || index >= g_table.count) return false;
    uint8_t v = active ? 1 : 0;
    return SafeWrite(g_table.entry0 + (size_t)index * g_table.stride + kActiveOff, &v, 1);
}

namespace {
using PfnSetTaskGroupState = bool(__fastcall*)(void* self, int32_t id, bool state);

// SEH-wrapped: this calls into the engine from a thread it knows nothing about. POD locals only.
int CallSetState(void* self, void* fn, int32_t id, bool state) {
    __try { return ((PfnSetTaskGroupState)fn)(self, id, state) ? 1 : 0; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return -1; }
}
} // namespace

bool FrameProbeSetActiveApi(int index, bool active) {
    if (!g_table.valid || !g_table.owner || !g_table.vtable) return false;
    if (index < 0 || index >= g_table.count) return false;
    // The id is NOT the index: it is 0x90000000 | index, read straight from the entry.
    const int32_t id = *(const int32_t*)(g_table.entry0 + (size_t)index * g_table.stride);
    // Slot 3: [0]=~dtor, [1]=add_task_group, [2]=unk_set_task_group, [3]=set_task_group_state.
    const int r = CallSetState(g_table.owner, g_table.vtable[3], id, active);
    if (r < 0) { LOGE("[m2] set_task_group_state(0x%08X, %d) FAULTED", (unsigned)id, (int)active); return false; }
    return true;
}

void FrameProbeTick() {
    const bool fullStereo = Cfg().vrStereoMode == "full";
    // Alternate-eye stereo needs the same frame hook - each eye's report is pushed with the
    // camera the renderer really used, at the end of queue_tasks - but never a second pass.
    // Without it no report ever reached the queue and the mode stayed flat (2026-11-03).
    const bool altStereo = Cfg().vrStereoMode == "debugalt";
    if (!Cfg().m2TaskProbe && !fullStereo && !altStereo && !Cfg().m2ForcePass2) return;
    if (g_started.exchange(true)) return;
    // Its OWN thread, deliberately. This can scan a lot; the worker thread it used to run on
    // also drives BackendsTick(), so blocking there meant xrCreateInstance never ran and the
    // headset never came up. Nothing the probe does needs to be in step with anything else.
    std::thread([fullStereo, altStereo] {
        // The table is built during engine init, so early attempts can legitimately miss.
        for (int attempt = 1; attempt <= 6; ++attempt) {
            if (RunOnce(attempt)) {
                InstallIgtTrace();
                // Test harness: unconditional second pass, no headset needed. Checked
                // BEFORE the stereo arm so it wins if both are set.
                if (Cfg().m2ForcePass2) {
                    QueueTasksProbeInstall(true);
                    if (g_origQueueTasks) {
                        g_pumpMode = 1;
                        LOGW("[stereo] ForcePass2: second pass runs UNCONDITIONALLY (test "
                             "harness; no XR, no second-eye camera)");
                    } else {
                        LOGW("[stereo] ForcePass2 requested but the queue_tasks hook failed");
                    }
                } else if (fullStereo) {
                    // Production: install the frame-driver hook and arm the stereo pump.
                    // Pass 2 only runs on frames where the game stashed a second eye, so
                    // this is inert until an OpenXR session is live and requesting stereo.
                    QueueTasksProbeInstall(true);
                    if (g_origQueueTasks) {
                        g_pumpMode = 2;
                        LOGI("[stereo] frame pump ARMED: draw-only second pass per stereo frame");
                    } else {
                        LOGW("[stereo] frame pump unavailable (hook failed); full stereo cannot run");
                    }
                } else if (altStereo) {
                    QueueTasksProbeInstall(true);
                    if (g_origQueueTasks)
                        LOGI("[stereo] alternate-eye: frame hook installed (eye reports at frame end, "
                             "no second pass)");
                    else
                        LOGW("[stereo] alternate-eye: frame hook unavailable; no eye reports, stays flat");
                }
                if (Cfg().m2TaskProbe) {
                    RegisterExperiment();
                    QueueTasksProbeInstall();
                    DumpTaskLists();
                    ContentProbeShots();
                    DoublePumpTest();
                    ToggleExperiment();
                }
                return;
            }
            Sleep(10000);
        }
        LOGW("[m2] task-group table never appeared; giving up");
    }).detach();
}

void FrameProbeOnPresent(const Dx12Frame& f) {
    if (!g_csOn.load(std::memory_order_relaxed)) return;
    // Pack this present's pass + pitch state into the tag so it travels through the sampler's
    // latency and comes back paired with the luma of the frame it was actually captured on.
    const uint32_t tagIn = (uint32_t)FramePumpCurrentPass() |
                           (g_probePitchOn.load(std::memory_order_relaxed) ? 0x100u : 0u);
    uint32_t tagOut = 0;
    float luma = 0.f;
    if (!Dx12SampleBackbuffer(f.queue, f.backBuffer, tagIn, &tagOut, &luma)) return;
    const int i = g_csN.fetch_add(1, std::memory_order_relaxed);
    if (i < kMaxCSamples)
        g_cs[i] = {(uint8_t)(tagOut & 0xff), (uint8_t)((tagOut & 0x100) ? 1 : 0), luma};
}

void FrameProbeSetPass2Keep(const char* spec) { SetPass2Override(spec ? spec : ""); }

uint8_t* FramePumpStepsDstWatchTarget(uint32_t* outLen) {
    if (g_stepsDstN.load(std::memory_order_acquire) < 1) return nullptr;
    // The staged frame-state block in the task-runner: steps at +0xdc, flag at +0xd8; the
    // dt float the dispatch hands to tasks should be staged nearby. 0x40 bytes of context.
    if (outLen) *outLen = 0x40;
    return g_stepsDst[0] + 0xc0;
}

uint8_t* FramePumpDtWatchTarget(uint32_t* outLen) {
    uint8_t* s = g_dtStruct.load(std::memory_order_acquire);
    if (!s) return nullptr;
    // steps int at +0x18, dt float at +0x1c, cached qword at +0x20: whoever COPIES the
    // frame dt downstream reads in here - the census names the copy site.
    if (outLen) *outLen = 0xC;
    return s + 0x18;
}

bool FramePumpReady() { return g_origQueueTasks != nullptr && g_pumpMode == 2; }
void FramePumpNotePass2Present() { g_p2PresentTick.store(GetTickCount64(), std::memory_order_relaxed); }
bool FramePumpSafeSetActive() { return g_safeSetActive.load(std::memory_order_relaxed); }
int  FramePumpCurrentPass() { return g_curPass.load(std::memory_order_acquire); }
unsigned long FramePumpFrameThread() { return g_qtThread; }

int FramePumpCurrentTaskGroup() {
    void* task = g_table.owner;
    if (!task) return -2;
    __try {
        uint8_t* runner = *(uint8_t**)((uint8_t*)task + 0x78);
        return runner ? *(int32_t*)(runner + 0x18) : -3;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return -4; }
}
long FramePumpTakePasses() { return InterlockedExchange(&g_pumpPasses, 0); }
PumpPassTimes FramePumpTakePassTimes() {
    PumpPassTimes t;
    for (int i = 0; i < 2; ++i) {
        const long n = g_passN[i].exchange(0);
        const long long us = g_passUs[i].exchange(0), mx = g_passMaxUs[i].exchange(0);
        if (n > 0) { t.msAvg[i] = (float)us / 1000.f / (float)n; t.msMax[i] = (float)mx / 1000.f; }
    }
    return t;
}
float FramePumpLastMs() { return g_pumpLastMs.load(std::memory_order_relaxed); }
void FramePumpSetGpuQueue(ID3D12CommandQueue* q) { g_gpuQueue.store(q, std::memory_order_release); }
float FramePumpGpuSyncLastMs() { return g_syncLastMs.load(std::memory_order_relaxed); }
}
